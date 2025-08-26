from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
import os
import uvicorn

# --- Crawl helpers ---
import re, time, requests
from urllib.parse import urljoin, urlparse
from collections import deque
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes  # robust encoding

# --- OpenAI (LLM aggregation) ---
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def chat_complete(model: str, prompt: str) -> str:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise company analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
except Exception:
    import openai as _openai_legacy
    _openai_legacy.api_key = os.environ.get("OPENAI_API_KEY")
    def chat_complete(model: str, prompt: str) -> str:
        resp = _openai_legacy.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise company analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return resp["choices"][0]["message"]["content"].strip()

SKIP_EXT = re.compile(r"\.(pdf|jpg|jpeg|png|gif|svg|webp|zip|mp4|mp3|avi|css|js|ico)$", re.I)

# === Short, fast per-page prompt (used during crawl) ===
PER_PAGE_PROMPT = (
    "In 3–5 bullets, extract: (1) what this page says the company does, "
    "(2) key products/services mentioned, (3) any proof points (clients, "
    "certifications, awards), (4) contact/location if present. Keep under 80 words. No fluff."
)

# ---------- Robust decoding & sanitization ----------
SMART_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-", "\u00A0": " "
}
def _sanitize_text(s: str) -> str:
    for k, v in SMART_MAP.items():
        s = s.replace(k, v)
    return " ".join(s.split())

def _decode_html_bytes(raw: bytes, declared: str | None) -> str:
    # Prefer server-declared encoding if valid
    if declared:
        try:
            return raw.decode(declared, errors="strict")
        except UnicodeDecodeError:
            pass
    # Detect best encoding (handles cp1252/latin-1, etc.)
    guess = from_bytes(raw).best()
    if guess:
        return str(guess)
    # Fallbacks
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")

def _fetch_page_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers={"User-Agent":"CoPilotBot/1.0"}, timeout=timeout)
    html = _decode_html_bytes(r.content, r.encoding)
    soup = BeautifulSoup(html, "lxml")
    return _sanitize_text(soup.get_text(" ", strip=True))

# ---------- URL helpers ----------
def _same_domain(u: str, root: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(root).netloc
    except Exception:
        return False

def _normalize(u: str) -> str:
    p = urlparse(u)
    return p._replace(fragment="").geturl()

def _get_sitemap_urls(start_url: str, limit: int = 80, timeout: int = 10):
    base = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
    for cand in (urljoin(base, "/sitemap.xml"), urljoin(base, "/sitemap_index.xml")):
        try:
            r = requests.get(cand, timeout=timeout, headers={"User-Agent": "CoPilotBot/1.0"})
            if r.status_code == 200 and "<urlset" in r.text:
                soup = BeautifulSoup(r.text, "xml")
                urls = [loc.text.strip() for loc in soup.find_all("loc")]
                return urls[:limit]
        except Exception:
            pass
    return []

def discover_urls(
    start_url: str,
    max_pages: int = 60,
    max_depth: int = 3,
    delay: float = 0.5,
    max_runtime_sec: int = 240
):
    """Polite same-domain crawl with sitemap seeding, priority pages, and a hard 4-min cap."""
    began = time.time()
    found, seen = [], set()
    q = deque()

    priority_paths = [
        "/about", "/company", "/team", "/leadership",
        "/services", "/products", "/solutions",
        "/pricing", "/contact",
        "/case-studies", "/clients", "/press", "/news", "/blog"
    ]
    base = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"

    seeds = _get_sitemap_urls(start_url, limit=max_pages)
    if not seeds:
        seeds = [start_url] + [urljoin(base, p) for p in priority_paths]

    uniq = []
    for s in seeds:
        nu = _normalize(s)
        if _same_domain(nu, start_url) and not SKIP_EXT.search(nu) and nu not in uniq:
            uniq.append(nu)

    for u in uniq[:max_pages]:
        seen.add(u)
        q.append((u, 0))

    while q and len(found) < max_pages:
        if time.time() - began > max_runtime_sec:
            break
        url, depth = q.popleft()
        found.append(url)
        if depth >= max_depth:
            continue
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "CoPilotBot/1.0"})
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            html = _decode_html_bytes(resp.content, resp.encoding)  # robust decode
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                href = urljoin(url, a.get("href"))
                if not _same_domain(href, start_url) or SKIP_EXT.search(href or ""):
                    continue
                nu = _normalize(href)
                if nu not in seen:
                    seen.add(nu)
                    q.append((nu, depth + 1))
            time.sleep(delay)
        except Exception:
            continue

    junk = ("login", "cart", "checkout", "wp-json", "/tag/", "/search?")
    return [u for u in found if not any(j in u for j in junk)]

# --- App ---
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ScrapeGraphAI is alive"}

# === SmartScraper with resilient fallback ===
def run_smart_scraper(url: str, question: str, page_mode: bool = False):
    """
    Try SmartScraperGraph first (uses Playwright).
    If it raises (decode error / page crash), fall back to requests+BS4 text and summarize via OpenAI.
    """
    config = {
        "llm": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo",  # consider "gpt-4o-mini" for better quality/similar speed
            "temperature": 0,
        },
        "graph_config": {"browser_args": [
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--single-process"
        ]},
        "prompt_type": "simple",
        "verbose": True,
    }
    prompt = PER_PAGE_PROMPT if page_mode else question

    # 1) Try SmartScraperGraph
    try:
        graph = SmartScraperGraph(prompt=prompt, source=url, config=config)
        out = graph.run()
        if isinstance(out, dict):
            text = out.get("summary") or out.get("content") or str(out)
            return {"summary": str(text).strip()}
        else:
            return {"summary": str(out).strip()}
    except Exception as e:
        # 2) Fallback: manual fetch + local LLM summarization (avoids UTF-8 & Playwright crashes)
        try:
            text = _fetch_page_text(url)[:12000]  # cap for speed
            fallback_prompt = (
                f"Use only the following text from {url} to answer.\n\n"
                f"TEXT:\n{text}\n\nQUESTION:\n{prompt}"
            )
            ans = chat_complete("gpt-3.5-turbo", fallback_prompt)
            return {"summary": ans.strip()}
        except Exception as inner:
            return {"summary": f"(Partial) Unable to parse page automatically. Reason: {e} | Fallback failed: {inner}"}

def _truncate(text: str, max_chars: int = 12000) -> str:
    return (text or "")[:max_chars]

def build_overview_with_llm(pages: list, model: str, user_question: str) -> str:
    """Aggregate per-page summaries into one polished output using YOUR sales prompt."""
    items = [p for p in pages if p.get("summary")][:20]
    combined = "\n\n".join(
        f"Page: {p.get('url','')}\nSummary: {p.get('summary','')}" for p in items
    )
    combined = _truncate(combined, 12000)
    guidance = (
        "Please produce a clear, concise, de-duplicated answer using the page summaries. "
        "Avoid hallucinations; omit details that aren’t present."
    )
    prompt = f"{user_question}\n\n{guidance}\n\nPage summaries:\n{combined}"
    return chat_complete(model, prompt)

@app.post("/scrape")
async def scrape(request: Request):
    try:
        body = await request.json()
        url = body.get("url")
        question = body.get("question")
        crawl = bool(body.get("crawl", False))
        max_pages = int(body.get("max_pages", 60))
        max_depth = int(body.get("max_depth", 3))

        if not url or not question:
            text = "Missing 'url' or 'question'"
            return {"overview": {"summary": text}, "result": {"summary": text, "content": text}}

        # --- No crawl: single page ---
        if not crawl:
            single = await run_in_threadpool(run_smart_scraper, url, question, False)
            text = single.get("summary", "")
            return {"overview": {"summary": text}, "result": {"summary": text, "content": text}}

        # --- Crawl: fast per-page summaries, then merge with sales prompt ---
        urls = discover_urls(url, max_pages=max_pages, max_depth=max_depth, max_runtime_sec=240)
        pages, emails = [], set()

        for u in urls:
            page = await run_in_threadpool(run_smart_scraper, u, question, True)  # page_mode=True
            page["url"] = u
            pages.append(page)
            for e in page.get("emails", []):
                emails.add(str(e).lower())

            # Early stop if we already hit key sections and have ~10 pages
            if len(pages) >= 10 and any(k in u for k in ("/about","/services","/products","/pricing","/clients","/contact")):
                break

        polished_overview = await run_in_threadpool(
            build_overview_with_llm, pages, "gpt-3.5-turbo", question
        )

        payload = {
            "domain": urlparse(url).netloc,
            "stats": {"pages_crawled": len(pages), "max_pages": max_pages, "max_depth": max_depth},
            "entities": {"emails": sorted(emails)},
            "pages": pages,
            "overview": {"summary": polished_overview},
            # Backward-compat mirrors
            "result": {"summary": polished_overview, "content": polished_overview}
        }
        return payload

    except Exception as e:
        text = f"Internal Server Error: {e}"
        return {"overview": {"summary": text}, "result": {"summary": text, "content": text}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
