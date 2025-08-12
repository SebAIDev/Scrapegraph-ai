from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
import os
import uvicorn

# NEW: tiny helpers for discovery
import re, time, requests
from urllib.parse import urljoin, urlparse
from collections import deque
from bs4 import BeautifulSoup

SKIP_EXT = re.compile(r"\.(pdf|jpg|jpeg|png|gif|svg|webp|zip|mp4|mp3|avi|css|js|ico)$", re.I)

def _same_domain(u, root):
    try:
        return urlparse(u).netloc == urlparse(root).netloc
    except Exception:
        return False

def _normalize(u):
    p = urlparse(u)
    return p._replace(fragment="").geturl()

def _get_sitemap_urls(start_url, limit=80, timeout=10):
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

def discover_urls(start_url, max_pages=60, max_depth=3, delay=0.5):
    """Polite same-domain crawl (sitemap → fallback BFS)."""
    found, seen = [], set()
    q = deque()

    seeds = _get_sitemap_urls(start_url, limit=max_pages) or [start_url]
    for u in seeds:
        if _same_domain(u, start_url) and not SKIP_EXT.search(u or ""):
            nu = _normalize(u)
            if nu not in seen:
                seen.add(nu)
                q.append((nu, 0))

    while q and len(found) < max_pages:
        url, depth = q.popleft()
        found.append(url)
        if depth >= max_depth:
            continue
        try:
            resp = requests.get(url, timeout=12, headers={"User-Agent": "CoPilotBot/1.0"})
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.select("a[href]"):
                href = urljoin(url, a.get("href"))
                if not _same_domain(href, start_url): 
                    continue
                if SKIP_EXT.search(href or ""):
                    continue
                nu = _normalize(href)
                if nu not in seen:
                    seen.add(nu)
                    q.append((nu, depth + 1))
            time.sleep(delay)  # be polite / avoid hammering
        except Exception:
            continue

    junk = ("login", "cart", "checkout", "wp-json", "/tag/", "/search?")
    return [u for u in found if not any(j in u for j in junk)]

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ScrapeGraphAI is alive"}

def run_smart_scraper(url: str, question: str):
    """Your existing single-page scrape, extracted into a function."""
    config = {
        "llm": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo",
            "temperature": 0,
        },
        "graph_config": {"browser_args": ["--no-sandbox", "--disable-dev-shm-usage"]},
        "prompt_type": "simple",
        "verbose": True,
    }
    graph = SmartScraperGraph(prompt=question, source=url, config=config)
    return graph.run()  # will be called via run_in_threadpool

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
            return {"error": "Missing 'url' or 'question'"}

        # === No-crawl (current behavior) ===
        if not crawl:
            raw_output = await run_in_threadpool(run_smart_scraper, url, question)
            result = raw_output if isinstance(raw_output, dict) else {"summary": str(raw_output).strip()}
            return {"result": result}

        # === Crawl mode ===
        urls = discover_urls(url, max_pages=max_pages, max_depth=max_depth)
        pages, emails = [], set()

        for u in urls:
            raw = await run_in_threadpool(run_smart_scraper, u, question)
            page = raw if isinstance(raw, dict) else {"summary": str(raw).strip()}
            page["url"] = u
            pages.append(page)
            # naive email harvest if present in page dict
            for e in page.get("emails", []):
                emails.add(str(e).lower())

        # Reduce/aggregate (simple join; you can replace with an LLM later)
        overview = "\n\n".join(p.get("summary", "") for p in pages[:15])

        return {
            "domain": urlparse(url).netloc,
            "stats": {"pages_crawled": len(pages), "max_pages": max_pages, "max_depth": max_depth},
            "entities": {"emails": sorted(emails)},
            "pages": pages,
            "overview": {"summary": overview}
        }

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
