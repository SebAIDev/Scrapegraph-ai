from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
import os
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ScrapeGraphAI is alive"}

@app.post("/scrape")
async def scrape(request: Request):
    try:
        body = await request.json()
        url = body.get("url")
        question = body.get("question")

        if not url or not question:
            return {"error": "Missing 'url' or 'question'"}

       config = {
    "llm": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-3.5-turbo",
        "temperature": 0,
    },
    "graph_config": {
        "browser_args": ["--no-sandbox", "--disable-dev-shm-usage"]
    },
    "verbose": True,
       }

        graph = SmartScraperGraph(prompt=question, source=url, config=config)

        # Run the blocking .run() method in a thread to avoid blocking event loop
        result = await run_in_threadpool(graph.run)

        return {"result": result}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
