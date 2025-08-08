from fastapi import FastAPI
from pydantic import BaseModel
from scrapegraphai.graphs import SmartScraperGraph
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Define the request schema
class ScrapeRequest(BaseModel):
    url: str
    question: str

# Local definition of convert_to_openai_message to replace missing import
def convert_to_openai_message(prompt):
    return [{"role": "user", "content": prompt}]

# Run sync code in a threadpool
executor = ThreadPoolExecutor()

def run_graph_sync(prompt, url, config):
    graph = SmartScraperGraph(prompt=prompt, source=url, config=config)
    result = graph.run()
    output_parser = graph.output_parser
    return output_parser.parse(result if isinstance(result, str) else json.dumps(result))

@app.post("/")
async def scrape_website(request: ScrapeRequest):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"error": "Missing OPENAI_API_KEY environment variable"}

    graph_config = {
        "llm": {
            "model": "gpt-3.5-turbo",
            "api_key": openai_api_key,
        },
        "verbose": True,
    }

    try:
        # Run the sync graph in a threadpool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            executor, run_graph_sync, request.question, request.url, graph_config
        )
        return {"summary": result}

    except Exception as e:
        return {
            "error": "Output parsing failed",
            "raw_output": None,
            "details": str(e)
        }
