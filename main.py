from fastapi import FastAPI, Request
from pydantic import BaseModel
from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import convert_graph_to_dag
import uvicorn
import os

# Define input model
class ScrapeRequest(BaseModel):
    url: str
    question: str

app = FastAPI()

@app.post("/")
async def scrape(request: ScrapeRequest):
    graph_config = {
        "llm": {
            "model": "ollama/llama3",
            "temperature": 0,
        },
        "embeddings": {
            "model": "all-MiniLM-L6-v2",
        },
        "verbose": True,
    }

    prompt = (
        "Extract relevant information from the website to answer the question: "
        f"{request.question}. Only use the context of the website: {request.url}"
    )

    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=request.url,
        config=graph_config,
    )

    dag = convert_graph_to_dag(smart_scraper_graph)

    try:
        raw_output = dag.execute()

        # Handle both dict and string responses robustly
        try:
            result = {"summary": raw_output.get("summary", str(raw_output))}
        except AttributeError:
            result = {"summary": str(raw_output).strip()}

        return result

    except Exception as e:
        return {"error": "Output parsing failed", "raw_output": raw_output, "details": str(e)}

# Optional: uncomment if running locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
