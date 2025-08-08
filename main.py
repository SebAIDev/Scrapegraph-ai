from fastapi import FastAPI
from pydantic import BaseModel
from scrapegraphai.graphs import SmartScraperGraph
import os
import json

app = FastAPI()

# Define the request schema
class ScrapeRequest(BaseModel):
    url: str
    question: str

# Local definition of convert_to_openai_message to replace missing import
def convert_to_openai_message(prompt):
    """
    Convert a plain string prompt to OpenAI chat message format.
    """
    return [{"role": "user", "content": prompt}]

@app.post("/")
async def scrape_website(request: ScrapeRequest):
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"error": "Missing OPENAI_API_KEY environment variable"}

    try:
        # Create the SmartScraperGraph with configuration
        graph_config = {
            "llm": {
                "model": "gpt-3.5-turbo",
                "api_key": openai_api_key,
            },
            "verbose": True,
        }

        graph = SmartScraperGraph(
            prompt=request.question,
            source=request.url,
            config=graph_config,
        )

        # Use async run method
        result = await graph.arun()

        # Safely handle both string and dict result outputs
        output_parser = graph.output_parser
        output = output_parser.parse(result if isinstance(result, str) else json.dumps(result))

        return {"s

