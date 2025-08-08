from fastapi import FastAPI
from pydantic import BaseModel
from scrapegraphai.graphs import SmartScraperGraph
import os
import json

app = FastAPI()

class ScrapeRequest(BaseModel):
    url: str
    question: str

# Local definition of convert_to_openai_message to replace missing import
def convert_to_openai_message(prompt):
    return [{"role": "user", "content": prompt}]

@app.post("/")
async def scrape_website(request: ScrapeRequest):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"error": "Missing OPENAI_API_KEY environment variable"}

    try:
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

        # ✅ Use synchronous run (this is what worked before)
        result = graph.run()

        # ✅ Safely parse the output
        output_parser = graph.output_parser
        output = output_parser.parse(result if isinstance(result, str) else json.dumps(result))

        return {"summary": output}

    except Exception as e:
        return {
            "error": "Output parsing failed",
            "raw_output": result if 'result' in locals() else None,
            "details": str(e)
        }

