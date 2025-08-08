from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers.structure import ResponseSchema
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

        # ScrapeGraphAI config
        config = {
            "llm": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": "gpt-3.5-turbo",
                "temperature": 0,
            },
            "graph_config": {
                "browser_args": ["--no-sandbox", "--disable-dev-shm-usage"]
            },
            "prompt_type": "simple",  # makes output easier to parse
            "verbose": True,
        }

        # Define expected output structure
        response_schema = ResponseSchema(name="summary", description="A concise summary of the website's content")
        parser = StructuredOutputParser.from_response_schemas([response_schema])

        # Create the scraping graph
        graph = SmartScraperGraph(prompt=question, source=url, config=config)

        # Run scraper in thread to avoid blocking
        raw_output = await run_in_threadpool(graph.run)

        # Robustly handle string or dict
        if isinstance(raw_output, str):
            parsed_output = parser.parse(raw_output)
        else:
            parsed_output = raw_output

        return {"result": parsed_output}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
