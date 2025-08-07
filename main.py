from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
import uvicorn

app = FastAPI()

# Define what structured output you want from the LLM
response_schemas = [
    ResponseSchema(name="summary", description="Concise summary of the web page content.")
]

# Create the parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

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
                "temperature": 0.3,
            },
            "graph_config": {
                "browser_args": ["--no-sandbox", "--disable-dev-shm-usage"]
            },
            "prompt_type": "simple",  # Ensures cleaner output format
            "verbose": True,
        }

        graph = SmartScraperGraph(prompt=question + "\n" + parser.get_format_instructions(), source=url, config=config)

        # Run graph and print raw output for debugging
        result = await run_in_threadpool(graph.run)
        print("RAW RESULT FROM GRAPH:", result)

        # Try to extract a string from result
        if isinstance(result, dict):
            raw_output = result.get("result") or result.get("output") or str(result)
        else:
            raw_output = str(result)

        # Parse the structured result
        structured_result = parser.parse(raw_output)

        return {"result": structured_result}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
