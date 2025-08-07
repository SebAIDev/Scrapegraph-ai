from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
import uvicorn

app = FastAPI()

# Define structured output format
response_schemas = [
    ResponseSchema(
        name="summary",
        description="A concise, plain-English summary of what this website or page is about"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

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
            "prompt_type": "simple",
            "verbose": True,
        }

        # Append format instructions to prompt for reliable JSON output
        full_prompt = f"{question}\n\n{format_instructions}"

        graph = SmartScraperGraph(prompt=full_prompt, source=url, config=config)

        # Run graph in thread
        raw_result = await run_in_threadpool(graph.run)
        print("RAW RESULT FROM GRAPH:", raw_result)

        # Extract raw string from possible dict
        if isinstance(raw_result, dict):
            output_text = raw_result.get("result") or raw_result.get("output") or str(raw_result)
        else:
            output_text = str(raw_result)

        # Clean and parse
        structured_result = parser.parse(output_text)

        return {"result": structured_result}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
