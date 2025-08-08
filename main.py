from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from scrapegraphai.graphs import SmartScraperGraph
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
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

        # Define expected output format
        response_schema = ResponseSchema(
            name="summary",
            description="A clear, human-readable summary of the website"
        )
        parser = StructuredOutputParser.from_response_schemas([response_schema])
        format_instructions = parser.get_format_instructions()

        # Rebuild prompt with format guidance
        prompt = f"{question}\n\n{format_instructions}"

        config = {
            "llm": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
            },
            "graph_config": {
                "browser_args": ["--no-sandbox", "--disable-dev-shm-usage"]
            },
            "prompt_type": "simple",
            "verbose": True,
        }

        graph = SmartScraperGraph(prompt=prompt, source=url, config=config)
        raw_output = await run_in_threadpool(graph.run)

        # Try parsing the output into JSON format
        try:
            parsed_output = parser.parse(raw_output)
        except Exception as e:
            return {
                "error": "Output parsing failed",
                "raw_output": raw_output,
                "details": str(e)
            }

        return {"result": parsed_output}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
