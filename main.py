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

        # Define schema for output
        response_schema = ResponseSchema(
            name="answer",
            description="The concise answer to the user's question about the page"
        )
        parser = StructuredOutputParser.from_response_schemas([response_schema])

        # Modify the prompt with format instructions
        format_instructions = parser.get_format_instructions()

        # Final question prompt to send to the LLM
        formatted_question = f"{question}\n\n{format_instructions}"

        config = {
            "llm": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": "gpt-3.5-turbo",
                "temperature": 0,
            },
            "graph_config": {
                "browser_args": ["--no-sandbox", "--disable-dev-shm-usage"]
            },
            "prompt_type": "simple",
            "verbose": True,
        }

        graph = SmartScraperGraph(prompt=formatted_question, source=url, config=config)
        result = await run_in_threadpool(graph.run)

        # Parse and return the structured response
        structured_result = parser.parse(result)
        return {"result": structured_result}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
