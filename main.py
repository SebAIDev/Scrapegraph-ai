from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import convert_to_openai_messages
import os

from flask import Flask, request, jsonify

# Robust __file__ fallback for environments like Docker or Jupyter
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

prompt_file_path = os.path.join(current_dir, "prompt.txt")
with open(prompt_file_path, "r") as file:
    prompt_text = file.read()

app = Flask(__name__)

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.json
    if not data or "url" not in data or "question" not in data:
        return jsonify({"error": "Missing 'url' or 'question' in request body"}), 400

    graph_config = {
        "llm": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo",
        },
        "embeddings": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
        },
        "verbose": True,
    }

    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt_text,
        source=data["url"],
        config=graph_config
    )

    try:
        result = smart_scraper_graph.run(data["question"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
