FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Install ScrapeGraphAI and browser dependencies
RUN pip install --no-cache-dir scrapegraphai[burr]

RUN python3 -m playwright install-deps && \
    python3 -m playwright install

# Expose the port ScrapeGraphAI uses (default: 8000)
EXPOSE 8000

# Start the API server
CMD ["python", "-m", "scrapegraphai.api"]
