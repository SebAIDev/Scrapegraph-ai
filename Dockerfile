# Use Python slim base
FROM python:3.11-slim

# Install system dependencies for Playwright and Chromium
RUN apt-get update && \
    apt-get install -y wget gnupg unzip curl && \
    apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 libxshmfence1 libasound2 libxtst6 libxrandr2 libgtk-3-0 libdrm2 libgbm1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install ScrapeGraphAI with API and Playwright extras
RUN pip install --upgrade pip
RUN pip install "scrapegraphai[api,playwright]"

# Install Playwright dependencies and Chromium browser
RUN python3 -m playwright install --with-deps

# Expose port for the API server
EXPOSE 8000

# Start the ScrapeGraphAI FastAPI server
CMD ["python", "-m", "scrapegraphai.api"]
