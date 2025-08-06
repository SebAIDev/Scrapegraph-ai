# Use Python slim base
FROM python:3.11-slim

# Install system dependencies required by Playwright and Chromium
RUN apt-get update && \
    apt-get install -y wget gnupg unzip curl && \
    apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 libxshmfence1 libasound2 libxtst6 libxrandr2 libgtk-3-0 libdrm2 libgbm1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install ScrapeGraphAI with all extras
RUN pip install --upgrade pip
RUN pip install "scrapegraphai[api,playwright]"

# Install Playwright browsers
RUN python3 -m playwright install --with-deps

# Expose the API port
EXPOSE 8000

# Proper launch command
CMD ["python", "-m", "scrapegraphai.api"]
