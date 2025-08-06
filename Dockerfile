FROM python:3.11-slim

# Install system dependencies & Playwright libs
RUN apt-get update && apt-get install -y curl git \
    libglib2.0-0 libnspr4 libnss3 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2 && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy everything
COPY . .

# Upgrade pip and install ScrapeGraphAI with burr + latest version
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade "scrapegraphai[burr]>=0.1.28"

# Install runtime dependencies
RUN pip install fastapi uvicorn openai playwright
RUN python3 -m playwright install

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
