FROM python:3.11-slim

# Install system dependencies required by Playwright + Git
RUN apt-get update && apt-get install -y \
    git curl wget gnupg ca-certificates fonts-liberation libasound2 \
    libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 libdrm2 \
    libgbm1 libgtk-3-0 libnspr4 libnss3 libx11-xcb1 libxcomposite1 \
    libxdamage1 libxrandr2 xdg-utils libxkbcommon0 libxshmfence1 \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy all files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# ✅ Install ScrapeGraphAI from GitHub (requires git)
RUN pip install --no-cache-dir --upgrade git+https://github.com/michellechandra/scrapegraphai.git

# Install Python dependencies
RUN pip install fastapi uvicorn openai playwright

# Install Playwright browser dependencies
RUN python3 -m playwright install

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
