# Use the official Playwright image that already includes all browser deps
FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
# Install deps (you can pin scrapegraphai if you want stability)
RUN pip install --no-cache-dir \
    "scrapegraphai[burr]" \
    fastapi uvicorn openai requests beautifulsoup4 lxml charset-normalizer

# Browsers are preinstalled in this base image
# If you want to be explicit, you could also run:
# RUN python -m playwright install --with-deps

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
