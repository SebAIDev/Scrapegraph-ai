# Playwright base with all browser deps preinstalled
FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    "scrapegraphai[burr]" \
    fastapi uvicorn openai requests beautifulsoup4 lxml charset-normalizer reportlab

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
