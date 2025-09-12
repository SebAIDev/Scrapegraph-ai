# Playwright base with all browser deps preinstalled
FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
# Added: reportlab for PDF generation
RUN pip install --no-cache-dir \
    "scrapegraphai[burr]" \
    fastapi uvicorn openai requests beautifulsoup4 lxml charset-normalizer reportlab

# Browsers already installed in this base image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
