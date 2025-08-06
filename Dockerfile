FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install scrapegraphai[burr]

# Install runtime dependencies
RUN pip install fastapi uvicorn openai playwright
RUN python3 -m playwright install

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
