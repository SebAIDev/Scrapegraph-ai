# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt first for caching
COPY requirements.txt /app/

# Install dependencies from requirements.txt (including FastAPI and Scrapegraph-ai)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of your app files
COPY . /app

# Expose port
EXPOSE 8000

# Use uvicorn to serve the FastAPI app (recommended)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
