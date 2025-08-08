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

# ✅ Install ScrapeGraphAI from your public GitHub repo
RUN pip install --no-cache-dir --upgrade \
    git+https://github.com/SebAIDev/Scrapegraph-ai.git

# Copy the rest of your app (optional if you're adding other files)
COPY . /app

# Expose port (adjust if your app uses a different one)
EXPOSE 8000

# Start your app (update this line based on your actual entry point)
CMD ["python", "main.py"]
