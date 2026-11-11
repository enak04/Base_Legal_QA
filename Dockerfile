# Use a lightweight official Python runtime
FROM python:3.10-slim

# Set environment variables to prevent writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first to utilize Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run the model caching script during the build phase so the model is baked into the image
RUN python scratch/download_model.py

# Expose the port (Cloud Run defaults to 8080)
EXPOSE 8080

# Run the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
