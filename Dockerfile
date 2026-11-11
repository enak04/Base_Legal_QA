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

# Download InLegalBERT and bake an int8-quantized copy into the image
# (models/inlegalbert_qint8.pt, ~180 MB). At runtime the server loads this file
# directly, so the ~440 MB fp32 model is never in memory: loading and quantizing
# it at startup is what exceeded Render's 512 MB limit.
RUN python scratch/download_model.py && python -m inference.bert_loader

# Limit glibc malloc arenas: each worker thread otherwise gets its own arena,
# which inflates RSS on small (512 MB) instances
ENV MALLOC_ARENA_MAX=2 \
    TOKENIZERS_PARALLELISM=false

# Expose the port (Cloud Run defaults to 8080)
EXPOSE 8080

# Run the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
