# Dockerfile for ScrollSafe Backend API (Instance 1)
# FastAPI application serving the main API

FROM python:3.12-slim

# Install system dependencies for video processing (deep scan workers need this)
# Also install build dependencies for Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    gcc \
    g++ \
    libmagic1 \
    cargo \
    rustc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
# Replace python-magic-bin (Windows-only) with python-magic for Linux
RUN grep -v "python-magic-bin" requirements.txt > requirements-docker.txt && \
    echo "python-magic==0.4.27" >> requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip install --no-cache-dir psycopg[binary] psycopg-pool

# Copy application code
COPY *.py ./
COPY deep_scan ./deep_scan
COPY services ./services

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command - run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-max-request-size", "64"]
