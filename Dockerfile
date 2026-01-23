# Base Image: Lightweight Python
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies (gcc needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to cache dependencies)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final Stage
FROM python:3.9-slim

WORKDIR /app

# Copy installed libraries from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Set Python path to find 'src' modules
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Default command: Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]