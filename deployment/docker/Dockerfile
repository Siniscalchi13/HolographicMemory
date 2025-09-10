FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY services/holographic-memory/api/requirements.txt ./api_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy source code
COPY . .

# Build GPU components (if supported)
RUN cd services/holographic-memory/core/native/holographic/build && \
    make clean && make || echo "GPU build failed, continuing with CPU-only"

# Expose ports
EXPOSE 8081 8082 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Start command
CMD ["python3.13", "main.py"]
