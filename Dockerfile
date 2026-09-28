FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV (used internally by pytorch-grad-cam)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch before the rest to avoid pulling the CUDA variant
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY static/ ./static/

# Run as an unprivileged user. models/ must be writable by it because the
# weights are downloaded from HF Hub on first start if not volume-mounted.
RUN useradd --create-home --uid 1000 appuser     && mkdir -p models     && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Readiness, not liveness: the container is only "healthy" once the model is
# loaded. start-period covers the first-boot weight download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready', timeout=4)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
