FROM python:3.11-slim

WORKDIR /app

# Install CPU PyTorch before the rest to avoid pulling the CUDA variant
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

# models/ is volume-mounted at runtime — not baked into the image
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
