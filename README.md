# Medical Imaging Diagnosis API

![CI](https://github.com/REZZOUGAICHA/Medical-Imaging-Diagnosis-API/actions/workflows/ci.yml/badge.svg)

A production-ready REST API for **diabetic retinopathy severity classification** from retinal fundus images. Returns a diagnosis, confidence score, a Grad-CAM heatmap, and an AI-generated clinical explanation powered by Flan-T5-Large.

Built with EfficientNet-B4 fine-tuned on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset. Ships with a medical web UI, full monitoring via Prometheus and Grafana, 18 unit tests, and a GitHub Actions CI pipeline.

> **Disclaimer:** This tool is for research purposes only and does not constitute medical advice.

**Live demo:** https://medical-imaging-diagnosis-api-production.up.railway.app

---

## Architecture

```
Client (browser or API)
        │
        ▼
┌──────────────────────────────────────────┐
│              FastAPI  :8000              │
│                                          │
│  GET  /          → Medical web UI        │
│  POST /predict   → EfficientNet-B4       │    ┌──────────────────────┐
│                    + Grad-CAM heatmap ───┼───▶│  HF Hub              │
│  POST /explain   → Flan-T5-Large NLP     │    │  best_model.pth      │
│  GET  /metrics   → Prometheus scrape     │    └──────────────────────┘
└──────────────────────────────────────────┘
        │  scrape /metrics every 15s
        ▼
┌───────────────────┐      query      ┌──────────────────────┐
│   Prometheus :9090│ ──────────────▶ │   Grafana  :3000     │
│   (time-series DB)│                 │   (dashboards)       │
└───────────────────┘                 └──────────────────────┘
```

---

## Features

- **5-class DR classification** — No DR / Mild / Moderate / Severe / Proliferative
- **Grad-CAM explainability** — heatmap overlay showing which retinal regions drove the prediction
- **AI clinical explanation** — Flan-T5-Large generates a natural language report for each prediction
- **Medical web UI** — drag-and-drop upload, color-coded severity, probability bars, side-by-side image comparison
- **Prometheus + Grafana monitoring** — latency, request counts, per-class prediction counts, confidence distribution
- **18 unit tests** — model architecture, inference logic, API endpoints
- **GitHub Actions CI** — tests run automatically on every push
- **One-command Docker deployment** — `docker-compose up --build`

---

## Quick Start

**Requirements:** Docker and Docker Compose installed.

### 1. Clone and start

```bash
git clone https://github.com/REZZOUGAICHA/Medical-Imaging-Diagnosis-API.git
cd Medical-Imaging-Diagnosis-API
docker-compose up --build
```

Model weights are downloaded automatically from [Hugging Face Hub](https://huggingface.co/aicharzg/diabetic-retinopathy-efficientnet-b4) on first startup.

This starts three services:

| Service    | URL                        | Purpose                         |
|------------|----------------------------|---------------------------------|
| API + UI   | http://localhost:8000      | Web UI and inference endpoints  |
| Prometheus | http://localhost:9090      | Metrics storage                 |
| Grafana    | http://localhost:3000      | Monitoring dashboard            |

Grafana login: `admin / admin`

### 2. Optional — AI explanations

To enable the Flan-T5 clinical explanation feature, set your Hugging Face token:

```bash
# .env (never commit this file)
HF_TOKEN=your_hf_token_here
```

Then restart: `docker-compose up`.

---

## Web UI

Open **http://localhost:8000** for the interactive medical interface:

1. Drag and drop a retinal fundus image (JPEG or PNG)
2. The model returns predicted DR severity + confidence + probability distribution
3. Grad-CAM heatmap shows which retinal regions drove the prediction
4. Click **Generate AI Clinical Explanation** for a Flan-T5 natural language report

---

## API Reference

### `GET /health`
Returns server and device status.

```json
{
  "status": "healthy",
  "model": "EfficientNet-B4",
  "device": "cpu"
}
```

### `GET /model-info`
Returns model metadata and class labels.

### `POST /predict`
Accepts a retinal fundus image (JPEG or PNG) and returns a diagnosis.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@retinal_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "predicted_class": 2,
  "class_name": "Moderate DR",
  "description": "Moderate non-proliferative DR. Medical review advised.",
  "confidence": 0.8743,
  "probabilities": {
    "No Diabetic Retinopathy": 0.04,
    "Mild DR": 0.06,
    "Moderate DR": 0.87,
    "Severe DR": 0.02,
    "Proliferative DR": 0.01
  },
  "gradcam_heatmap": "<base64-encoded PNG>",
  "disclaimer": "This tool is for research purposes only..."
}
```

Decode the heatmap to visualize:

```python
import base64, io
from PIL import Image

img = Image.open(io.BytesIO(base64.b64decode(response["gradcam_heatmap"])))
img.show()
```

### `POST /explain`
Sends prediction data to Flan-T5-Large and returns a natural language clinical explanation.

**Request:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"class_name": "Moderate DR", "confidence": 0.87, "probabilities": {...}}'
```

**Response:**
```json
{
  "explanation": "Moderate non-proliferative diabetic retinopathy indicates..."
}
```

Requires `HF_TOKEN` environment variable to be set.

### `GET /metrics`
Prometheus-format metrics endpoint. Scraped automatically.

---

## Monitoring

Once the stack is running, open Grafana at http://localhost:3000 (`admin / admin`).

The Prometheus datasource is pre-configured. Key metrics:

| Metric | What it shows |
|--------|--------------|
| `http_request_duration_seconds` | Inference latency distribution |
| `http_requests_total` | Request volume by endpoint and status |
| `predictions_total` | Prediction count per DR severity class |
| `prediction_confidence` | Distribution of model confidence scores |

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

18 tests covering model architecture, inference logic, and all API endpoints. No model weights needed — tests use a mock.

---

## Training

To retrain from scratch on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset:

```bash
# 1. Install dependencies locally
pip install -r requirements.txt

# 2. Download the dataset from Kaggle:
#    https://www.kaggle.com/c/aptos2019-blindness-detection/data
#    Then place the files following this structure:
#    data/train_images/train_images/*.png
#    data/val_images/val_images/*.png
#    data/train_1.csv  (columns: id_code, diagnosis)
#    data/valid.csv

# 3. Run training
python -m src.train
```

The best checkpoint is saved to `models/best_model.pth` when validation loss improves.

**Training config** (see [src/config.py](src/config.py)):
- Image size: 224×224
- Batch size: 32
- Optimizer: Adam (lr=1e-4) with ReduceLROnPlateau
- Loss: Weighted CrossEntropyLoss (handles class imbalance)

---

## Project Structure

```
├── src/
│   ├── api.py          # FastAPI app, endpoints, HF weight download
│   ├── config.py       # Paths and hyperparameters
│   ├── dataset.py      # PyTorch Dataset + augmentations
│   ├── gradcam.py      # Grad-CAM heatmap generation
│   ├── model.py        # EfficientNet-B4 architecture
│   ├── predict.py      # Inference logic
│   └── train.py        # Training pipeline
├── static/
│   └── index.html      # Medical web UI
├── tests/
│   ├── conftest.py     # Shared fixtures + model mock
│   ├── test_api.py     # FastAPI endpoint tests
│   ├── test_model.py   # Model architecture tests
│   └── test_predict.py # Inference logic tests
├── models/             # Model weights (not tracked in git)
├── data/               # APTOS 2019 dataset (not tracked in git)
├── monitoring/
│   ├── prometheus.yml  # Prometheus scrape config
│   └── grafana/        # Grafana datasource provisioning
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt        # Full deps (training)
├── requirements-api.txt    # Lean deps (API + Docker)
└── requirements-dev.txt    # Test deps
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| CV Model | EfficientNet-B4 (PyTorch) |
| API | FastAPI + uvicorn |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| NLP Explanation | Flan-T5-Large (Hugging Face Inference API) |
| Web UI | Vanilla HTML/CSS/JS (served by FastAPI) |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Model Hosting | Hugging Face Hub |
| CI/CD | GitHub Actions + Railway |
| Dataset | [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle) |
