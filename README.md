# Medical Imaging Diagnosis API

![CI](https://github.com/REZZOUGAICHA/Medical-Imaging-Diagnosis-API/actions/workflows/ci.yml/badge.svg)

A production-ready REST API for **diabetic retinopathy severity classification** from retinal fundus images. Returns a diagnosis, confidence score, a Grad-CAM heatmap, and an AI-generated clinical explanation powered by Flan-T5-Base.

Built with EfficientNet-B4 fine-tuned on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset. Ships with a medical web UI, full monitoring via Prometheus and Grafana, 33 tests, and a GitHub Actions CI pipeline.

**Model performance (held-out test split, 366 images):** accuracy **78.7%**, quadratic weighted kappa **0.863**, macro-F1 **0.608**. See [Model Performance](#model-performance).

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
│  POST /explain   → Flan-T5-Base NLP      │    │  best_model.pth      │
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
- **AI clinical explanation** — Flan-T5-Base generates a natural language report for each prediction
- **Medical web UI** — drag-and-drop upload, color-coded severity, probability bars, side-by-side image comparison
- **Prometheus + Grafana monitoring** — latency, request counts, per-class prediction counts, confidence distribution
- **Production serving** — model loaded in FastAPI `lifespan`, inference off the event loop, typed request/response schemas, upload size cap, split liveness/readiness probes
- **Structured JSON logging** — every log line and error response carries an `X-Request-ID`
- **33 tests** — model architecture, inference logic, API endpoints, validation and error paths
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

Grafana login: the `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` you set in `.env`.

### 2. Configuration

Copy the template and fill it in (compose refuses to start without a Grafana password):

```bash
cp .env.example .env
```

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | Enables `POST /explain` (Flan-T5-Base) | unset → `/explain` returns 503 |
| `MAX_UPLOAD_MB` | Upload size cap for `/predict` | `10` |
| `LOG_LEVEL` | Log verbosity | `INFO` |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | Grafana login | password required |

The container runs as a non-root user (uid 1000). If you bind-mount `./models` on Linux, make sure it is writable by that uid so the first-boot weight download succeeds.

---

## Web UI

Open **http://localhost:8000** for the interactive medical interface:

1. Drag and drop a retinal fundus image (JPEG or PNG)
2. The model returns predicted DR severity + confidence + probability distribution
3. Grad-CAM heatmap shows which retinal regions drove the prediction
4. Click **Generate AI Clinical Explanation** for a Flan-T5-Base natural language report

---

## API Reference

Interactive schema docs for every endpoint are at `/docs`. All error responses have the shape `{"detail": "...", "request_id": "..."}`; the same ID is returned in the `X-Request-ID` header and appears in the server logs, so a failure can be traced from the client to the exact log line.

### `GET /health/live`
Liveness — the process is up. Always 200 while the server runs.

### `GET /health/ready`
Readiness — the model is loaded and inference is possible. Returns **503** until startup finishes, or permanently if the weights failed to load (the process keeps running so the error shows up in the logs instead of a restart loop).

```json
{
  "status": "ready",
  "model_loaded": true,
  "model": "EfficientNet-B4",
  "device": "cpu",
  "explanation_service": true
}
```

`GET /health` is kept as an alias of `/health/ready` for existing monitors.

### `GET /model-info`
Returns model metadata and class labels.

### `POST /predict`
Accepts a retinal fundus image (JPEG or PNG, max 10 MB) and returns a diagnosis. The `Content-Type` header is only a first filter — the file must actually decode as a JPEG or PNG. Returns 400 for invalid images, 413 for oversized uploads, 503 if the model isn't loaded.

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
Sends prediction data to Flan-T5-Base and returns a natural language clinical explanation.

**Request:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"class_name": "Moderate DR", "confidence": 0.87, "probabilities": {"Moderate DR": 0.87, "Mild DR": 0.06}}'
```

**Response:**
```json
{
  "explanation": "Moderate non-proliferative diabetic retinopathy indicates..."
}
```

Body is validated: `class_name` 1–64 chars, `confidence` in [0, 1] (422 otherwise). Requires `HF_TOKEN`; returns 503 without it, 502/504 if the upstream model fails or times out.

### `GET /metrics`
Prometheus-format metrics endpoint. Scraped automatically.

---

## Monitoring

Once the stack is running, open Grafana at http://localhost:3000 with the credentials from your `.env`.

The Prometheus datasource and a **Medical Imaging API** dashboard are provisioned automatically (Dashboards → Medical Imaging API). It shows request rate, `/predict` p50/p95 latency, error rate, predictions per DR grade and the confidence distribution.

To put some traffic through it (needs the APTOS test images in `data/`):

```bash
python -m scripts.load_test --n 50
```

Key metrics:

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

33 tests covering model architecture, inference logic, all API endpoints, readiness when the model fails to load, upload size limits, spoofed content types, request validation, and that internal error text never reaches the client. No model weights needed — the model loader is patched in the test fixtures.

Manual smoke test against a running server:

```bash
python -m scripts.smoke_predict path/to/fundus.png --url http://localhost:8000
```

---

## Model Performance

Evaluated with [scripts/evaluate.py](scripts/evaluate.py) on the fine-tuned checkpoint (`python -m scripts.evaluate --split test`). Single evaluation run, deterministic inference, September 2026.

| Split | Images | Accuracy | Quadratic weighted kappa | Macro-F1 |
|-------|--------|----------|--------------------------|----------|
| **Test (held out)** | 366 | **78.7%** | **0.863** | **0.608** |
| Validation* | 366 | 76.5% | 0.859 | 0.600 |

\*The validation split was used to select the best checkpoint (by validation loss), so it is not independent. The test split is the headline figure. Validation actually scores slightly below test, so the selection bias looks small; a 2-point gap on 366 images is within noise.

Quadratic weighted kappa is the APTOS 2019 competition metric: it penalises a prediction by the square of its distance from the true grade, so calling Proliferative DR "Mild" costs far more than calling it "Severe".

**Test confusion matrix** (rows = true grade, columns = predicted):

| | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| **No DR** (199) | **195** | 3 | 1 | 0 | 0 |
| **Mild** (30) | 3 | **19** | 6 | 1 | 1 |
| **Moderate** (87) | 2 | 14 | **51** | 12 | 8 |
| **Severe** (17) | 0 | 0 | 8 | **8** | 1 |
| **Proliferative** (33) | 0 | 3 | 9 | 6 | **15** |

No DR vs DR is separated well (195/199) and most mistakes are one grade off, which is why kappa is high. The weak point is the rare classes: Severe DR recall is 8/17, Proliferative 15/33, and 3 Proliferative cases were predicted as Mild. Those are the errors that would matter clinically, so this is a research demo and not a screening tool.

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
│   ├── api.py          # FastAPI app, lifespan, middleware, endpoints
│   ├── schemas.py      # Pydantic request/response models
│   ├── logging_config.py # JSON logging + request-ID context
│   ├── config.py       # Paths and hyperparameters
│   ├── dataset.py      # PyTorch Dataset + augmentations
│   ├── gradcam.py      # Grad-CAM heatmap generation
│   ├── model.py        # EfficientNet-B4 architecture
│   ├── predict.py      # Inference logic
│   └── train.py        # Training pipeline
├── scripts/
│   ├── evaluate.py     # Accuracy / QWK / F1 on a labelled split
│   ├── load_test.py    # Traffic generator for the Grafana dashboard
│   └── smoke_predict.py # Manual end-to-end check against a running server
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
│   └── grafana/        # Grafana datasource + dashboard provisioning
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
├── .env.example
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
| NLP Explanation | Flan-T5-Base (Hugging Face Inference API) |
| Web UI | Vanilla HTML/CSS/JS (served by FastAPI) |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Model Hosting | Hugging Face Hub |
| CI/CD | GitHub Actions + Railway |
| Dataset | [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle) |
