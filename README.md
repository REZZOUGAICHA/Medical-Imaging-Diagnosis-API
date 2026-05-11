# Medical Imaging Diagnosis API

A production-ready REST API for **diabetic retinopathy severity classification** from retinal fundus images. Returns a diagnosis, confidence score, and a Grad-CAM heatmap explaining which regions of the image drove the prediction.

Built with EfficientNet-B4 fine-tuned on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset. Fully containerized with monitoring via Prometheus and Grafana.

> **Disclaimer:** This tool is for research purposes only and does not constitute medical advice.

---

## Architecture

```
Client (image upload)
        │
        ▼
┌───────────────────────────────────┐
│         FastAPI  :8000            │
│                                   │
│  POST /predict                    │
│    → validate image               │
│    → EfficientNet-B4 inference    │    ┌─────────────────────┐
│    → Grad-CAM heatmap         ────┼───▶│  models/            │
│    → return JSON + base64 image   │    │  best_model.pth     │
│                                   │    └─────────────────────┘
│  GET  /metrics  (Prometheus)      │
└───────────────────────────────────┘
        │  scrape /metrics every 15s
        ▼
┌───────────────────┐      query      ┌──────────────────────┐
│   Prometheus :9090│ ──────────────▶ │   Grafana  :3000     │
│   (time-series DB)│                 │   (dashboards)       │
└───────────────────┘                 └──────────────────────┘
```

---

## Quick Start

**Requirements:** Docker and Docker Compose installed.

### 1. Get model weights

The trained weights are not stored in this repo (too large for git). Place your `best_model.pth` file in the `models/` directory:

```
models/
└── best_model.pth   ← required
```

To train from scratch, see [Training](#training).

### 2. Start everything

```bash
docker-compose up --build
```

This starts three services:

| Service    | URL                        | Purpose                         |
|------------|----------------------------|---------------------------------|
| API        | http://localhost:8000      | FastAPI inference server        |
| Prometheus | http://localhost:9090      | Metrics storage                 |
| Grafana    | http://localhost:3000      | Monitoring dashboard            |

Grafana login: `admin / admin`

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

The `gradcam_heatmap` field is a base64-encoded PNG overlay showing which retinal regions the model attended to. Decode it to visualize:

```python
import base64, io
from PIL import Image

img = Image.open(io.BytesIO(base64.b64decode(response["gradcam_heatmap"])))
img.show()
```

### `GET /metrics`
Prometheus-format metrics endpoint. Scraped automatically — not intended for direct use.

---

## Monitoring

Once the stack is running, open Grafana at http://localhost:3000 (`admin / admin`).

The Prometheus datasource is pre-configured. Key metrics to dashboard:

| Metric | What it shows |
|--------|--------------|
| `http_request_duration_seconds` | Inference latency distribution |
| `http_requests_total` | Request volume by endpoint and status |
| `predictions_total` | Prediction count per DR severity class |
| `prediction_confidence` | Distribution of model confidence scores |

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
│   ├── api.py          # FastAPI app and endpoints
│   ├── config.py       # Paths and hyperparameters
│   ├── dataset.py      # PyTorch Dataset + augmentations
│   ├── gradcam.py      # Grad-CAM heatmap generation
│   ├── model.py        # EfficientNet-B4 architecture
│   ├── predict.py      # Inference logic
│   └── train.py        # Training pipeline
├── models/             # Model weights (not tracked in git)
├── data/               # APTOS 2019 dataset (not tracked in git)
├── monitoring/
│   ├── prometheus.yml  # Prometheus scrape config
│   └── grafana/        # Grafana datasource provisioning
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | EfficientNet-B4 (PyTorch) |
| API | FastAPI + uvicorn |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Dataset | [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle) |
