import io
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from huggingface_hub import hf_hub_download

from src.predict import load_model, predict, predict_with_explainability
from src.config import SAVE_PATH, MODELS_DIR, ROOT_DIR

HF_REPO_ID = "aicharzg/diabetic-retinopathy-efficientnet-b4"

if not os.path.exists(SAVE_PATH):
    os.makedirs(MODELS_DIR, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="best_model.pth",
        local_dir=MODELS_DIR,
    )


app = FastAPI(
    title="Medical Imaging Diagnosis API",
    description="Diabetic retinopathy severity classification using EfficientNet-B4",
    version="1.0.0"
)

# Wire up automatic HTTP metrics (latency, request count, in-flight)
# and expose them at GET /metrics for Prometheus to scrape
Instrumentator().instrument(app).expose(app)

# Custom metric: count predictions per DR severity class
# labels=["predicted_class"] means each class gets its own counter line
predictions_counter = Counter(
    "predictions_total",
    "Total predictions by DR severity class",
    ["predicted_class"]
)

# Custom metric: distribution of confidence scores
# buckets divide scores into ranges — tells us if the model is uncertain overall
confidence_histogram = Histogram(
    "prediction_confidence",
    "Distribution of model confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

model, device = load_model(SAVE_PATH)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    with open(os.path.join(ROOT_DIR, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "EfficientNet-B4",
        "device": str(device)
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name":  "EfficientNet-B4",
        "dataset":     "APTOS 2019 Blindness Detection",
        "task":        "Diabetic Retinopathy Classification",
        "num_classes": 5,
        "input_size":  "224x224 RGB",
        "classes": {
            "0": "No Diabetic Retinopathy",
            "1": "Mild DR",
            "2": "Moderate DR",
            "3": "Severe DR",
            "4": "Proliferative DR"
        }
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG accepted."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        result = predict_with_explainability(image, model, device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Record custom metrics after every successful prediction
    predictions_counter.labels(predicted_class=str(result["predicted_class"])).inc()
    confidence_histogram.observe(result["confidence"])

    return JSONResponse(content={
        "success":         True,
        "predicted_class": result["predicted_class"],
        "class_name":      result["class_name"],
        "description":     result["description"],
        "confidence":      result["confidence"],
        "probabilities":   result["probabilities"],
        "gradcam_heatmap": result["gradcam_heatmap"],
        "disclaimer":      "This tool is for research purposes only and does not constitute medical advice."
    })
