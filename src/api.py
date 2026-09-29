import asyncio
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import InferenceClient, hf_hub_download
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import MODELS_DIR, ROOT_DIR, SAVE_PATH
from src.logging_config import request_id_var, setup_logging
from src.predict import CLASS_NAMES, load_model, predict_with_explainability
from src.schemas import (
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    LivenessResponse,
    ModelInfoResponse,
    PredictResponse,
    ReadinessResponse,
)

load_dotenv()
setup_logging()
logger = logging.getLogger("api")

HF_REPO_ID       = "aicharzg/diabetic-retinopathy-efficientnet-b4"
EXPLAIN_MODEL    = os.environ.get("EXPLAIN_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_TYPES    = {"image/jpeg", "image/png"}
DISCLAIMER       = "This tool is for research purposes only and does not constitute medical advice."


def download_weights():
    if os.path.exists(SAVE_PATH):
        return
    logger.info("Downloading weights from %s", HF_REPO_ID)
    os.makedirs(MODELS_DIR, exist_ok=True)
    hf_hub_download(repo_id=HF_REPO_ID, filename="best_model.pth", local_dir=MODELS_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.device = None

    # if loading fails we keep running but /health/ready stays 503
    try:
        await run_in_threadpool(download_weights)
        app.state.model, app.state.device = await run_in_threadpool(load_model, SAVE_PATH)
    except Exception:
        logger.exception("Could not load model")

    hf_token = os.environ.get("HF_TOKEN")
    app.state.hf_client = InferenceClient(token=hf_token, timeout=20) if hf_token else None

    yield


app = FastAPI(
    title="Medical Imaging Diagnosis API",
    description="Diabetic retinopathy severity classification using EfficientNet-B4",
    version="1.1.0",
    lifespan=lifespan,
)

# Wire up automatic HTTP metrics (latency, request count, in-flight)
# and expose them at GET /metrics for Prometheus to scrape.
# default latency buckets stop at 1s, but /predict takes a few seconds on CPU
Instrumentator().add(
    metrics.default(latency_lowr_buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 15))
).instrument(app).expose(app)

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

ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 413, 500, 502, 503, 504)}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error")
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, elapsed_ms)
    request_id_var.reset(token)
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id_var.get()},
        headers=exc.headers,
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    with open(os.path.join(ROOT_DIR, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/health/live", response_model=LivenessResponse, tags=["health"])
def health_live():
    return {"status": "alive"}


@app.get("/health", include_in_schema=False)
@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"],
         responses={503: {"model": ReadinessResponse}})
def health_ready(request: Request):
    state = request.app.state
    ready = state.model is not None
    body = ReadinessResponse(
        status="ready" if ready else "not_ready",
        model_loaded=ready,
        model="EfficientNet-B4",
        device=str(state.device) if ready else None,
        explanation_service=state.hf_client is not None,
    )
    return JSONResponse(status_code=200 if ready else 503, content=body.model_dump())


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "model_name":  "EfficientNet-B4",
        "dataset":     "APTOS 2019 Blindness Detection",
        "task":        "Diabetic Retinopathy Classification",
        "num_classes": len(CLASS_NAMES),
        "input_size":  "224x224 RGB",
        "classes":     {str(k): v for k, v in CLASS_NAMES.items()},
    }


@app.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    model, device = request.app.state.model, request.app.state.device
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # content_type comes from the client so it's only a first check,
    # the real check is whether PIL can open the file
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG accepted.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // 2**20} MB).")

    try:
        image = Image.open(io.BytesIO(contents))
        if image.format not in ("JPEG", "PNG"):
            raise ValueError(image.format)
        image = image.convert("RGB")
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # model + grad-cam are blocking, run them in a thread so the event loop stays free
    try:
        result = await run_in_threadpool(predict_with_explainability, image, model, device)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.")

    predictions_counter.labels(predicted_class=str(result["predicted_class"])).inc()
    confidence_histogram.observe(result["confidence"])

    return PredictResponse(success=True, disclaimer=DISCLAIMER, **result)


@app.post("/explain", response_model=ExplainResponse, responses=ERROR_RESPONSES)
async def explain_endpoint(request: Request, body: ExplainRequest):
    client = request.app.state.hf_client
    if client is None:
        raise HTTPException(status_code=503, detail="Explanation service not configured.")

    top_probs = ", ".join(
        f"{k} {v*100:.1f}%"
        for k, v in sorted(body.probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    )

    prompt = (
        f"A retinal fundus image was graded by a deep learning model for diabetic retinopathy. "
        f"Prediction: {body.class_name} ({body.confidence*100:.1f}% confidence). "
        f"Top probabilities: {top_probs}. "
        f"In 3 sentences, explain what this grade means, which retinal findings are typically "
        f"associated with it, and what follow-up is usually recommended."
    )
    messages = [
        {"role": "system", "content": "You write short, factual notes for clinicians. No markdown, no lists."},
        {"role": "user", "content": prompt},
    ]

    try:
        completion = await asyncio.wait_for(
            asyncio.to_thread(client.chat_completion, messages, model=EXPLAIN_MODEL, max_tokens=200),
            timeout=25,
        )
        text = completion.choices[0].message.content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Model took too long to respond, please try again.")
    except Exception:
        logger.exception("HF inference call failed")
        raise HTTPException(status_code=502, detail="Explanation service error.")

    if not text or not text.strip():
        raise HTTPException(status_code=502, detail="Empty response from language model.")

    return ExplainResponse(explanation=text.strip(), model=EXPLAIN_MODEL)
