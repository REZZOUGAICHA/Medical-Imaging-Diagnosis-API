# Request and response models used by the API (also what /docs shows)
from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    status: Literal["alive"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model: str
    device: str | None
    explanation_service: bool = Field(description="True if HF_TOKEN is configured and /explain is usable")


class ModelInfoResponse(BaseModel):
    model_name: str
    dataset: str
    task: str
    num_classes: int
    input_size: str
    classes: dict[str, str]


class PredictResponse(BaseModel):
    success: bool
    predicted_class: int = Field(ge=0, le=4)
    class_name: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    gradcam_heatmap: str = Field(description="Base64-encoded PNG overlay")
    disclaimer: str


class ExplainRequest(BaseModel):
    class_name: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict, max_length=10)


class ExplainResponse(BaseModel):
    explanation: str
    model: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
