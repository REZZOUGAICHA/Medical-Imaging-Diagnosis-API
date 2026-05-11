from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

_FAKE_RESULT = {
    "predicted_class": 0,
    "class_name": "No Diabetic Retinopathy",
    "description": "No signs of diabetic retinopathy detected.",
    "confidence": 0.95,
    "probabilities": {
        "No Diabetic Retinopathy": 0.95,
        "Mild DR": 0.03,
        "Moderate DR": 0.01,
        "Severe DR": 0.005,
        "Proliferative DR": 0.005,
    },
    "gradcam_heatmap": "abc123base64",
}


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body():
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["model"] == "EfficientNet-B4"
    assert "device" in data


def test_model_info_returns_200():
    resp = client.get("/model-info")
    assert resp.status_code == 200


def test_model_info_body():
    data = client.get("/model-info").json()
    assert data["num_classes"] == 5
    assert len(data["classes"]) == 5


def test_predict_rejects_non_image(jpeg_bytes):
    resp = client.post(
        "/predict",
        files={"file": ("doc.txt", jpeg_bytes, "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_success(jpeg_bytes):
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        resp = client.post(
            "/predict",
            files={"file": ("retina.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "confidence" in data
    assert "gradcam_heatmap" in data
    assert "disclaimer" in data


def test_predict_response_has_disclaimer(jpeg_bytes):
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        data = client.post(
            "/predict",
            files={"file": ("retina.jpg", jpeg_bytes, "image/jpeg")},
        ).json()
    assert "research purposes only" in data["disclaimer"].lower()
