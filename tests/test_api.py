from unittest.mock import MagicMock, patch

from src.api import app

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


def test_liveness_returns_200(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_when_model_loaded(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model"] == "EfficientNet-B4"
    assert data["device"] == "cpu"


def test_readiness_503_when_model_failed(client_model_failed):
    resp = client_model_failed.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["model_loaded"] is False


def test_liveness_ok_even_when_model_failed(client_model_failed):
    assert client_model_failed.get("/health/live").status_code == 200


def test_legacy_health_alias(client):
    assert client.get("/health").status_code == 200


def test_model_info_returns_200(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200


def test_model_info_body(client):
    data = client.get("/model-info").json()
    assert data["num_classes"] == 5
    assert len(data["classes"]) == 5


def test_predict_rejects_non_image_content_type(client, jpeg_bytes):
    resp = client.post("/predict", files={"file": ("doc.txt", jpeg_bytes, "text/plain")})
    assert resp.status_code == 400


def test_predict_rejects_spoofed_content_type(client):
    resp = client.post("/predict", files={"file": ("x.png", b"not an image at all", "image/png")})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Could not decode image."


def test_predict_rejects_oversized_upload(client):
    with patch("src.api.MAX_UPLOAD_BYTES", 1024):
        resp = client.post("/predict", files={"file": ("big.jpg", b"\xff" * 2048, "image/jpeg")})
    assert resp.status_code == 413


def test_predict_503_when_model_not_loaded(client_model_failed, jpeg_bytes):
    resp = client_model_failed.post("/predict", files={"file": ("r.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 503


def test_predict_success(client, jpeg_bytes):
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        resp = client.post("/predict", files={"file": ("retina.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "confidence" in data
    assert "gradcam_heatmap" in data
    assert "disclaimer" in data


def test_predict_response_has_disclaimer(client, jpeg_bytes):
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        data = client.post("/predict", files={"file": ("retina.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert "research purposes only" in data["disclaimer"].lower()


def test_predict_error_does_not_leak_internals(client, jpeg_bytes):
    with patch("src.api.predict_with_explainability", side_effect=RuntimeError("CUDA secret path /opt/x")):
        resp = client.post("/predict", files={"file": ("retina.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 500
    body = resp.json()
    assert "secret" not in body["detail"]
    assert body["request_id"] == resp.headers["X-Request-ID"]


def test_predict_runs_in_threadpool(client, jpeg_bytes):
    # inference must not run directly on the event loop
    async def fake_pool(fn, *args):
        return _FAKE_RESULT

    with patch("src.api.run_in_threadpool", side_effect=fake_pool) as pool:
        resp = client.post("/predict", files={"file": ("retina.jpg", jpeg_bytes, "image/jpeg")})
    assert resp.status_code == 200
    assert pool.call_args.args[0].__name__ == "predict_with_explainability"


def test_response_has_request_id(client):
    assert client.get("/health/live").headers.get("X-Request-ID")


def test_upstream_request_id_is_propagated(client):
    resp = client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


def test_explain_503_without_token(client):
    resp = client.post("/explain", json={"class_name": "Mild DR", "confidence": 0.8})
    assert resp.status_code == 503


def test_explain_validates_confidence_range(client):
    resp = client.post("/explain", json={"class_name": "Mild DR", "confidence": 1.7})
    assert resp.status_code == 422


def test_explain_validates_missing_class_name(client):
    resp = client.post("/explain", json={"confidence": 0.5})
    assert resp.status_code == 422


def test_explain_success_with_mock_client(client):
    fake = MagicMock()
    fake.chat_completion.return_value.choices = [MagicMock()]
    fake.chat_completion.return_value.choices[0].message.content = "  Mild DR means early changes.  "
    app.state.hf_client = fake
    resp = client.post(
        "/explain",
        json={"class_name": "Mild DR", "confidence": 0.8, "probabilities": {"Mild DR": 0.8}},
    )
    assert resp.status_code == 200
    assert resp.json()["explanation"] == "Mild DR means early changes."
    assert resp.json()["model"]


def test_explain_upstream_error_is_generic(client):
    fake = MagicMock()
    fake.chat_completion.side_effect = RuntimeError("401 token hf_abc invalid")
    app.state.hf_client = fake
    resp = client.post("/explain", json={"class_name": "Mild DR", "confidence": 0.8})
    assert resp.status_code == 502
    assert "hf_abc" not in resp.json()["detail"]


def test_rate_limit_off_by_default(client, jpeg_bytes):
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        for _ in range(5):
            resp = client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")})
            assert resp.status_code == 200


def test_rate_limit_returns_429(client, jpeg_bytes):
    client.app.state.rate_limit = 2
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        codes = [
            client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")}).status_code
            for _ in range(3)
        ]
    assert codes == [200, 200, 429]
    resp = client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")})
    assert "Retry-After" in resp.headers
    assert "request_id" in resp.json()


def test_rate_limit_is_per_client_ip(client, jpeg_bytes):
    client.app.state.rate_limit = 1
    with patch("src.api.predict_with_explainability", return_value=_FAKE_RESULT):
        a = client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")},
                        headers={"X-Forwarded-For": "1.1.1.1"})
        b = client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")},
                        headers={"X-Forwarded-For": "2.2.2.2"})
    assert (a.status_code, b.status_code) == (200, 200)
