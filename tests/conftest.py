import io
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.model import build_model

# real EfficientNet-B4 architecture but random weights, so no best_model.pth needed.
# the model is loaded in the app lifespan, so the client fixtures patch the loader
# and use `with TestClient(app)` to trigger startup
_model = build_model(pretrained=False)
_model.eval()
_device = torch.device("cpu")


@pytest.fixture(scope="session")
def model():
    return _model


@pytest.fixture(scope="session")
def device():
    return _device


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from src.api import app
    with patch("src.api.download_weights"), \
         patch("src.api.load_model", return_value=(_model, _device)):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_model_failed(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from src.api import app
    with patch("src.api.download_weights"), \
         patch("src.api.load_model", side_effect=RuntimeError("weights corrupt")):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_image():
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def jpeg_bytes(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()
