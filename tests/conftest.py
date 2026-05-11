"""
Shared fixtures and session-level mock setup.

Why the module-level patch?
  api.py runs `model, device = load_model(SAVE_PATH)` at import time.
  If load_model runs for real it needs best_model.pth, which doesn't exist in CI.
  By patching src.predict.load_model *before* any test file imports src.api,
  the `from src.predict import load_model` line inside api.py picks up the mock.
"""
import io

import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch

from src.model import build_model

# Build a real EfficientNet-B4 with ImageNet weights but WITHOUT the custom
# DR fine-tuned weights (pretrained=False skips loading best_model.pth).
# This gives us a properly-shaped model that supports real forward passes
# and Grad-CAM (which needs model.features[-1] to exist).
_model = build_model(pretrained=False)
_model.eval()
_device = torch.device("cpu")

# Start the patch at module level — conftest.py is fully executed before
# pytest imports any test_*.py file, so api.py hasn't been imported yet.
_patcher = patch("src.predict.load_model", return_value=(_model, _device))
_patcher.start()


@pytest.fixture(scope="session")
def model():
    return _model


@pytest.fixture(scope="session")
def device():
    return _device


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
