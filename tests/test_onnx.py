import numpy as np
import pytest
import torch

from scripts.export_onnx import MEAN, STD, export

ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def onnx_session(tmp_path_factory, model):
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export(model, str(path))
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def raw_image():
    torch.manual_seed(0)
    return torch.rand(1, 3, 224, 224)


def normalise(x):
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def test_onnx_logits_match_pytorch(onnx_session, model, raw_image):
    logits, _ = onnx_session.run(None, {"image": raw_image.numpy()})
    with torch.no_grad():
        expected = model(normalise(raw_image)).numpy()
    np.testing.assert_allclose(logits, expected, atol=1e-4)


def test_onnx_cam_matches_gradcam(onnx_session, model, raw_image):
    # Grad-CAM from its definition: channel weights = spatial mean of d(logit)/d(A).
    # Compared before min-max scaling: with random weights the maps are ~1e-16 and
    # pytorch-grad-cam's 1e-7 epsilon in the scaling would swamp them.
    _, cams = onnx_session.run(None, {"image": raw_image.numpy()})
    acts = {}
    hook = model.features[-1].register_forward_hook(lambda m, i, o: acts.update(a=o))
    try:
        logits = model(normalise(raw_image))
    finally:
        hook.remove()

    for target in range(5):
        grads = torch.autograd.grad(logits[0, target], acts["a"], retain_graph=True)[0]
        expected = torch.relu((grads.mean((2, 3), keepdim=True) * acts["a"]).sum(1))[0]
        ours = np.maximum(cams[0, target], 0)
        np.testing.assert_allclose(ours, expected.detach().numpy(), rtol=1e-3, atol=1e-6 * float(expected.max()))
