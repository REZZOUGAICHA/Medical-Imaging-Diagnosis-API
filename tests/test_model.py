import torch
import torch.nn as nn

from src.model import build_model


def test_default_output_shape():
    model = build_model(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


def test_custom_num_classes():
    model = build_model(num_classes=3, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)


def test_classifier_contains_dropout():
    model = build_model(pretrained=False)
    assert any(isinstance(layer, nn.Dropout) for layer in model.classifier)


def test_classifier_output_matches_num_classes():
    model = build_model(num_classes=5, pretrained=False)
    linear = model.classifier[-1]
    assert isinstance(linear, nn.Linear)
    assert linear.out_features == 5
