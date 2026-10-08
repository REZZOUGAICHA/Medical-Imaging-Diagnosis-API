# Export the classifier to ONNX for in-browser inference (static/demo on Netlify).
#   python -m scripts.export_onnx --out models/dr_efficientnet_b4.onnx
#
# The graph takes a [1, 3, 224, 224] image scaled to 0..1 (ImageNet normalisation
# happens inside) and returns:
#   logits [1, 5]
#   cams   [1, 5, 7, 7]  class activation map for every grade
#
# Why cams is the same as Grad-CAM here: the head is avgpool -> dropout -> linear,
# so d(logit_c)/d(A_k[i,j]) = W[c,k] / (H*W) at every position. Grad-CAM's channel
# weights are therefore W[c,k] / (H*W), and the map is ReLU(sum_k W[c,k] * A_k) / (H*W).
# The browser only has to apply ReLU, min-max scale and upsample, like pytorch-grad-cam.
import argparse
import os

import torch
import torch.nn as nn

from src.config import MODELS_DIR, SAVE_PATH
from src.predict import load_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CamExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.linear = model.classifier[1]  # classifier[0] is dropout, a no-op in eval
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = self.features(x)                                   # [1, 1792, 7, 7]
        logits = self.linear(torch.flatten(self.avgpool(feats), 1))
        hw = feats.shape[2] * feats.shape[3]
        cams = torch.einsum("ck,bkhw->bchw", self.linear.weight, feats) / hw
        return logits, cams


def export(model, out_path):
    wrapper = CamExportModel(model).eval()
    dummy = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["image"], output_names=["logits", "cams"],
        opset_version=17, dynamo=False,
    )
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(MODELS_DIR, "dr_efficientnet_b4.onnx"))
    args = parser.parse_args()

    model, _ = load_model(SAVE_PATH, device=torch.device("cpu"))
    export(model, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
