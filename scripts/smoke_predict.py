"""
Manual smoke test against a running server: send one image to /predict and
open the returned Grad-CAM overlay.

Usage:
    python -m scripts.smoke_predict path/to/fundus.png [--url http://localhost:8000]
"""
import argparse
import base64
import io

import httpx
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        resp = httpx.post(f"{args.url}/predict", files={"file": (args.image, f, "image/png")}, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print(f"{data['class_name']} ({data['confidence']:.1%})  request_id={resp.headers.get('X-Request-ID')}")
    Image.open(io.BytesIO(base64.b64decode(data["gradcam_heatmap"]))).show()


if __name__ == "__main__":
    main()
