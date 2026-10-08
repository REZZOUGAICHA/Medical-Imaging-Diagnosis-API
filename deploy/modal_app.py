# Full FastAPI service on Modal (https://modal.com), free Starter credits.
#   modal secret create dr-grading HF_TOKEN=hf_... ALLOWED_ORIGINS=https://<site>.netlify.app RATE_LIMIT_PER_MINUTE=10
#   modal deploy deploy/modal_app.py
# Serves the same app as the Docker image: UI at /, /predict, /explain, /docs, /metrics.
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
HF_REPO_ID = "aicharzg/diabetic-retinopathy-efficientnet-b4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libxcb1")  # OpenCV, used by pytorch-grad-cam
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "requirements-api.txt"))
    # bake the weights into the image so a cold start doesn't download 70 MB
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id='{HF_REPO_ID}', filename='best_model.pth', local_dir='/root/models')\""
    )
    .env({"PYTHONUNBUFFERED": "1"})
    # code goes in last so editing it doesn't rebuild the layers above
    .add_local_dir(str(ROOT / "src"), "/root/src")
    .add_local_dir(str(ROOT / "static"), "/root/static")
)

app = modal.App("dr-grading", image=image)


@app.function(
    secrets=[modal.Secret.from_name("dr-grading")],
    cpu=1.0,
    memory=2048,
    # one container keeps spend bounded and the in-memory rate limiter exact
    max_containers=1,
    scaledown_window=120,
    timeout=120,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(label="dr-grading")
def api():
    from src.api import app as fastapi_app

    return fastapi_app
