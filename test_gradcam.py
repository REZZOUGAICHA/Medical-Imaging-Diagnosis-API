import requests, base64
from PIL import Image
import io

response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("data/test_images/test_images/e4dcca36ceb4.png", "rb")}
)

data = response.json()

# Decode and view the heatmap
img_bytes = base64.b64decode(data["gradcam_heatmap"])
heatmap   = Image.open(io.BytesIO(img_bytes))
heatmap.show()