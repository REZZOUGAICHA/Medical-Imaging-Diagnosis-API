import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.config import NUM_CLASSES, IMAGE_SIZE, SAVE_PATH


CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild non-proliferative DR. Monitoring recommended.",
    2: "Moderate non-proliferative DR. Medical review advised.",
    3: "Severe non-proliferative DR. Urgent referral recommended.",
    4: "Proliferative DR. Immediate medical attention required."
}

inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = SAVE_PATH

    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path} on {device}")
    return model, device


def predict(image: Image.Image, model, device):
    # Preprocess
    tensor = inference_transforms(image)   # [3, 224, 224]
    tensor = tensor.unsqueeze(0)           # [1, 3, 224, 224]
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)             # [1, 5]
        probs  = F.softmax(logits, dim=1)  # [1, 5]

    # Parse results
    probs_list      = probs.squeeze().cpu().tolist()
    predicted_class = int(torch.argmax(probs, dim=1).item())
    confidence      = probs_list[predicted_class]

    return {
        "predicted_class": predicted_class,
        "class_name":      CLASS_NAMES[predicted_class],
        "description":     CLASS_DESCRIPTIONS[predicted_class],
        "confidence":      round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(p, 4)
            for i, p in enumerate(probs_list)
        }
    }


def predict_with_explainability(image: Image.Image, model, device) -> dict:
    from src.gradcam import get_gradcam_heatmap

    # Step 1 - standard prediction
    result = predict(image, model, device)

    # Step 2 - Grad-CAM heatmap for the predicted class
    heatmap_b64 = get_gradcam_heatmap(
        image=image,
        model=model,
        device=device,
        predicted_class=result["predicted_class"]
    )

    result["gradcam_heatmap"] = heatmap_b64
    return result