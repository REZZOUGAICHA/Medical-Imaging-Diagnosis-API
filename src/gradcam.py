import io
import base64
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_heatmap(image: Image.Image, model, device, predicted_class: int) -> str:
    """
    Generate a Grad-CAM heatmap for the predicted class.

    Grad-CAM works by computing gradients of the predicted class score
    with respect to the last convolutional layer's feature maps.
    Channels with large gradients contributed most to the prediction.

    We target the last conv block of EfficientNet-B4: model.features[-1]
    This is the deepest layer before the classifier — it has the most
    semantically rich features, making it the best layer to visualize.

    Returns a base64-encoded PNG string so it can be sent directly
    in a JSON API response without writing anything to disk.
    """

    # EfficientNet-B4 target layer — last block of the feature extractor
    # This is where spatial + semantic information is richest
    target_layer = [model.features[-1]]

    # Prepare image as float numpy array in range [0, 1]
    # show_cam_on_image requires this format to overlay the heatmap
    img_resized = image.resize((224, 224))
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0

    # Preprocess tensor for the model (same as inference_transforms)
    from src.predict import inference_transforms
    tensor = inference_transforms(image).unsqueeze(0).to(device)

    # GradCAM context — targets the predicted class
    # ClassifierOutputTarget tells Grad-CAM which class score to
    # backpropagate from. Without this it defaults to the highest score,
    # which is fine but being explicit is better practice.
    targets = [ClassifierOutputTarget(predicted_class)]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # remove batch dim -> [224, 224]

    # Overlay heatmap on original image
    # show_cam_on_image blends the jet colormap heatmap with the original
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # Convert to base64 PNG so it can travel in JSON
    pil_img    = Image.fromarray(visualization)
    buffer     = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64_string