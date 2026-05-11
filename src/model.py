import torch
import torch.nn as nn
from torchvision import models
from src.config import EFFICIENTNET_WEIGHTS

def build_model(num_classes=5, pretrained=True):
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    if pretrained:
        state_dict = torch.load(EFFICIENTNET_WEIGHTS, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully")
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    
    return model