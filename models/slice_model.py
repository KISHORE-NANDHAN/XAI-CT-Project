# ============================================================
# File: models/slice_model.py
# Description: 2D CNN model backbone for slice-level classification
# ============================================================

import torch
import torch.nn as nn
from torchvision import models

def get_model(cfg):
    name = cfg.get("name", "resnet50")
    pretrained = cfg.get("pretrained", True)
    num_classes = cfg.get("num_classes", 2)

    if name.lower() == "resnet50":
        backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name.lower() == "convnext_tiny":
        backbone = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {name}")

    head = nn.Linear(in_features, num_classes)

    class SliceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            # If grayscale → repeat channels, else leave as-is
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            feats = self.backbone(x)
            logits = self.head(feats)
            return logits, feats


    return SliceClassifier()
