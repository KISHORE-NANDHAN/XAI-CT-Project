# ============================================================
# File: models/resnet2d.py
# Description: 2D CNN Architectures for COVID-19 CT Classification
# Supports: ResNet18/34/50, DenseNet121, EfficientNetB0
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


class ResNet2D(nn.Module):
    """
    A flexible 2D CNN wrapper supporting ResNet/DenseNet/EfficientNet backbones.
    """

    def __init__(self, architecture="resnet50", num_classes=2, in_channels=3, dropout=0.3, pretrained=True):
        super(ResNet2D, self).__init__()

        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.pretrained = pretrained

        # Load backbone dynamically
        self.backbone, in_features = self._get_backbone()

        # Replace first conv layer if in_channels != 3
        if in_channels != 3:
            self._adjust_input_channels()

        # Replace classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def _get_backbone(self):
        """
        Load backbone and return (backbone, num_features)
        """
        if self.architecture == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if self.pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif self.architecture == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1" if self.pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif self.architecture == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if self.pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif self.architecture == "densenet121":
            model = models.densenet121(weights="IMAGENET1K_V1" if self.pretrained else None)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()
        elif self.architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1" if self.pretrained else None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        return model, in_features

    def _adjust_input_channels(self):
        """
        Adjust the first conv layer to support grayscale input.
        """
        first_conv = None
        if "resnet" in self.architecture:
            first_conv = self.backbone.conv1
        elif "densenet" in self.architecture:
            first_conv = self.backbone.features.conv0
        elif "efficientnet" in self.architecture:
            first_conv = self.backbone.features[0][0]

        if first_conv is not None:
            new_conv = nn.Conv2d(
                self.in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )

            if self.in_channels == 1:  # grayscale → RGB averaging
                new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
            else:
                new_conv.weight.data[:, :3, :, :] = first_conv.weight.data

            if "resnet" in self.architecture:
                self.backbone.conv1 = new_conv
            elif "densenet" in self.architecture:
                self.backbone.features.conv0 = new_conv
            elif "efficientnet" in self.architecture:
                self.backbone.features[0][0] = new_conv

    def forward(self, x):
        if "resnet" in self.architecture:
            features = self.backbone(x)
        elif "densenet" in self.architecture:
            features = self.backbone(x)
        elif "efficientnet" in self.architecture:
            features = self.backbone(x)
        else:
            raise ValueError("Unsupported architecture for forward pass")

        out = self.classifier(features)
        return out


# ============================================================
# Helper: Build model from config
# ============================================================
def build_model_from_config(cfg):
    """
    Utility to instantiate model from YAML config.
    """
    model = ResNet2D(
        architecture=cfg["model"]["architecture"],
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"]
    )
    return model


# ============================================================
# Debug: Run standalone test
# ============================================================
if __name__ == "__main__":
    dummy_cfg = {
        "model": {
            "architecture": "resnet50",
            "num_classes": 2,
            "in_channels": 3,
            "dropout": 0.3,
            "pretrained": False
        }
    }
    model = build_model_from_config(dummy_cfg)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")  # Expected: [2, 2]
