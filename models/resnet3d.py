# ============================================================
# File: models/resnet3d.py
# Fixed: accept 1-channel CT input when pretrained=False
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


class ResNet3D(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, embedding_dim=None):
        super().__init__()
        self.backbone = r3d_18(weights="KINETICS400_V1" if pretrained else None)

        # --- Fix input channels (1 instead of 3) ---
        if not pretrained:
            conv1 = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(
                in_channels=1,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None,
            )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Lightweight MLP head for embedding-type input
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim or 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.ndim == 5:
            return self.backbone(x)
        elif x.ndim == 2:
            return self.mlp_head(x)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class ResNet3D_Embed(nn.Module):
    """For extracting embeddings for fusion."""
    def __init__(self, checkpoint_path=None, embedding_dim=512, device="cuda"):
        super().__init__()
        self.model = ResNet3D(num_classes=2, embedding_dim=embedding_dim)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state.get("model_state", state))
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, x):
        if x.ndim == 5:
            feats = self.model.backbone.stem(x)
            feats = self.model.backbone.layer1(feats)
            feats = self.model.backbone.layer2(feats)
            feats = self.model.backbone.layer3(feats)
            feats = self.model.backbone.layer4(feats)
            pooled = F.adaptive_avg_pool3d(feats, 1).flatten(1)
            return pooled
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")
