#!/usr/bin/env python3
"""
Train a PyTorch multi-class fusion head on fused embeddings.

Inputs:
  - outputs/m6/fused_embeddings.npy
  - outputs/m6/fused_manifest_enhanced.csv  (must contain `label` column)

Outputs:
  - outputs/m6/torch_fusion_best.pth
  - outputs/m6/torch_fusion_history.json
"""
import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--emb", default="outputs/m6/fused_embeddings.npy")
parser.add_argument("--manifest", default="outputs/m6/fused_manifest_enhanced.csv")
parser.add_argument("--label_col", default="label")
parser.add_argument("--out_dir", default="outputs/m6")

parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--patience", type=int, default=8)
parser.add_argument("--device", default="cuda")

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ------------------------------
# Load embeddings
# ------------------------------
X = np.load(args.emb)
if X.ndim != 2:
    raise RuntimeError("Embeddings must be a 2D array (N x D). Got shape: %s" % (X.shape,))
N, D = X.shape

# ------------------------------
# Load labels
# ------------------------------
df = pd.read_csv(args.manifest)
if args.label_col not in df.columns:
    raise RuntimeError(f"Label column '{args.label_col}' not found in manifest. Columns: {df.columns.tolist()}")

labels_raw = df[args.label_col].values

# convert labels → integer classes
le = LabelEncoder()
y = le.fit_transform(labels_raw)
num_classes = len(le.classes_)

print("Detected classes:", le.classes_)
print("Class count:", num_classes)

# ------------------------------
# Split
# ------------------------------
strat = y if len(np.unique(y))>1 else None
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat
)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

train_ds = torch.utils.data.TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_ds = torch.utils.data.TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

# IMPORTANT: num_workers=0 avoids Windows spawn issues you hit earlier
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size*2,
                                         shuffle=False, num_workers=0)

# ------------------------------
# Multi-class fusion head
# ------------------------------
class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)             # logits (N, C)

model = FusionHead(D, args.hidden, num_classes).to(device)

opt = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

best_auc = 0.0
patience = 0
history = []

# ------------------------------
# Training loop
# ------------------------------
for epoch in range(1, args.epochs+1):

    model.train()
    train_loss_vals = []

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()
        train_loss_vals.append(loss.item())

    train_loss = float(np.mean(train_loss_vals)) if train_loss_vals else 0.0

    # Validate
    model.eval()
    all_probs = []
    all_gt = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_gt.append(yb.numpy())

    if len(all_probs) == 0:
        val_auc = 0.0
    else:
        all_probs = np.concatenate(all_probs)
        all_gt    = np.concatenate(all_gt)
        try:
            # multi-class OVR
            if len(np.unique(all_gt)) > 1 and all_probs.shape[1] > 1:
                val_auc = roc_auc_score(all_gt, all_probs, multi_class="ovr")
            else:
                val_auc = 0.0
        except Exception:
            val_auc = 0.0

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_auc": float(val_auc)
    })

    print(f"Epoch {epoch}: loss={train_loss:.4f}  val_auc={val_auc:.4f}")

    # Save best
    if val_auc > best_auc:
        best_auc = val_auc
        patience = 0
        torch.save(model.state_dict(), f"{args.out_dir}/torch_fusion_best.pth")
        print("  ✔ Saved new best model.")
    else:
        patience += 1
        if patience >= args.patience:
            print("  Early stopping.")
            break

with open(f"{args.out_dir}/torch_fusion_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("Training complete. Best AUC:", best_auc)
print("Saved:", f"{args.out_dir}/torch_fusion_best.pth")
