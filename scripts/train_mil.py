# ============================================================
# File: scripts/train_mil.py
# Description: Train MIL / Transformer Aggregator for M4 phase
# ============================================================

import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mil_transformer import build_aggregator
from scripts.prepare_mil_data import StudyEmbeddingDataset, collate_fn


# ============================================================
# Training Utilities
# ============================================================

def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return {"acc": acc, "f1": f1, "auc": auc}


# ============================================================
# Training / Evaluation Functions
# ============================================================

def run_epoch(model, dataloader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []

    for embs, labels, uids, _ in tqdm(dataloader, desc="Train" if train else "Val"):
        embs, labels = embs.to(device), labels.to(device)
        logits, _ = model(embs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * embs.size(0)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(dataloader.dataset)
    metrics["loss"] = avg_loss
    return metrics


# ============================================================
# Main Training Function
# ============================================================

def main(cfg_path, device="cuda"):
    cfg = load_cfg(cfg_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # --- Directories ---
    output_dir = Path(cfg["output"]["checkpoints_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["output"]["logs_dir"]) / "mil_train_log.json"
    Path(cfg["output"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    # --- Model ---
    print("🔹 Building aggregator...")
    model = build_aggregator(cfg["aggregator"]).to(device)

    # --- Datasets ---
    print("📂 Loading datasets...")
    train_list = Path("data/mil_prepared/train_list.json")
    val_list = Path("data/mil_prepared/val_list.json")

    train_ds = StudyEmbeddingDataset(train_list.parent / "train", max_slices=cfg["aggregator"].get("max_slices", 256))
    val_ds = StudyEmbeddingDataset(val_list.parent / "val", max_slices=cfg["aggregator"].get("max_slices", 256))

    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, collate_fn=collate_fn)

    # --- Training setup ---
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.05))

    best_auc, history = 0.0, []

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(1, epochs + 1):
        print(f"\n🌍 Epoch {epoch}/{epochs}")
        train_metrics = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_dl, criterion, optimizer, device, train=False)

        print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, f1={train_metrics['f1']:.4f}")
        print(f" Val : loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}, f1={val_metrics['f1']:.4f}, auc={val_metrics['auc']:.4f}")

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Save best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"✅ Best model updated (AUC={best_auc:.4f})")

        # Save periodic checkpoints
        if epoch % 2 == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch}.pt")

        # Log progress
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\n🎯 Training complete! Best AUC =", best_auc)
    print(f"📦 Saved final model → {output_dir / 'best_model.pt'}")
    print(f"🧾 Log file → {log_path}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIL / Transformer Aggregator (M4)")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g., config/mil.yaml)")
    parser.add_argument("--device", default="cuda", help="Computation device")
    args = parser.parse_args()

    main(args.config, args.device)
