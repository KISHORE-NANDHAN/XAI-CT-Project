# ============================================================
# File: main_train_slice.py
# Description: Train baseline 2D slice classifier (M3) — GPU + AMP
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import yaml, os, json
from tqdm import tqdm
from scripts.slice_dataset import CovidSliceDataset
from models.slice_model import get_model


# -----------------------------
# Utility: Config Loader
# -----------------------------
def load_config(path="config/slice_classifier.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Training Function (with AMP)
# -----------------------------
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, preds, labels = 0, [], []
    for imgs, targets in tqdm(dataloader, desc="Train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # ✅ Mixed precision
            outputs, _ = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds += outputs.softmax(1)[:, 1].detach().cpu().tolist()
        labels += targets.cpu().tolist()

    return total_loss / len(dataloader), roc_auc_score(labels, preds)


# -----------------------------
# Validation Function (with AMP)
# -----------------------------
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Val", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs, _ = model(imgs)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds += outputs.softmax(1)[:, 1].cpu().tolist()
            labels += targets.cpu().tolist()

    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, [1 if p > 0.5 else 0 for p in preds])
    acc = accuracy_score(labels, [1 if p > 0.5 else 0 for p in preds])
    return total_loss / len(dataloader), auc, f1, acc


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["embeddings_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["logs_dir"], exist_ok=True)

    # Datasets
    train_set = CovidSliceDataset(cfg["data"]["dataset_roots"], split="train", img_size=cfg["data"]["img_size"])
    val_set = CovidSliceDataset(cfg["data"]["dataset_roots"], split="val", img_size=cfg["data"]["img_size"])
    train_loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # Model
    model = get_model(cfg["model"]).to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    scaler = torch.cuda.amp.GradScaler()  # ✅ for mixed precision
    best_auc = 0
    log_data = []

    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        tr_loss, tr_auc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auc, val_f1, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train AUC: {tr_auc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        log_data.append({
            "epoch": epoch+1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_acc": val_acc
        })

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(cfg["output"]["checkpoints_dir"], "best_model.pt"))
            print(f"💾 Saved new best model (AUC={best_auc:.4f})")

        torch.cuda.empty_cache()  # ✅ helps with VRAM reuse

    # Save logs
    with open(os.path.join(cfg["output"]["logs_dir"], "train_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n✅ Training complete. Best Validation AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
