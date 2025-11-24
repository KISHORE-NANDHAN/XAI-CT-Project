# ============================================================
# File: main_train_3d.py
# Description: Training loop for lightweight 3D ResNet (M5)
# Optimized for RTX 3050 (6GB)
# ============================================================

import os
import argparse
import yaml
import random
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------- GPU / CUDA PERFORMANCE SETTINGS ----------------
import torch.backends.cudnn as cudnn
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True
cudnn.deterministic = False
# -----------------------------------------------------------------

# local imports
from scripts.m5.dataset_loader_3d import get_3d_dataloaders
from models.resnet3d import ResNet3D


def multiclass_metrics(all_labels, all_probs, num_classes):
    """
    Robust 4-class metrics:
      - Accuracy: argmax
      - AUC (macro): mean over classes that appear (skip absent)
      - per_class_auc: list with AUC for each class or None
    """
    import numpy as _np
    from sklearn.metrics import roc_auc_score as _roc_auc_score, accuracy_score as _accuracy_score

    y_true = _np.array(all_labels)        # (N,)
    y_prob = _np.array(all_probs)         # (N, C)
    y_pred = y_prob.argmax(axis=1)

    acc = float(_accuracy_score(y_true, y_pred))

    per_class_auc = [None] * num_classes
    valid_aucs = []
    for k in range(num_classes):
        pos = (y_true == k)
        neg = (y_true != k)
        if pos.any() and neg.any():
            try:
                auc_k = _roc_auc_score(pos.astype(int), y_prob[:, k])
                per_class_auc[k] = float(auc_k)
                valid_aucs.append(auc_k)
            except Exception:
                per_class_auc[k] = None

    auc_macro = float(_np.mean(valid_aucs)) if len(valid_aucs) > 0 else None
    return {"auc": auc_macro, "acc": acc, "per_class_auc": per_class_auc}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path):
    torch.save(state, path)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, cfg):
    model.train()
    losses, all_probs, all_labels = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=cfg["training"]["amp"]):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if cfg["training"].get("grad_clip", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        # ✅ keep full distribution for multiclass (B, C)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

        m = multiclass_metrics(all_labels, all_probs, cfg["model"]["num_classes"])
        pbar.set_postfix(loss=np.mean(losses), auc=(m["auc"] if m["auc"] is not None else 0.0))

    m = multiclass_metrics(all_labels, all_probs, cfg["model"]["num_classes"])
    metrics = {
        "loss": float(np.mean(losses)),
        "auc": m["auc"],
        "acc": m["acc"],
    }
    return metrics

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    losses, all_probs, all_labels = [], [], []

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=cfg["training"]["amp"]):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # (B, C)
        all_probs.extend(probs.tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
        losses.append(loss.item())

    m = multiclass_metrics(all_labels, all_probs, cfg["model"]["num_classes"])
    metrics = {
        "loss": float(np.mean(losses)),
        "auc": m["auc"],
        "acc": m["acc"],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/model3d.yaml")
    parser.add_argument("--workdir", type=str, default="workdir/3d")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.workdir, exist_ok=True)
    set_seed(cfg.get("seed", 42))

    # ✅ Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ✅ Data loaders (use pin_memory for CUDA)
    train_loader, val_loader, _ = get_3d_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
    )

    # ✅ Model
    model = ResNet3D(num_classes=cfg["model"]["num_classes"],
                     pretrained=cfg["model"].get("pretrained", False))
    model = model.to(device)

    # ✅ Torch compile for faster training (PyTorch 2.x+)
    print("torch.compile disabled to avoid Triton error.")


    criterion = nn.CrossEntropyLoss()
    lr = float(cfg["training"]["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(cfg["training"].get("weight_decay", 1e-5)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=cfg["training"].get("lr_patience", 3), verbose=True)

    scaler = GradScaler(enabled=cfg["training"]["amp"])
    start_epoch, best_val_auc = 0, -1.0

    # Optional WandB
    use_wandb = False
    if not args.no_wandb and cfg.get("wandb", {}).get("use", False):
        try:
            import wandb
            use_wandb = True
            wandb.init(project=cfg["wandb"]["project"], config=cfg)
            wandb.run.name = Path(args.workdir).name
        except Exception as e:
            print("WandB init failed:", e)

    # Resume
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val_auc = ck.get("best_val_auc", best_val_auc)
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_val_auc={best_val_auc}")

    # ---------------- TRAINING LOOP ----------------
    n_epochs = cfg["training"]["n_epochs"]
    for epoch in range(start_epoch, n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, cfg)
        val_metrics = validate(model, val_loader, criterion, device, cfg)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["auc"] if val_metrics["auc"] is not None else 0.0)
        else:
            scheduler.step()

        print(f"Train loss {train_metrics['loss']:.4f} | Train AUC {train_metrics['auc']}")
        print(f"Val loss   {val_metrics['loss']:.4f} | Val AUC   {val_metrics['auc']}")

        if use_wandb:
            wandb.log({
                "train/loss": train_metrics["loss"],
                "train/auc": train_metrics["auc"],
                "val/loss": val_metrics["loss"],
                "val/auc": val_metrics["auc"],
                "epoch": epoch
            })

        ckpt_path = os.path.join(args.workdir, f"checkpoint_epoch{epoch+1}.pth")
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val_auc": best_val_auc
        }, ckpt_path)

        if val_metrics["auc"] is not None and val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_path = os.path.join(args.workdir, "best.pth")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val_auc": best_val_auc,
                "cfg": cfg
            }, best_path)
            print(f"✅ Saved new best -> {best_path}")

    # Save summary
    summary = {"best_val_auc": best_val_auc, "config": cfg}
    with open(os.path.join(args.workdir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
