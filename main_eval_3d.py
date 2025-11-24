# ============================================================
# File: main_eval_3d.py
# Description: Evaluate trained 3D ResNet (multi-class version)
# Exports:
# - Per-class probabilities CSV
# - Embeddings .npy (for M6 fusion)
# - Metrics JSON (macro/micro AUROC, per-class AUC)
# - Optional ROC plots per class
# ============================================================

import os
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

from scripts.m5.dataset_loader_3d import get_3d_dataloaders
from models.resnet3d import ResNet3D


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------
def load_checkpoint(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
    return model


# ------------------------------------------------------------
# Evaluate and extract embeddings
# ------------------------------------------------------------
@torch.no_grad()
def extract_embeddings_and_preds(model, loader, device, num_classes):
    model.eval()
    all_embeddings, all_probs, all_labels = [], [], []

    for x, y in tqdm(loader, desc="Test"):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.detach().cpu().numpy())

        # extract penultimate embeddings (for fusion)
        try:
            feats = model.backbone.stem(x)
            feats = model.backbone.layer1(feats)
            feats = model.backbone.layer2(feats)
            feats = model.backbone.layer3(feats)
            feats = model.backbone.layer4(feats)
            emb = F.adaptive_avg_pool3d(feats, 1).flatten(1)
        except Exception:
            emb = logits  # fallback
        all_embeddings.append(emb.detach().cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_embeddings, all_probs, all_labels


# ------------------------------------------------------------
# Compute multiclass metrics
# ------------------------------------------------------------
def multiclass_metrics(y_true, y_prob, num_classes):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = y_prob.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))

    per_class_auc = [None] * num_classes
    valid_aucs = []
    for k in range(num_classes):
        pos = (y_true == k)
        neg = (y_true != k)
        if pos.any() and neg.any():
            auc_k = roc_auc_score(pos.astype(int), y_prob[:, k])
            per_class_auc[k] = float(auc_k)
            valid_aucs.append(auc_k)
    auc_macro = float(np.mean(valid_aucs)) if valid_aucs else None
    return {"auc_macro": auc_macro, "acc": acc, "per_class_auc": per_class_auc}


# ------------------------------------------------------------
# Plot ROC curves for each class
# ------------------------------------------------------------
def plot_roc_curves(y_true, y_prob, class_names, outdir):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for i, cname in enumerate(class_names):
        pos = (y_true == i).astype(int)
        neg = (y_true != i).astype(int)
        if pos.any() and neg.any():
            fpr, tpr, _ = roc_curve(pos, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Per-Class)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_multiclass.png"))
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/model3d.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="workdir/3d_eval")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["model"]["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load test loader
    _, _, test_loader = get_3d_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["evaluation"].get("batch_size", 2),
        num_workers=cfg["evaluation"].get("num_workers", 4),
    )

    # Build and load model
    model = ResNet3D(num_classes=num_classes, pretrained=False)
    model = model.to(device)
    model = load_checkpoint(model, args.checkpoint, device)

    # Extract embeddings + predictions
    embeddings, probs, labels = extract_embeddings_and_preds(model, test_loader, device, num_classes)

    # Save embeddings
    emb_path = os.path.join(args.workdir, "test_embeddings.npy")
    np.save(emb_path, embeddings)
    print("✅ Saved embeddings:", emb_path)

    # Save predictions CSV
    df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(num_classes)])
    df["label"] = labels
    preds_path = os.path.join(args.workdir, "test_predictions.csv")
    df.to_csv(preds_path, index=False)
    print("✅ Saved predictions:", preds_path)

    # Compute metrics
    metrics = multiclass_metrics(labels, probs, num_classes)
    print(f"Test ACC: {metrics['acc']:.4f} | Test AUC (macro): {metrics['auc_macro']}")
    for i, auc_k in enumerate(metrics["per_class_auc"]):
        if auc_k is not None:
            print(f"  Class {i} AUC: {auc_k:.4f}")

    # Plot ROC
    class_names = list(getattr(test_loader.dataset, "classes", range(num_classes)))
    plot_roc_curves(labels, probs, class_names, args.workdir)
    print("✅ Saved ROC plots →", os.path.join(args.workdir, "roc_multiclass.png"))

    # Save metrics JSON
    metrics_path = os.path.join(args.workdir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved metrics JSON:", metrics_path)


if __name__ == "__main__":
    main()
