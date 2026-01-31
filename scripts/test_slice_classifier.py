# ============================================================
# File: scripts/test_slice_classifier.py
# Description: Test baseline 2D slice classifier (M3)
# ============================================================

import torch
import yaml
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm

# Adjust path to allow imports from root
import sys
sys.path.append(os.getcwd())

from scripts.slice_dataset import CovidSliceDataset
from models.slice_model import get_model

# -----------------------------
# Utility: Config Loader
# -----------------------------
def load_config(path="config/slice_classifier.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds_probs = []
    labels_list = []
    
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Testing"):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs, _ = model(imgs)
            probs = outputs.softmax(1)[:, 1]  # Probability of class 1 (COVID)
            
            preds_probs.extend(probs.cpu().tolist())
            labels_list.extend(targets.cpu().tolist())

    return np.array(labels_list), np.array(preds_probs)

# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Device: {device}")

    # 1. Load Data
    print("📂 Loading Test Data...")
    dataset_roots = cfg["data"]["dataset_roots"]
    img_size = cfg["data"]["img_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = 0 # Force 0 workers on Windows to avoid RuntimeError

    test_set = CovidSliceDataset(dataset_roots, split="test", img_size=img_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if len(test_set) == 0:
        print("❌ No test data found! Please check your manifests or dataset paths.")
        return

    # 2. Load Model
    print("🧠 Loading Model...")
    model = get_model(cfg["model"]).to(device)
    
    checkpoint_path = os.path.join(cfg["output"]["checkpoints_dir"], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ Model loaded from {checkpoint_path}")

    # 3. Run Inference
    print("🏃 Running Inference...")
    y_true, y_probs = evaluate(model, test_loader, device)
    y_pred = (y_probs > 0.5).astype(int)

    # 4. Metrics
    print("\n" + "="*40)
    print("📊 TEST RESULTS")
    print("="*40)

    # Classification Report
    print("\n🔹 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-COVID", "COVID"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n🔹 Confusion Matrix:")
    print(cm)

    # AUC
    try:
        auc = roc_auc_score(y_true, y_probs)
        print(f"\n🔹 AUROC: {auc:.4f}")
    except ValueError:
        print("\n🔹 AUROC: Undefined (only one class present in test set)")

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"🔹 Accuracy: {acc:.4f}")

    # 5. Save Plots (Optional)
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-COVID", "COVID"], yticklabels=["Non-COVID", "COVID"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\n✅ Confusion matrix plot saved to {cm_path}")

    # Save ROC Curve Plot
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = os.path.join(output_dir, "test_roc_curve.png")
        plt.savefig(roc_path)
        print(f"✅ ROC curve plot saved to {roc_path}")
    
    print("\n🎉 Testing Complete.")

if __name__ == "__main__":
    main()
