# ============================================================
# File: scripts/plot_training_curves.py
# Description: Parse console log or metrics file and plot
#              (1) Accuracy vs Epoch
#              (2) Loss vs Epoch
# ============================================================

import re
import matplotlib.pyplot as plt

log_file = "outputs/m5/training_log.txt"  # <-- save your console output here

epochs, train_loss, val_loss, train_auc, val_auc = [], [], [], [], []

with open(log_file, "r") as f:
    for line in f:
        # match epoch number
        m_epoch = re.match(r"Epoch (\d+)/(\d+)", line)
        if m_epoch:
            epochs.append(int(m_epoch.group(1)))

        # match training and validation lines
        if "Train loss" in line:
            parts = line.strip().split()
            train_loss.append(float(parts[2]))
            train_auc.append(float(parts[-1]))
        if "Val loss" in line:
            parts = line.strip().split()
            val_loss.append(float(parts[2]))
            val_auc.append(float(parts[-1]))

# ----------------------------
# Plot 1: Loss vs Epoch
# ----------------------------
plt.figure(figsize=(7,5))
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_loss, label="Val Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Validation Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/m5/train_val_loss.png")
plt.show()

# ----------------------------
# Plot 2: AUC vs Epoch (Accuracy proxy)
# ----------------------------
plt.figure(figsize=(7,5))
plt.plot(epochs, train_auc, label="Train AUC", marker='o')
plt.plot(epochs, val_auc, label="Val AUC", marker='s')
plt.xlabel("Epoch")
plt.ylabel("AUC (Accuracy proxy)")
plt.title("Train/Validation Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/m5/train_val_auc.png")
plt.show()

print("✅ Saved plots in outputs/m5/:")
print(" - train_val_loss.png")
print(" - train_val_auc.png")
