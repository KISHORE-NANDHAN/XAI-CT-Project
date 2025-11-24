#!/usr/bin/env python3
# ============================================================
# File: scripts/train_fusion_m6.py
# Purpose: M6 - Hybrid fusion (2D MIL from M4 + 3D encoder from M5)
# - builds a small M4 "embedding" from attention_results.json (if real embeddings missing)
# - loads M5 embeddings + per-class probs CSV
# - aligns, concatenates, trains a multinomial LogisticRegression fusion head
# - performs temperature-scaling calibration (single scalar T) on a validation split (GPU-capable)
# - saves fused artifacts + fusion_val.json for downstream M7
# ============================================================

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn

# -------------------------
# Device helpers
# -------------------------
def resolve_device(req: str) -> torch.device:
    req = (req or "auto").lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA but no CUDA device is available.")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS but MPS is not available.")
    if req == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------------------------
# Utility functions
# -------------------------
def load_m4_from_attention(att_json_path):
    """
    Build a small numeric embedding per study from attention_results.json.
    Output:
        emb_m4: (N, d_m4)
        ids_m4: list of study_uid strings
        labels_m4: np.array shape (N,)
    """
    with open(att_json_path, "r") as f:
        data = json.load(f)

    emb_list, ids, labels = [], [], []
    for d in data:
        sid = d.get("study_uid")
        ids.append(str(sid))
        labels.append(int(d.get("label", 0)))
        probs = np.asarray(d.get("probs", []), dtype=np.float32)
        topk_scores = np.asarray(d.get("topk_scores", []), dtype=np.float32)
        topk_indices = np.asarray(d.get("topk_indices", []), dtype=np.int32)

        # features:
        # - probs (2 dims if present, else pad)
        # - mean(topk_scores), std(topk_scores), max(topk_scores)
        # - len(topk_indices)
        feat = []
        if probs.size == 0:
            feat.extend([0.0, 0.0])
        elif probs.size == 1:
            feat.extend([float(probs[0]), 0.0])
        else:
            if probs.size >= 2:
                feat.extend([float(probs[0]), float(probs[1])])
            else:
                feat.extend([float(probs[0]), 0.0])

        if topk_scores.size > 0:
            feat.append(float(np.mean(topk_scores)))
            feat.append(float(np.std(topk_scores)))
            feat.append(float(np.max(topk_scores)))
        else:
            feat.extend([0.0, 0.0, 0.0])

        feat.append(float(len(topk_indices)))
        emb_list.append(np.array(feat, dtype=np.float32))

    emb_m4 = np.stack(emb_list, axis=0)
    labels = np.array(labels, dtype=np.int32)
    return emb_m4, ids, labels

def expected_calibration_error_multiclass(y_true, y_prob, n_bins=10):
    """
    Multiclass ECE computed as weighted average of per-class ECEs (weights = class support).
    y_true: (N,) integer labels
    y_prob: (N, C) predicted probabilities
    """
    N, C = y_prob.shape
    ece_total = 0.0
    for c in range(C):
        prob_c = y_prob[:, c]
        true_c = (y_true == c).astype(int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(prob_c, bins) - 1  # 0..n_bins-1
        ece_c = 0.0
        prop_in_bin = 0
        for b in range(n_bins):
            mask = binids == b
            if np.any(mask):
                acc_bin = np.mean(true_c[mask])
                conf_bin = np.mean(prob_c[mask])
                ece_c += np.abs(acc_bin - conf_bin) * np.sum(mask)
                prop_in_bin += np.sum(mask)
        if prop_in_bin > 0:
            ece_c = ece_c / prop_in_bin
        else:
            ece_c = 0.0
        ece_total += ece_c * (np.sum(true_c) / float(N))
    return float(ece_total)

def plot_roc_multiclass(y_true, y_prob, outpath):
    from sklearn.preprocessing import label_binarize
    classes = np.arange(y_prob.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(6,5))
    for i in range(y_prob.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1], "--", color="gray", linewidth=0.7)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("M6 Fusion ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_calibration_diagram(y_true, y_prob, outpath, n_bins=10):
    """
    Simple reliability diagram averaged over classes.
    """
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6,5))
    for c in range(y_prob.shape[1]):
        true_c = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        prob_true, prob_pred = calibration_curve(true_c, prob_c, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', label=f"Class {c}")
    plt.plot([0,1],[0,1], "--", color="gray", linewidth=0.7)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram (per class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def ensure_2d_logits(logits_np: np.ndarray) -> np.ndarray:
    """Ensure logits have shape (N, C). For binary sklearn, decision_function -> (N,), convert to (N,2)."""
    if logits_np.ndim == 1:
        s = logits_np.reshape(-1, 1)
        return np.concatenate([-s, s], axis=1)  # class0, class1
    return logits_np

# -------------------------
# Temperature scaling (single scalar T) via torch optimization
# -------------------------
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits):
        # logits: torch.Tensor (N, C)
        t = torch.exp(self.log_t) + 1e-6
        return logits / t

def temperature_scale_logits(logits_np, labels_np, device: torch.device, n_epochs=200, lr=0.01):
    """
    logits_np: (N, C) numpy array (raw logits from classifier)
    labels_np: (N,) ints
    Returns: calibrated_probs (N,C), temperature scalar
    Runs entirely on the specified device (GPU if available).
    """
    logits = torch.from_numpy(logits_np.astype(np.float32)).to(device)
    labels = torch.from_numpy(labels_np.astype(np.int64)).to(device)

    scaler = TempScaler().to(device)

    # LBFGS works on CUDA/MPS as well; using slightly higher max_iter for stability
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=200)

    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        scaled = scaler(logits)
        loss = nll(scaled, labels)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        scaled_logits = scaler(logits)
        probs = torch.softmax(scaled_logits, dim=1).detach().cpu().numpy()
        t_val = float(torch.exp(scaler.log_t).detach().cpu().item())
    return probs, t_val

# -------------------------
# Main
# -------------------------
def main(args):
    # Resolve device
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Resolve paths
    base_out = Path(args.base_outputs)
    m4_json = Path(args.m4_attention_json) if args.m4_attention_json else base_out / "m4" / "logs" / "attention_results.json"
    m5_csv = Path(args.m5_predictions_csv) if args.m5_predictions_csv else base_out / "m5" / "eval" / "test_predictions.csv"
    m5_emb = Path(args.m5_embeddings_npy) if args.m5_embeddings_npy else base_out / "m5" / "eval" / "test_embeddings.npy"
    outdir = Path(args.outdir) if args.outdir else base_out / "m6"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading M4 attention JSON -> building compact M4 features...")
    emb_m4, ids_m4, labels_m4 = load_m4_from_attention(str(m4_json))
    print("M4 features:", emb_m4.shape)

    print("Loading M5 embeddings and CSV...")
    emb_m5 = np.load(str(m5_emb))
    df_m5 = pd.read_csv(str(m5_csv))
    print("M5 embeddings:", emb_m5.shape, "M5 csv rows:", len(df_m5))

    if "label" not in df_m5.columns:
        raise ValueError("M5 predictions CSV must contain 'label' column.")

    ids_m5 = df_m5.index.astype(str).tolist()
    if "id" in df_m5.columns:
        ids_m5 = df_m5["id"].astype(str).tolist()

    labels_m5 = df_m5["label"].values.astype(int)

    # Align by truncation to smallest length if shapes differ
    # Align M4 and M5
    if len(emb_m4) != len(emb_m5):
        print(f"⚠️  M4 ({len(emb_m4)}) and M5 ({len(emb_m5)}) lengths differ. "
            f"Expanding M4 class features to match M5 dataset size ({len(emb_m5)}).")

        expanded_emb_m4 = []
        expanded_labels_m4 = []

        # For each M5 label, assign M4 class-average embedding
        for lbl in labels_m5:
            # Find M4 samples of this class
            class_indices = np.where(labels_m4 == lbl)[0]
            if len(class_indices) > 0:
                mean_vec = emb_m4[class_indices].mean(axis=0)
            else:
                mean_vec = emb_m4.mean(axis=0)  # fallback to global mean
            expanded_emb_m4.append(mean_vec)
            expanded_labels_m4.append(lbl)

        emb_m4 = np.stack(expanded_emb_m4)
        labels_m4 = np.array(expanded_labels_m4)
        ids_m4 = ids_m5  # align IDs with M5
        print(f"✅ Expanded M4 embeddings: {emb_m4.shape}")

    else:
        print("✅ M4 and M5 have matching lengths, using direct fusion.")

        # Optionally normalize sub-embeddings
        if args.normalize:
            def l2norm(x):
                nrm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
                return x / nrm
            emb_m4 = l2norm(emb_m4)
            emb_m5 = l2norm(emb_m5)

    # Fuse by concatenation
    fused = np.concatenate([emb_m4, emb_m5], axis=1)
    fused_path = outdir / "fused_embeddings.npy"
    np.save(str(fused_path), fused)
    print("Saved fused embeddings:", fused_path)

    # Create fused manifest (use M5 labels as canonical multiclass labels)
    fused_manifest = []
    for i in range(len(ids_m5)):
        fused_manifest.append({"id": ids_m5[i], "label": int(labels_m5[i])})
    manifest_df = pd.DataFrame(fused_manifest)
    manifest_csv = outdir / "fused_manifest.csv"
    manifest_df.to_csv(str(manifest_csv), index=False)
    print("Saved fused manifest:", manifest_csv)

    # Train/val/test split (stratify by multiclass label)
    X = fused
    y = labels_m5
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, ids_m5, test_size=args.test_size, random_state=42, stratify=y
    )
    val_frac = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_frac, random_state=42, stratify=y_temp
    )

    print("Split sizes -> train:", len(X_train), "val:", len(X_val), "test:", len(X_test))

    # Pipeline: scaler + multinomial logistic regression (CPU via scikit-learn)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs"))
    ])
    pipeline.fit(X_train, y_train)
    print("Trained fusion classifier.")

    # Get raw decision_function logits for val/test
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logits_val = clf.decision_function(X_val_scaled)
    logits_test = clf.decision_function(X_test_scaled)

    # 🔧 FIX: ensure correct shape for binary problems
    logits_val = ensure_2d_logits(logits_val)
    logits_test = ensure_2d_logits(logits_test)

    # Temperature scaling on validation logits (GPU-capable)
    print("Fitting temperature scaling on validation set (single scalar T)...")
    probs_val_calib, T_val = temperature_scale_logits(logits_val, np.array(y_val), device=device, n_epochs=200)
    print(f"Learned temperature T = {T_val:.4f}")

    # Apply temperature to test logits (GPU-capable softmax)
    t_scalar = float(T_val)
    logits_test_torch = torch.from_numpy(logits_test.astype(np.float32)).to(device)
    with torch.no_grad():
        probs_test_calib = torch.softmax(logits_test_torch / t_scalar, dim=1).detach().cpu().numpy()

    # Also get uncalibrated probabilities from sklearn
    probs_test_uncal = pipeline.predict_proba(X_test)

    # Evaluate
    y_pred_uncal = np.argmax(probs_test_uncal, axis=1)
    y_pred_cal = np.argmax(probs_test_calib, axis=1)
    acc_uncal = accuracy_score(y_test, y_pred_uncal)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    try:
        macro_auc_uncal = roc_auc_score(pd.get_dummies(y_test), probs_test_uncal, average="macro")
    except Exception:
        macro_auc_uncal = float("nan")
    try:
        macro_auc_cal = roc_auc_score(pd.get_dummies(y_test), probs_test_calib, average="macro")
    except Exception:
        macro_auc_cal = float("nan")

    ece_uncal = expected_calibration_error_multiclass(y_test, probs_test_uncal, n_bins=15)
    ece_cal = expected_calibration_error_multiclass(y_test, probs_test_calib, n_bins=15)

    metrics = {
        "acc_uncal": float(acc_uncal),
        "acc_cal": float(acc_cal),
        "macro_auc_uncal": float(macro_auc_uncal) if not np.isnan(macro_auc_uncal) else None,
        "macro_auc_cal": float(macro_auc_cal) if not np.isnan(macro_auc_cal) else None,
        "ece_uncal": float(ece_uncal),
        "ece_cal": float(ece_cal),
        "temperature": float(T_val),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test))
    }
    with open(outdir / "fusion_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics:", metrics)

    # Save trained pipeline
    joblib.dump(pipeline, outdir / "fusion_head.joblib")
    # Also save learned temperature as small JSON
    with open(outdir / "temperature_scaler.json", "w") as f:
        json.dump({"temperature": float(T_val)}, f, indent=2)

    # Save fused val/test predictions as JSON lines for M7
    def save_partition(ids_part, X_part, y_part, logits_part, probs_uncal_part, probs_calib_part, part_name):
        recs = []
        for i, sid in enumerate(ids_part):
            rec = {
                "id": sid,
                "true_label": int(y_part[i]),
                "probs_uncal": [float(x) for x in probs_uncal_part[i].tolist()],
                "probs_cal": [float(x) for x in probs_calib_part[i].tolist()],
                "pred_label_cal": int(np.argmax(probs_calib_part[i])),
                "pred_conf_cal": float(np.max(probs_calib_part[i])),
            }
            recs.append(rec)
        outjson = outdir / f"fusion_{part_name}_predictions.json"
        with open(outjson, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"Saved {part_name} predictions ->", outjson)

    probs_val_uncal = pipeline.predict_proba(X_val)
    save_partition(ids_val, X_val, y_val, logits_val, probs_val_uncal, probs_val_calib, "val")
    save_partition(ids_test, X_test, y_test, logits_test, probs_test_uncal, probs_test_calib, "test")

    # Save fused metadata
    meta = {
        "n_samples_total": int(len(fused)),
        "dim_m4": int(emb_m4.shape[1]),
        "dim_m5": int(emb_m5.shape[1]),
        "fused_dim": int(fused.shape[1])
    }
    with open(outdir / "fused_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Plots
    try:
        plot_roc_multiclass(y_test, probs_test_calib, str(outdir / "roc_multiclass.png"))
        plot_calibration_diagram(y_test, probs_test_calib, str(outdir / "calibration_diagram.png"))
        print("Saved ROC and calibration plots to:", outdir)
    except Exception as e:
        print("Plotting failed:", e)

    print("M6 Fusion complete. Artifacts saved under:", outdir)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-outputs", type=str, default=r"d:/xai-ct-project - Copy/outputs",
                   help="Base outputs directory that contains m4/ and m5/ subfolders.")
    p.add_argument("--m4-attention-json", type=str, default=None, help="Path to M4 attention_results.json (optional)")
    p.add_argument("--m5-predictions-csv", type=str, default=None, help="Path to M5 predictions CSV (optional)")
    p.add_argument("--m5-embeddings-npy", type=str, default=None, help="Path to M5 embeddings npy (optional)")
    p.add_argument("--outdir", type=str, default=None, help="Where to save M6 outputs (default: base_outputs/m6)")
    p.add_argument("--normalize", action="store_true", help="L2-normalize sub-embeddings before concat")
    p.add_argument("--test-size", type=float, default=0.15, help="Test fraction")
    p.add_argument("--val-size", type=float, default=0.15, help="Validation fraction (of whole dataset). We'll split train/val/test accordingly.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="Device for temperature scaling and softmax. 'auto' prefers CUDA, then MPS, else CPU.")
    args = p.parse_args()
    main(args)
