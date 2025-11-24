#!/usr/bin/env python3
"""
Feature-level Insertion/Deletion for fused embeddings.

- Identify top-K% important features (from weight importance)
- Deletion: important dims set to median first → gradually restored
- Insertion: important dims are restored first → gradually add features

Outputs:
  outputs/m9/feature_insertion_deletion.json
  outputs/m9/feature_insertion_deletion_summary.json

Usage:
  python m9/feature_insertion_deletion.py --config config/m9.yaml
"""

import os, sys, json, yaml, argparse, logging
from pathlib import Path
import numpy as np
import torch

# --------------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")

def load_config(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def load_embeddings(path):
    return np.load(path)

def load_model(path, device="cpu"):
    m = torch.jit.load(path, map_location=device)
    m.eval()
    return m

# --------------------------
def extract_feature_importance(model, D, fallback_var):
    """
    Weight-based importance: average absolute value of input layer weights.
    """
    try:
        params = dict(model.named_parameters())
        for k,v in params.items():
            if v.ndim == 2 and v.shape[1] == D:
                imp = np.abs(v.detach().cpu().numpy()).mean(axis=0)
                if imp.sum() > 0:
                    return imp / (imp.sum()+1e-12)
    except:
        pass
    logging.warning("Weight-based importance unavailable → using variance fallback.")
    return fallback_var / (fallback_var.sum()+1e-12)

def auc(xs, ys):
    """Simple trapezoidal AUC."""
    return float(np.trapz(ys, xs))

# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    emb_path = cfg.get("embeddings_path", "outputs/m6/fused_embeddings.npy")
    model_path = cfg.get("model_path")
    outdir = cfg.get("out_dir","outputs/m9")

    device = args.device if args.device else cfg.get("device","cpu")
    steps = args.steps if args.steps else cfg.get("steps", 50)
    topk_frac = cfg.get("topk",0.10)

    os.makedirs(outdir, exist_ok=True)

    logging.info("Loading embeddings: %s", emb_path)
    X = load_embeddings(emb_path)   # (N,D)
    N,D = X.shape

    logging.info("Loading TorchScript: %s", model_path)
    model = load_model(model_path, device=device)

    # median values for masking
    med = np.median(X, axis=0)

    # importance
    var = np.var(X, axis=0)
    imp = extract_feature_importance(model, D, var)
    feat_order = np.argsort(imp)[::-1]  # descending

    k = max(1, int(D * topk_frac))
    important_dims = feat_order[:k]

    logging.info(f"Top-k dims = {k}/{D}, steps={steps}")

    results = []

    # prepare interpolation indices
    step_sizes = np.linspace(0, k, steps, dtype=int)

    for i in range(N):
        emb = X[i].astype(np.float32)
        emb = emb.reshape(1, -1)

        # original prediction
        with torch.no_grad():
            orig_prob_all = model(torch.tensor(emb).to(device)).cpu().numpy()[0]
        pred = orig_prob_all.argmax()
        orig_prob = float(orig_prob_all[pred])

        # Pre-create masked version for deletion
        base = emb.copy()
        base[0, important_dims] = med[important_dims]

        # CURVES
        deletion_curve = []
        insertion_curve = []

        for s in step_sizes:
            dims = important_dims[:s]

            # ----- deletion -----
            emb_del = base.copy()
            emb_del[0, dims] = emb[0, dims]  # restore these dims
            with torch.no_grad():
                p = model(torch.tensor(emb_del).to(device)).cpu().numpy()[0][pred]
            deletion_curve.append(float(p))

            # ----- insertion -----
            emb_ins = med.copy().reshape(1,-1)
            emb_ins = emb_ins.astype(np.float32)
            # keep the unimportant dims always set to med; add important dims gradually
            emb_ins[0, dims] = emb[0, dims]
            with torch.no_grad():
                p2 = model(torch.tensor(emb_ins).to(device)).cpu().numpy()[0][pred]
            insertion_curve.append(float(p2))

        xs = np.linspace(0, 1, steps)
        auc_del = auc(xs, deletion_curve)
        auc_ins = auc(xs, insertion_curve)

        results.append({
            "index": i,
            "pred_class": int(pred),
            "orig_prob": orig_prob,
            "auc_insertion": auc_ins,
            "auc_deletion": auc_del,
            "insertion_curve": insertion_curve,
            "deletion_curve": deletion_curve
        })

    # summary
    mean_ins = float(np.mean([r["auc_insertion"] for r in results]))
    mean_del = float(np.mean([r["auc_deletion"] for r in results]))
    summary = {
        "mean_insertion_auc": mean_ins,
        "mean_deletion_auc": mean_del,
        "n": len(results)
    }

    with open(os.path.join(outdir,"feature_insertion_deletion.json"),"w") as f:
        json.dump({"results":results}, f, indent=2)
    with open(os.path.join(outdir,"feature_insertion_deletion_summary.json"),"w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved insertion/deletion results to %s", outdir)

if __name__ == "__main__":
    main()
