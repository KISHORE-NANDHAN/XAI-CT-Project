#!/usr/bin/env python3
"""
Feature-level Sufficiency for fused embeddings.

For each study (embedding vector):
  - compute original prediction and predicted class probability
  - find top-k features (by importance)
  - keep only them (set other features to dataset median) -> compute preserved probability of predicted class
  - store prob_retained = kept_prob / orig_prob (if orig_prob>0 else 0)

Outputs:
  outputs/m9/feature_sufficiency.json
  outputs/m9/feature_sufficiency_summary.json

Usage:
  python m9/feature_sufficiency.py --config config/m9.yaml
"""
import os, sys, json, argparse, logging, yaml
from pathlib import Path
import numpy as np
import torch

# -----------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def load_config(path="config/m9.yaml"):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def load_embeddings(path):
    return np.load(path)  # (N, D)

def load_model(path, device="cpu"):
    m = torch.jit.load(path, map_location=device)
    m.eval()
    return m

def feature_importance_from_weights(ts_model):
    try:
        params = {k: v for k, v in ts_model.named_parameters()}
        for name, p in params.items():
            if p.ndim == 2:
                imp = np.abs(p.detach().cpu().numpy()).mean(axis=0)
                if imp.sum() > 0:
                    return imp / (imp.sum() + 1e-12)
    except Exception:
        pass
    return None

# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    emb_path = cfg.get("embeddings_path", "outputs/m6/fused_embeddings.npy")
    model_path = cfg.get("model_path")
    device = args.device if args.device is not None else cfg.get("device","cpu")
    outdir = cfg.get("out_dir","outputs/m9")
    os.makedirs(outdir, exist_ok=True)
    topk = args.topk if args.topk is not None else cfg.get("topk", 0.1)

    logging.info("Loading embeddings: %s", emb_path)
    X = load_embeddings(emb_path)
    N,D = X.shape
    logging.info("Loaded embeddings shape: %s", X.shape)

    logging.info("Loading model: %s", model_path)
    model = load_model(model_path, device=device)

    med = np.median(X, axis=0)
    imp = feature_importance_from_weights(model)
    if imp is None:
        var = np.var(X, axis=0)
        imp = var / (var.sum() + 1e-12)

    feat_order = np.argsort(imp)[::-1]
    k = max(1, int(np.floor(D * float(topk))))
    logging.info("topk fraction %s => keep %d features", topk, k)

    results = {"results":[]}
    for i in range(N):
        emb = X[i:i+1].astype(np.float32)
        orig = torch.tensor(emb).to(device)
        with torch.no_grad():
            probs = model(orig).cpu().numpy().squeeze(0)
        pred_class = int(probs.argmax())
        orig_prob = float(probs[pred_class])

        # keep only top-k features: set others to median
        emb_keep = emb.copy()
        other_idx = feat_order[k:]
        if len(other_idx) > 0:
            emb_keep[0, other_idx] = med[other_idx]

        with torch.no_grad():
            probs_keep = model(torch.tensor(emb_keep).to(device)).cpu().numpy().squeeze(0)
        kept_prob = float(probs_keep[pred_class])

        prob_retained = float(kept_prob / orig_prob) if orig_prob > 1e-12 else 0.0

        res = {
            "index": int(i),
            "study_id": f"study_{i}",
            "pred_class": pred_class,
            "orig_prob": orig_prob,
            "kept_prob": kept_prob,
            "prob_retained": prob_retained,
            "kept_count": int(k)
        }
        results["results"].append(res)

    # summary
    retained_vals = [r["prob_retained"] for r in results["results"]]
    summary = {
        "mean_retained": float(np.mean(retained_vals)),
        "median_retained": float(np.median(retained_vals)),
        "std_retained": float(np.std(retained_vals)),
        "n": len(retained_vals)
    }

    with open(os.path.join(outdir, "feature_sufficiency.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(outdir, "feature_sufficiency_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved feature_sufficiency results to %s", outdir)

if __name__ == "__main__":
    main()
