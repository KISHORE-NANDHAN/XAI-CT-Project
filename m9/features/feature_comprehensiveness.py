#!/usr/bin/env python3
"""
Feature-level Comprehensiveness for fused embeddings.

For each study (embedding vector):
  - compute original prediction and predicted class probability
  - find top-k features (by importance)
  - mask them (set to dataset median) -> compute probability of original predicted class
  - store prob_drop = orig_prob - masked_prob

Outputs:
  outputs/m9/feature_comprehensiveness.json
  outputs/m9/feature_comprehensiveness_summary.json

Usage:
  python m9/feature_comprehensiveness.py --config config/m9.yaml
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

# -----------------------
def load_embeddings(path):
    return np.load(path)  # (N, D)

def load_manifest(manifest_path):
    import pandas as pd
    if Path(manifest_path).exists():
        return pd.read_csv(manifest_path)
    # try json sibling
    j = Path(manifest_path).with_suffix(".json")
    if j.exists():
        import json
        return json.load(open(j))
    raise FileNotFoundError("Manifest not found: " + str(manifest_path))

def load_model(path, device="cpu"):
    # expects TorchScript that accepts a FloatTensor (B, D) and returns probs (B, C)
    m = torch.jit.load(path, map_location=device)
    m.eval()
    return m

# -----------------------
def feature_importance_from_weights(ts_model):
    """
    Try to extract input-layer weights from TorchScript model.
    Fallback: compute simple global variance measure per feature.
    Returns importance vector (D,) normalized descending.
    """
    # Try to access the torchscript module's parameters
    try:
        params = {k: v for k, v in ts_model.named_parameters()}
        # heuristic: find the first parameter with 2 dims where second dim equals input dim
        for name, p in params.items():
            if p.ndim == 2:
                # p.shape: (out_dim, in_dim)
                imp = np.abs(p.detach().cpu().numpy()).mean(axis=0)
                if imp.sum() > 0:
                    return imp / (imp.sum() + 1e-12)
    except Exception:
        pass
    # fallback: return uniform importance
    logging.warning("Could not extract weight-based importance from model; using variance of embeddings fallback.")
    return None

# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    parser.add_argument("--manifest", default=None, help="override manifest path")
    parser.add_argument("--topk", type=float, default=None, help="override topk fraction (e.g. 0.1)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    emb_path = cfg.get("embeddings_path", "outputs/m6/fused_embeddings.npy")
    manifest_path = args.manifest or cfg.get("manifest", "outputs/m6/results/fused_manifest_enhanced.csv")
    model_path = cfg.get("model_path")
    if args.device:
        device = args.device
    else:
        device = cfg.get("device","cpu")
    outdir = cfg.get("out_dir","outputs/m9")
    os.makedirs(outdir, exist_ok=True)

    topk = args.topk if args.topk is not None else cfg.get("topk", 0.1)

    logging.info("Loading embeddings: %s", emb_path)
    X = load_embeddings(emb_path)   # (N,D)
    N,D = X.shape
    logging.info("Embeddings shape: %s", X.shape)

    logging.info("Loading manifest: %s", manifest_path)
    # manifest is only used for ids mapping; try to read if possible
    try:
        import pandas as pd
        df = pd.read_csv(manifest_path)
        ids = df.get("study").tolist() if "study" in df.columns else df.index.astype(str).tolist()
    except Exception:
        ids = [f"study_{i}" for i in range(N)]

    logging.info("Loading TorchScript model: %s", model_path)
    model = load_model(model_path, device=device)

    # compute dataset median per feature (for masking)
    med = np.median(X, axis=0)

    # compute importance vector from model weights if possible
    imp = feature_importance_from_weights(model)
    if imp is None:
        # fallback to global variance of features
        var = np.var(X, axis=0)
        imp = var / (var.sum() + 1e-12)

    # order features descending
    feat_order = np.argsort(imp)[::-1]

    k = max(1, int(np.floor(D * float(topk))))
    logging.info("Using topk=%s => top features count = %d / %d", topk, k, D)

    results = {"results":[]}

    # iterate each study
    for i in range(N):
        emb = X[i:i+1].astype(np.float32)   # (1,D)
        orig = torch.tensor(emb).to(device)
        with torch.no_grad():
            probs = model(orig).cpu().numpy()   # (1,C)
        probs = probs.squeeze(0)
        pred_class = int(probs.argmax())
        orig_prob = float(probs[pred_class])

        # mask top-k features -> set to median
        emb_masked = emb.copy()
        emb_masked[0, feat_order[:k]] = med[feat_order[:k]]

        with torch.no_grad():
            probs_masked = model(torch.tensor(emb_masked).to(device)).cpu().numpy().squeeze(0)
        masked_prob = float(probs_masked[pred_class])

        prob_drop = orig_prob - masked_prob

        res = {
            "study_id": ids[i] if i < len(ids) else f"idx_{i}",
            "index": int(i),
            "pred_class": int(pred_class),
            "orig_prob": float(orig_prob),
            "masked_prob": float(masked_prob),
            "prob_drop": float(prob_drop),
            "topk_count": int(k)
        }
        results["results"].append(res)

    # summary stats
    drops = [r["prob_drop"] for r in results["results"]]
    summary = {
        "mean_drop": float(np.mean(drops)),
        "median_drop": float(np.median(drops)),
        "std_drop": float(np.std(drops)),
        "n": len(drops)
    }

    # save
    with open(os.path.join(outdir, "feature_comprehensiveness.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(outdir, "feature_comprehensiveness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved feature_comprehensiveness results to %s", outdir)

if __name__ == "__main__":
    main()
