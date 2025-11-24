#!/usr/bin/env python3
# ============================================================
# File: scripts/fuse_embeddings.py
# Purpose: Fuse embeddings from M4 (MIL) and M5 (3D) by concat
# Features:
# - supports loading .npy embeddings and optional CSV manifests with ids/labels
# - aligns by sample id (if both manifests provided) or by index fallback
# - outputs fused embeddings .npy and optional classifier training (sklearn)
# - saves metrics and model (joblib)
# ============================================================

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

def load_npy(path):
    return np.load(path)

def load_manifest_csv(path):
    """
    Expect columns: id, filepath(optional), label(optional)
    At minimum 'id' preferred for alignment.
    """
    df = pd.read_csv(path)
    if "id" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "id"})
    if "id" not in df.columns:
        raise ValueError(f"Manifest {path} needs an 'id' or 'filename' column.")
    return df

def align_by_id(df_a, df_b, emb_a, emb_b):
    """
    Align two dataframes (manifests) on 'id' and reorder embeddings accordingly.
    Returns: ids, emb_a_aligned, emb_b_aligned, labels (if available else None)
    """
    merged = df_a.merge(df_b[["id", "label"]].rename(columns={"label": "label_b"}) if "label" in df_b.columns else df_a.merge(df_b, on="id", how="inner"), on="id", how="inner")
    # But simpler: create mapping from id->row index
    map_a = {rid: i for i, rid in enumerate(df_a["id"].tolist())}
    map_b = {rid: i for i, rid in enumerate(df_b["id"].tolist())}
    common = sorted(list(set(map_a.keys()).intersection(set(map_b.keys()))))
    if len(common) == 0:
        raise ValueError("No common ids between manifests.")
    emb_a_aligned = np.stack([emb_a[map_a[r]] for r in common], axis=0)
    emb_b_aligned = np.stack([emb_b[map_b[r]] for r in common], axis=0)
    # labels: prefer label from df_a then df_b
    labels = None
    if "label" in df_a.columns:
        labels = df_a.set_index("id").loc[common]["label"].values
    elif "label" in df_b.columns:
        labels = df_b.set_index("id").loc[common]["label"].values
    return common, emb_a_aligned, emb_b_aligned, labels

def align_by_index(emb_a, emb_b):
    if emb_a.shape[0] != emb_b.shape[0]:
        raise ValueError(f"Embedding counts mismatch and no manifests provided: {emb_a.shape[0]} vs {emb_b.shape[0]}")
    ids = [str(i) for i in range(emb_a.shape[0])]
    return ids, emb_a, emb_b, None

def normalize_emb(emb, eps=1e-8):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norm + eps)

def train_small_head(X, y, model_type="logreg", outdir=".", seed=42):
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score

    X = np.asarray(X)
    y = np.asarray(y)
    if len(set(y)) < 2:
        raise ValueError("Need at least two classes to train classifier.")

    # simple 80/20 split for quick eval
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    pipe.fit(X_tr, y_tr)
    yp = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, yp)
    acc = accuracy_score(y_te, (yp >= 0.5).astype(int))

    # save model
    import joblib
    model_path = os.path.join(outdir, "fusion_head.joblib")
    joblib.dump(pipe, model_path)

    metrics = {"auc": float(auc), "accuracy": float(acc)}
    return model_path, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4_emb", type=str, required=True, help="Path to M4 embeddings (.npy)")
    parser.add_argument("--m5_emb", type=str, required=True, help="Path to M5 embeddings (.npy)")
    parser.add_argument("--m4_manifest", type=str, default=None, help="CSV manifest for M4 with 'id' column (optional)")
    parser.add_argument("--m5_manifest", type=str, default=None, help="CSV manifest for M5 with 'id' column (optional)")
    parser.add_argument("--outdir", type=str, default="workdir/fused", help="Where to save fused embeddings and artifacts")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before concat")
    parser.add_argument("--mode", choices=["concat", "train"], default="concat", help="concat: only save fused embeddings | train: also train logistic head")
    parser.add_argument("--save_csv", action="store_true", help="Save output CSV with id,label, and filepaths")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    emb_m4 = load_npy(args.m4_emb)
    emb_m5 = load_npy(args.m5_emb)
    print("Loaded shapes:", emb_m4.shape, emb_m5.shape)

    df_m4 = load_manifest_csv(args.m4_manifest) if args.m4_manifest else None
    df_m5 = load_manifest_csv(args.m5_manifest) if args.m5_manifest else None

    if df_m4 is not None and df_m5 is not None:
        ids, a_al, b_al, labels = align_by_id(df_m4, df_m5, emb_m4, emb_m5)
    elif df_m4 is not None and df_m5 is None:
        # try to align by id where m4 manifest lists id in same order as emb_m4; and emb_m5 has same order
        ids = df_m4["id"].tolist()
        if emb_m4.shape[0] != emb_m5.shape[0]:
            print("Warning: m4 manifest provided but shapes differ and m5 manifest missing. Attempting index align will fail if counts differ.")
        ids, a_al, b_al, labels = align_by_index(emb_m4, emb_m5)
        labels = df_m4["label"].values if "label" in df_m4.columns else None
    elif df_m4 is None and df_m5 is not None:
        ids = df_m5["id"].tolist()
        ids, a_al, b_al, labels = align_by_index(emb_m4, emb_m5)
        labels = df_m5["label"].values if "label" in df_m5.columns else None
    else:
        ids, a_al, b_al, labels = align_by_index(emb_m4, emb_m5)

    print("Aligned sample count:", len(ids))

    if args.normalize:
        a_al = normalize_emb(a_al)
        b_al = normalize_emb(b_al)

    fused = np.concatenate([a_al, b_al], axis=1)
    fused_path = os.path.join(args.outdir, "fused_embeddings.npy")
    np.save(fused_path, fused)
    print("Saved fused embeddings:", fused_path)

    meta = {"n_samples": fused.shape[0], "dim_m4": a_al.shape[1], "dim_m5": b_al.shape[1], "fused_dim": fused.shape[1]}
    with open(os.path.join(args.outdir, "fused_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Optionally save CSV
    if args.save_csv:
        rows = []
        for i, sid in enumerate(ids):
            row = {"id": sid}
            if labels is not None:
                row["label"] = int(labels[i])
            rows.append(row)
        df_out = pd.DataFrame(rows)
        csv_out = os.path.join(args.outdir, "fused_manifest.csv")
        df_out.to_csv(csv_out, index=False)
        print("Saved fused manifest csv:", csv_out)

    # Optional train a simple head
    if args.mode == "train":
        if labels is None:
            raise ValueError("Training mode requires labels available in manifests or provided.")
        model_path, metrics = train_small_head(fused, labels, outdir=args.outdir)
        with open(os.path.join(args.outdir, "fusion_train_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("Trained fusion head saved to:", model_path)
        print("Train metrics:", metrics)

if __name__ == "__main__":
    main()
