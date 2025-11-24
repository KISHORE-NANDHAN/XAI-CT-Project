#!/usr/bin/env python3
"""
m8_prepare_background.py

Prepare a curated background dataset for KernelSHAP / TreeSHAP.
Saves:
    - background_samples.npy         (np.array: n_background x n_features)
    - background_indices.json        (list of selected sample indices)
    - background_metadata.csv        (rows from fused_manifest corresponding to indices)

Inputs (CLI or config):
    - --embeddings : path to fused_embeddings.npy (required)
    - --manifest   : path to fused_manifest.csv (required)
    - --out_dir    : output directory to save background files (default: outputs/m8/shap)
    - --method     : sampling method: random | stratified | kmeans (default: stratified)
    - --n_samples  : number of background samples (default: 50)
    - --seed       : random seed (default: 42)
    - --label_col  : column in manifest for stratification (default: "label")
    - --kmeans_n_clusters: if method=kmeans, number of clusters (default: n_samples)
    - --min_per_class: minimum samples per class for stratified (default: 1)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LOG = logging.getLogger("m8_prepare_background")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOG.addHandler(handler)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_inputs(emb_path: Path, manifest_path: Path):
    LOG.info("Loading embeddings from %s", emb_path)
    embeddings = np.load(str(emb_path))
    LOG.info("Embeddings shape: %s", embeddings.shape)

    LOG.info("Loading manifest from %s", manifest_path)
    manifest = pd.read_csv(str(manifest_path))
    return embeddings, manifest


def stratified_sample(manifest: pd.DataFrame, n_samples: int, label_col: str, seed: int, min_per_class:int):
    rng = np.random.RandomState(seed)
    if label_col not in manifest.columns:
        raise ValueError(f"label_col '{label_col}' not found in manifest columns: {manifest.columns.tolist()}")
    selected_indices = []
    groups = manifest.groupby(label_col).groups
    n_classes = len(groups)
    # compute per-class allocation (at least min_per_class)
    base = max(min_per_class, n_samples // n_classes)
    # first give base to each class
    alloc = {k: base for k in groups}
    remaining = n_samples - base * n_classes
    # distribute remaining proportionally to class sizes
    sizes = {k: len(groups[k]) for k in groups}
    total = sum(sizes.values())
    for k in groups:
        if remaining <= 0:
            break
        extra = int(round(remaining * (sizes[k] / total)))
        alloc[k] += extra
    # if rounding left some, add to largest classes
    allocated = sum(alloc.values())
    idx_sort = sorted(groups.keys(), key=lambda k: sizes[k], reverse=True)
    i = 0
    while allocated < n_samples:
        alloc[idx_sort[i % len(idx_sort)]] += 1
        allocated += 1
        i += 1

    # sample
    for k, indices in groups.items():
        k_list = list(indices)
        k_n = min(len(k_list), alloc[k])
        if k_n <= 0:
            continue
        selected = rng.choice(k_list, size=k_n, replace=False).tolist()
        selected_indices.extend(selected)
    # final safety: if more than needed, truncate
    if len(selected_indices) > n_samples:
        selected_indices = rng.choice(selected_indices, size=n_samples, replace=False).tolist()
    return list(sorted(selected_indices))


def random_sample(manifest: pd.DataFrame, n_samples: int, seed: int):
    rng = np.random.RandomState(seed)
    n = len(manifest)
    idx = rng.choice(n, size=min(n_samples, n), replace=False).tolist()
    return sorted(idx)


def kmeans_medoids_sample(embeddings: np.ndarray, n_samples: int, seed: int):
    # KMeans + choose nearest-to-centroid samples
    LOG.info("Running KMeans with n_clusters=%d", n_samples)
    km = KMeans(n_clusters=min(n_samples, embeddings.shape[0]), random_state=seed, n_init=10)
    km.fit(embeddings)
    centers = km.cluster_centers_
    # assign each sample to cluster and pick nearest to each center
    labels = km.labels_
    chosen = []
    for cid in range(centers.shape[0]):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            continue
        dists = np.linalg.norm(embeddings[members] - centers[cid], axis=1)
        argmin = np.argmin(dists)
        chosen.append(int(members[argmin]))
    LOG.info("KMeans selected %d medoid-like samples", len(chosen))
    return sorted(chosen)


def save_background(embeddings: np.ndarray, manifest: pd.DataFrame, indices: list, out_dir: Path):
    ensure_dir(out_dir)
    out_samples = embeddings[indices]
    np.save(str(out_dir / "background_samples.npy"), out_samples)
    with open(str(out_dir / "background_indices.json"), "w") as f:
        json.dump(indices, f, indent=2)
    manifest.loc[indices].to_csv(str(out_dir / "background_metadata.csv"), index=False)
    LOG.info("Saved background_samples.npy (%s), background_indices.json, background_metadata.csv in %s", out_samples.shape, out_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a curated SHAP background dataset for M8")
    p.add_argument("--embeddings", type=str, required=True, help="Path to fused_embeddings.npy")
    p.add_argument("--manifest", type=str, required=True, help="Path to fused_manifest.csv")
    p.add_argument("--out_dir", type=str, default="outputs/m8/shap", help="Output directory")
    p.add_argument("--method", type=str, choices=["stratified", "random", "kmeans"], default="stratified")
    p.add_argument("--n_samples", type=int, default=50, help="Number of background samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_col", type=str, default="label", help="Manifest column used for stratified sampling")
    p.add_argument("--kmeans_n_clusters", type=int, default=None, help="If method=kmeans: override cluster count")
    p.add_argument("--min_per_class", type=int, default=1, help="Minimum per-class samples when stratified")
    return p.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    LOG.info("m8_prepare_background started. Project brief: /mnt/data/Problem.docx")
    embeddings, manifest = load_inputs(emb_path, manifest_path)

    # choose method
    if args.method == "stratified":
        LOG.info("Sampling method: stratified (label_col=%s)", args.label_col)
        indices = stratified_sample(manifest, args.n_samples, args.label_col, args.seed, args.min_per_class)
    elif args.method == "random":
        LOG.info("Sampling method: random")
        indices = random_sample(manifest, args.n_samples, args.seed)
    elif args.method == "kmeans":
        LOG.info("Sampling method: kmeans")
        n_clusters = args.kmeans_n_clusters if args.kmeans_n_clusters else args.n_samples
        indices = kmeans_medoids_sample(embeddings, n_clusters, args.seed)
        # if kmeans produced fewer than requested, pad with random
        if len(indices) < args.n_samples:
            extra = sorted(set(range(embeddings.shape[0])) - set(indices))
            take = min(len(extra), args.n_samples - len(indices))
            if take > 0:
                indices.extend(extra[:take])
    else:
        raise ValueError("Unknown method")

    # safety
    indices = sorted(list(set(indices)))[: args.n_samples]

    save_background(embeddings, manifest, indices, out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
