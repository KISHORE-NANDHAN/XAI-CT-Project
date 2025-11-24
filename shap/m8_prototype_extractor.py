#!/usr/bin/env python3
"""
m8_prototype_extractor.py

Extract prototype embedding vectors for M8 (ProtoPNet-style nearest neighbours).

This script:
    - Loads fused embeddings + manifest
    - Splits embeddings per class
    - For each class, performs clustering (kmeans or medoids) to find prototype centers
    - Stores:
        - prototype_vectors.npy
        - prototype_metadata.json

Prototype vectors are later used by m8_prototype_tiles.py to find nearest visual
examples ("this looks like...") for each prototype.

Inputs:
    --embeddings fused_embeddings.npy
    --manifest   fused_manifest.csv
    --out_dir    outputs/m8/prototypes/
    --per_class  number of prototypes per class (default: 5)
    --method     clustering: kmeans | medoids  (default: kmeans)
    --seed       random seed

Output:
    prototype_vectors.npy            (N_prototypes x 518-d)
    prototype_metadata.json          (prototype_id → class, cluster_center, etc.)

Project brief ref: /mnt/data/Problem.docx
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LOG = logging.getLogger("m8_prototype_extractor")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOG.addHandler(handler)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def kmeans_centroids(X, n_clusters, seed):
    LOG.info("Running KMeans for class with %d samples, clusters=%d", len(X), n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return km.cluster_centers_


def medoid_centroids(X, n_clusters, seed):
    """Classical medoid selection using kmeans to define cluster → pick nearest point."""
    LOG.info("Running medoid extraction via KMeans+nearest-sample")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    centers = km.cluster_centers_
    labels = km.labels_

    medoids = []
    for cid in range(n_clusters):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(X[members] - centers[cid], axis=1)
        medoids.append(X[members[np.argmin(dists)]])
    medoids = np.array(medoids)
    return medoids


def parse_args():
    p = argparse.ArgumentParser(description="Extract per-class prototype embeddings for M8")
    p.add_argument("--embeddings", required=True, help="Path to fused_embeddings.npy")
    p.add_argument("--manifest", required=True, help="Path to fused_manifest.csv")
    p.add_argument("--out_dir", default="outputs/m8/prototypes", help="Output directory")
    p.add_argument("--per_class", type=int, default=5, help="Prototypes per class")
    p.add_argument("--method", choices=["kmeans", "medoids"], default="kmeans")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    LOG.info("=== M8 Prototype Extractor Started ===")
    LOG.info("Reading embeddings & manifest")

    X = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest)

    if "label" not in manifest.columns:
        raise ValueError("Manifest missing 'label' column required for class grouping.")

    classes = sorted(manifest["label"].unique().tolist())
    LOG.info("Found classes: %s", classes)

    prototype_vectors = []
    prototype_meta = []

    for cls in classes:
        LOG.info("Processing class: %s", cls)
        idx = manifest.index[manifest["label"] == cls].tolist()
        Xc = X[idx]

        if len(Xc) == 0:
            LOG.warning("Class %s has no samples. Skipping.", cls)
            continue

        k = min(args.per_class, len(Xc))

        if args.method == "kmeans":
            centers = kmeans_centroids(Xc, k, args.seed)
        else:
            centers = medoid_centroids(Xc, k, args.seed)

        for i, proto in enumerate(centers):
            prototype_vectors.append(proto)
            prototype_meta.append({
                "prototype_id": len(prototype_vectors) - 1,
                "class": str(cls),
                "cluster_index": i,
                "method": args.method,
                "source_samples": idx  # optional; can be removed if too large
            })

    prototype_vectors = np.array(prototype_vectors)
    LOG.info("Generated %d prototypes total.", len(prototype_vectors))

    np.save(out_dir / "prototype_vectors.npy", prototype_vectors)

    with open(out_dir / "prototype_metadata.json", "w") as f:
        json.dump(prototype_meta, f, indent=2)

    LOG.info("Saved prototypes at %s", out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
