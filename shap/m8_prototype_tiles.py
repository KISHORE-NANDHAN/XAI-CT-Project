#!/usr/bin/env python3
"""
m8_prototype_tiles.py (PATCHED)

Automatically handles enhanced manifest from build_enhanced_manifest.py:
 - cam_pngs       (list)
 - cam_npys       (list)
 - slice_indices  (list)

Adds:
 - slice_path  (auto-generated: first cam_png)
 - cam_path    (auto-generated: first cam_png)
"""

import argparse
import json
import logging
import sys
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LOG = logging.getLogger("m8_prototype_tiles")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOG.addHandler(handler)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load slice image: {path}")
    return img


def load_cam(path):
    cam = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if cam is None:
        raise FileNotFoundError(f"Could not load CAM heatmap: {path}")
    return cam


def overlay_cam(img, cam, alpha=0.5):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cam = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX)

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(gray, 1 - alpha, heat, alpha, 0)


def parse_args():
    p = argparse.ArgumentParser(description="Generate prototype visual tiles")
    p.add_argument("--prototypes", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--manifest", required=True)

    p.add_argument("--img_root", required=True)
    p.add_argument("--cam_root", required=True)

    p.add_argument("--k_nearest", type=int, default=3)
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--out_dir", default="outputs/m8/prototypes/prototype_tiles")
    return p.parse_args()


def auto_fix_manifest(df):
    """Add slice_path and cam_path automatically from cam_pngs list."""
    if ("slice_path" in df.columns) and ("cam_path" in df.columns):
        LOG.info("Manifest already contains slice_path and cam_path")
        return df

    LOG.warning("Manifest missing slice_path/cam_path – auto-generating from cam_pngs")

    slice_paths = []
    cam_paths = []

    for entry in df["cam_pngs"]:
        try:
            lst = ast.literal_eval(entry) if isinstance(entry, str) else entry
            if isinstance(lst, list) and len(lst) > 0:
                slice_paths.append(lst[0])
                cam_paths.append(lst[0])
            else:
                slice_paths.append(None)
                cam_paths.append(None)
        except Exception:
            slice_paths.append(None)
            cam_paths.append(None)

    df["slice_path"] = slice_paths
    df["cam_path"] = cam_paths

    LOG.info("Auto-filled slice_path and cam_path.")

    return df


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    LOG.info("=== M8 Prototype Tile Generator Started ===")

    proto_vecs = np.load(args.prototypes)
    with open(args.metadata) as f:
        proto_meta = json.load(f)

    X = np.load(args.embeddings)
    df = pd.read_csv(args.manifest)

    # 🟩 FIX: auto-create slice_path & cam_path
    df = auto_fix_manifest(df)

    LOG.info(f"Total prototypes: {len(proto_vecs)}")
    LOG.info(f"Embeddings shape: {X.shape}")

    for pidx, proto in enumerate(proto_vecs):
        LOG.info(f"Processing prototype {pidx}...")

        d = np.linalg.norm(X - proto, axis=1)
        nearest = np.argsort(d)[: args.k_nearest]

        for rank, idx in enumerate(nearest):
            row = df.iloc[idx]

            slice_path = row["slice_path"]
            cam_path = row["cam_path"]

            if slice_path is None or cam_path is None:
                LOG.warning(f"Missing paths for row {idx}; skipping.")
                continue

            full_slice = Path(args.img_root) / slice_path
            full_cam = Path(args.cam_root) / cam_path

            try:
                img = load_image(full_slice)
                cam = load_cam(full_cam)
            except Exception as e:
                LOG.warning(f"Skipping row {idx}: {e}")
                continue

            merged = overlay_cam(img, cam, 0.40)

            sz = args.tile_size
            tile = np.concatenate([
                cv2.cvtColor(cv2.resize(img, (sz, sz)), cv2.COLOR_GRAY2RGB),
                cv2.resize(merged, (sz, sz))
            ], axis=1)

            out_path = out_dir / f"proto_{pidx}_neighbor_{rank}.png"
            cv2.imwrite(str(out_path), tile)
            LOG.info(f"Saved tile: {out_path}")

    LOG.info("All prototype tiles saved. Done.")


if __name__ == "__main__":
    main()
