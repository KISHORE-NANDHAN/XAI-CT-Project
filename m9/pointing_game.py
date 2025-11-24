#!/usr/bin/env python3
"""
pointing_game.py

Compute:
  - Pointing Game Score (CAM max inside lesion mask?)
  - IoU at fixed thresholds

Requires:
  lesion masks at: masks_dir/<study>.npy or <study>_mask.npy

Output:
  outputs/m9/pointing_game_scores.json
  outputs/m9/iou_scores.json
  outputs/m9/visuals/<study>_pg_iou.png
"""

import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm
from glob import glob
from pathlib import Path

# --------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_config(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def load_volume(path):
    return np.load(path)

def load_mask(mask_dir, study):
    candidates = [
        f"{mask_dir}/{study}.npy",
        f"{mask_dir}/{study}_mask.npy",
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.load(c)
    raise FileNotFoundError(f"No lesion mask for {study}")

def load_cam(cam_dir, study):
    cand = f"{cam_dir}/{study}.npy"
    if os.path.exists(cand):
        return np.load(cand)

    # fallback: 2D images
    imgs = sorted(glob(f"{cam_dir}/{study}*"))
    if not imgs:
        raise FileNotFoundError(f"No CAM for {study}")
    return np.stack([plt.imread(i) for i in imgs])

def norm_cam(cam):
    mn, mx = cam.min(), cam.max()
    if mx > mn:
        return (cam - mn)/(mx-mn)
    return np.zeros_like(cam)

def resize_to(arr, target):
    if arr.shape == target.shape:
        return arr
    if arr.ndim == 3:
        return zoom(arr, (
            target.shape[0]/arr.shape[0],
            target.shape[1]/arr.shape[1],
            target.shape[2]/arr.shape[2]
        ), order=0)
    return arr

# --------------------------------------------------------------------
# Pointing game + IoU
# --------------------------------------------------------------------

def pointing_game(cam, mask):
    """True if CAM max location is inside mask"""
    idx = np.argmax(cam)
    z = idx // (cam.shape[1]*cam.shape[2])
    rem = idx % (cam.shape[1]*cam.shape[2])
    y = rem // cam.shape[2]
    x = rem % cam.shape[2]
    return bool(mask[z,y,x])

def compute_iou(cam, mask, thresholds=(0.2,0.3,0.5)):
    """Returns IoU at each threshold"""
    ious = {}
    for t in thresholds:
        binary = cam >= t
        inter = np.logical_and(binary, mask).sum()
        union = np.logical_or(binary, mask).sum()
        ious[t] = float(inter/union) if union>0 else 0.0
    return ious

def plot_example(study, cam, mask, out_dir):
    zmid = cam.shape[0]//2
    c = cam[zmid]
    m = mask[zmid]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("CAM (slice mid)")
    plt.imshow(c, cmap="jet")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Mask overlay")
    plt.imshow(c, cmap="jet", alpha=0.5)
    plt.imshow(m, cmap="gray", alpha=0.3)
    plt.axis("off")

    os.makedirs(f"{out_dir}/visuals", exist_ok=True)
    plt.savefig(f"{out_dir}/visuals/{study}_pg_iou.png", dpi=150)
    plt.close()

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    out_dir = cfg.get("out_dir", "outputs/m9")
    os.makedirs(out_dir, exist_ok=True)

    studies = [Path(p).stem for p in glob(f"{cfg['volumes_dir']}/*.npy")]

    pg_results = {}
    iou_results = {}

    for s in tqdm(studies, desc="Pointing Game"):
        try:
            vol = load_volume(f"{cfg['volumes_dir']}/{s}.npy")
            cam = load_cam(cfg["cam_dir"], s)
            mask = load_mask(cfg["masks_dir"], s)

            cam = norm_cam(cam)

            if cam.shape != mask.shape:
                cam = resize_to(cam, mask)

            # pointing game
            hit = pointing_game(cam, mask)
            pg_results[s] = int(hit)

            # IoU
            ious = compute_iou(cam, mask)
            iou_results[s] = ious

            plot_example(s, cam, mask, out_dir)

        except Exception as e:
            logging.warning(f"[{s}] failed: {e}")

    # save
    with open(f"{out_dir}/pointing_game_scores.json", "w") as f:
        json.dump(pg_results, f, indent=2)

    with open(f"{out_dir}/iou_scores.json", "w") as f:
        json.dump(iou_results, f, indent=2)

    # summary
    hits = sum(pg_results.values())
    total = len(pg_results)

    logging.info(
        f"Pointing Game: {hits}/{total} = {hits/total:.3f}"
    )

if __name__ == "__main__":
    main()
