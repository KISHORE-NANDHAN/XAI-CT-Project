#!/usr/bin/env python3
"""
insertion_deletion.py

Compute insertion & deletion curves for explanation validity:
- Insertion: gradually ADD salient voxels (sorted by CAM score)
- Deletion: gradually REMOVE salient voxels
Compute AUC for both curves.

Output:
  outputs/m9/insertion_curves/<study>.json
  outputs/m9/deletion_curves/<study>.json
  outputs/m9/insertion_deletion_auc.json
  outputs/m9/visuals/<study>_insertion_deletion.png

Usage:
  python insertion_deletion.py --config config/m9.yaml
"""

import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import zoom
from skimage.transform import resize
from glob import glob

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_volume(path):
    return np.load(path)   # (Z, H, W)

def load_cam(cam_dir, study_id):
    candidates = [
        f"{cam_dir}/{study_id}.npy",
        f"{cam_dir}/{study_id}_cam.npy",
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.load(c)

    # fallback: image stack
    imgs = sorted(glob(f"{cam_dir}/{study_id}*"))
    if not imgs:
        raise FileNotFoundError(f"No CAM for {study_id}")
    return np.stack([plt.imread(i) for i in imgs])

def normalize_cam(cam):
    cmin, cmax = cam.min(), cam.max()
    if cmax > cmin:
        return (cam - cmin) / (cmax - cmin)
    return np.zeros_like(cam)

def resize_to(vol, target):
    """Resize CAM mask or prob map to match volume shape"""
    if vol.shape == target.shape:
        return vol
    if vol.ndim == 3:
        zf = target.shape[0] / vol.shape[0]
        yf = target.shape[1] / vol.shape[1]
        xf = target.shape[2] / vol.shape[2]
        return zoom(vol, (zf, yf, xf), order=0)
    return vol

# --------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------

class ModelWrapper:
    def __init__(self, path, device="cpu"):
        self.device = device
        try:
            self.model = torch.jit.load(path, map_location=device)
            logging.info("Loaded TorchScript model")
        except:
            logging.info("TorchScript load failed, trying torch.load")
            self.model = torch.load(path, map_location=device)
        self.model.eval()

    def predict_prob(self, vol):
        """Return prob of positive class"""
        x = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (1,1,Z,H,W)
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
        out = out.cpu().numpy()

        # interpret logits
        if out.ndim == 2 and out.shape[1] == 2:
            return float(1/(1+np.exp(-out[0,1])))
        if out.ndim == 2 and out.shape[1] == 1:
            return float(1/(1+np.exp(-out[0,0])))
        if out.ndim == 1:
            return float(1/(1+np.exp(-out[0])))

        # fallback softmax
        p = np.exp(out[0]) / np.sum(np.exp(out[0]))
        return float(p[-1])

# --------------------------------------------------------------------
# Insertion / Deletion logic
# --------------------------------------------------------------------

def run_insertion(volume, cam, model, steps=50):
    Z,H,W = volume.shape
    flat_cam = cam.flatten()
    idx = np.argsort(flat_cam)[::-1]  # descending

    insertion_curve = []
    base = np.full_like(volume, np.median(volume))   # neutral baseline
    current = base.copy()

    total = len(idx)
    step_size = total // steps

    for s in range(steps):
        end = (s+1)*step_size
        vox = idx[:end]
        current_flat = current.flatten()
        vol_flat = volume.flatten()
        current_flat[vox] = vol_flat[vox]
        current = current_flat.reshape(volume.shape)

        prob = model.predict_prob(current)
        insertion_curve.append(float(prob))

    return insertion_curve


def run_deletion(volume, cam, model, steps=50):
    Z,H,W = volume.shape
    flat_cam = cam.flatten()
    idx = np.argsort(flat_cam)[::-1]  # descending

    deletion_curve = []
    current = volume.copy()
    total = len(idx)
    step_size = total // steps

    for s in range(steps):
        end = (s+1)*step_size
        vox = idx[:end]
        current_flat = current.flatten()
        current_flat[vox] = np.median(volume)
        current = current_flat.reshape(volume.shape)

        prob = model.predict_prob(current)
        deletion_curve.append(float(prob))

    return deletion_curve


def auc_curve(values):
    """Simple trapezoidal AUC"""
    xs = np.linspace(0,1,len(values))
    return float(np.trapz(values, xs))


# --------------------------------------------------------------------
# Master per-study run
# --------------------------------------------------------------------

def run_one(study_id, cfg, model):
    vol_path = f"{cfg['volumes_dir']}/{study_id}.npy"
    volume = load_volume(vol_path)

    cam = load_cam(cfg["cam_dir"], study_id)
    cam = normalize_cam(cam)

    if cam.shape != volume.shape:
        cam = resize_to(cam, volume)

    steps = cfg.get("steps", 50)

    insertion = run_insertion(volume, cam, model, steps)
    deletion = run_deletion(volume, cam, model, steps)

    auc_ins = auc_curve(insertion)
    auc_del = auc_curve(deletion)

    return {
        "study_id": study_id,
        "insertion_curve": insertion,
        "deletion_curve": deletion,
        "auc_insertion": auc_ins,
        "auc_deletion": auc_del
    }, insertion, deletion


def plot_curves(study_id, insertion, deletion, out_dir):
    plt.figure(figsize=(6,4))
    x = np.linspace(0,1,len(insertion))

    plt.plot(x, insertion, label="Insertion")
    plt.plot(x, deletion, label="Deletion")
    plt.xlabel("Fraction of Salient Voxels")
    plt.ylabel("Predicted Probability")
    plt.title(study_id)
    plt.legend()

    os.makedirs(f"{out_dir}/visuals", exist_ok=True)
    plt.savefig(f"{out_dir}/visuals/{study_id}_insertion_deletion.png", dpi=150)
    plt.close()

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m9.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    out_dir = cfg.get("out_dir", "outputs/m9")
    os.makedirs(out_dir, exist_ok=True)

    model = ModelWrapper(cfg["model_path"], device=cfg.get("device","cpu"))

    studies = sorted([Path(p).stem for p in glob(f"{cfg['volumes_dir']}/*.npy")])

    all_results = []

    for s in tqdm(studies, desc="Insertion/Deletion"):
        try:
            res, ins, delc = run_one(s, cfg, model)

            # save curves
            os.makedirs(f"{out_dir}/insertion_curves", exist_ok=True)
            os.makedirs(f"{out_dir}/deletion_curves", exist_ok=True)

            with open(f"{out_dir}/insertion_curves/{s}.json", "w") as f:
                json.dump({"insertion": ins}, f, indent=2)

            with open(f"{out_dir}/deletion_curves/{s}.json", "w") as f:
                json.dump({"deletion": delc}, f, indent=2)

            plot_curves(s, ins, delc, out_dir)

            all_results.append(res)
        except Exception as e:
            logging.exception(f"[ERROR] {s}: {e}")

    # aggregate AUCs
    auc_ins = [r["auc_insertion"] for r in all_results]
    auc_del = [r["auc_deletion"] for r in all_results]

    summary = {
        "mean_auc_insertion": float(np.mean(auc_ins)),
        "mean_auc_deletion": float(np.mean(auc_del)),
        "studies": all_results
    }

    with open(f"{out_dir}/insertion_deletion_auc.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("DONE.")


if __name__ == "__main__":
    main()
