#!/usr/bin/env python3
"""
sufficiency.py

Keep ONLY salient regions (from CAM) and measure retained model probability.
Saves per-study results to outputs/m9/sufficiency_scores.json and visuals to outputs/m9/visuals/.

Usage:
    python scripts/m9/sufficiency.py --config config/m9.yaml
    python scripts/m9/sufficiency.py --model models/fusion_model.pt \
        --cam_dir outputs/m7/gradcam_2d --volumes_dir data/preprocessed --out_dir outputs/m9 --topk 0.1

Notes:
- Very similar expectations as comprehensiveness script.
"""

import os
import sys
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import torch
import csv
from scipy.ndimage import zoom, binary_erosion
from skimage.transform import resize

# ---------------------------
# Helpers (a bit pared-down)
# ---------------------------

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_studies_from_dir(volumes_dir, ext=".npy"):
    vols = sorted(glob(os.path.join(volumes_dir, f"*{ext}")))
    return [Path(v).stem for v in vols]

def load_volume(volume_path):
    return np.load(volume_path)  # shape (Z,H,W)

def load_cam_for_study(cam_dir, study_id):
    candidates = [
        os.path.join(cam_dir, f"{study_id}.npy"),
        os.path.join(cam_dir, f"{study_id}_cam.npy"),
        os.path.join(cam_dir, f"{study_id}.npz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.load(c)
    pattern = os.path.join(cam_dir, f"{study_id}*")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CAM found for {study_id} in {cam_dir}")
    cams = []
    for f in files:
        try:
            img = Image.open(f).convert("L")
            cams.append(np.array(img, dtype=np.float32) / 255.0)
        except Exception:
            continue
    if not cams:
        raise FileNotFoundError(f"No readable CAM images for {study_id} in {cam_dir}")
    return np.stack(cams, axis=0)

def normalize_cam(cam):
    cam = cam.astype(np.float32)
    if cam.ndim == 2:
        mn, mx = cam.min(), cam.max()
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)
        return cam
    else:
        cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            mn, mx = c.min(), c.max()
            if mx > mn:
                c = (c - mn) / (mx - mn)
            else:
                c = np.zeros_like(c)
            cams.append(c)
        return np.stack(cams, axis=0)

def compute_mask_from_cam(cam, mode='topk', topk=0.1, threshold=None, erode=0):
    cam = normalize_cam(cam)
    if cam.ndim == 2:
        flat = cam.flatten()
        if mode == 'topk':
            k = int(max(1, np.floor(flat.size * topk)))
            if k >= flat.size:
                thr = flat.min() - 1e-6
            else:
                thr = np.partition(flat, -k)[-k]
            mask = cam >= thr
        else:
            thr = threshold if threshold is not None else 0.5
            mask = cam >= thr
        if erode > 0:
            mask = binary_erosion(mask, iterations=erode)
        return mask.astype(np.bool_)
    else:
        masks = []
        for i in range(cam.shape[0]):
            masks.append(compute_mask_from_cam(cam[i], mode=mode, topk=topk, threshold=threshold, erode=erode))
        return np.stack(masks, axis=0)

def apply_mask_keep(volume, mask):
    """Keep only salient regions; replace others with median of non-salient region (or zero if none)."""
    vol = volume.copy()
    if mask.shape != vol.shape:
        # attempt to resize
        logging.info("Resizing mask to match volume shape: %s -> %s", mask.shape, vol.shape)
        if mask.ndim == 3:
            zf = vol.shape[0] / mask.shape[0]
            yf = vol.shape[1] / mask.shape[1]
            xf = vol.shape[2] / mask.shape[2]
            mask_resized = zoom(mask.astype(np.float32), (zf, yf, xf), order=0) >= 0.5
        else:
            mask_resized = resize(mask.astype(np.float32), vol.shape[1:], order=0, preserve_range=True) >= 0.5
            mask_resized = np.expand_dims(mask_resized, 0)
        mask = mask_resized
    keep = mask
    # for non-keep region, replace with median of kept region to avoid unnatural zeroing
    if keep.sum() == 0:
        fill = 0.0
    else:
        fill = float(np.median(vol[keep]))
    out = np.full_like(vol, fill)
    out[keep] = vol[keep]
    return out

# Reuse the ModelWrapper from comprehensiveness logic but simplified
class ModelWrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path):
        try:
            m = torch.jit.load(path, map_location=self.device)
            logging.info("Loaded TorchScript model from %s", path)
            return m
        except Exception:
            logging.info("Not a TorchScript model, attempting torch.load(...)")
        try:
            state = torch.load(path, map_location=self.device)
            if hasattr(state, 'eval'):
                logging.info("Loaded nn.Module via torch.load")
                return state
            else:
                raise RuntimeError("Model file requires instantiation from architecture code; use TorchScript for portability.")
        except Exception as e:
            logging.error("Failed to load model: %s", str(e))
            raise

    def predict_prob(self, volume_np):
        if not isinstance(volume_np, np.ndarray):
            volume_np = np.array(volume_np)
        x = torch.from_numpy(volume_np.astype(np.float32)).unsqueeze(0)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            out = out.detach().cpu().numpy()
            if out.ndim == 2 and out.shape[1] == 2:
                prob = float(1.0 / (1.0 + np.exp(-out[0,1])))
            elif out.ndim == 2 and out.shape[1] == 1:
                prob = float(1.0 / (1.0 + np.exp(-out[0,0])))
            elif out.ndim == 1:
                prob = float(1.0 / (1.0 + np.exp(-out[0])))
            else:
                try:
                    p = np.exp(out[0]) / np.sum(np.exp(out[0]))
                    prob = float(p[-1])
                except Exception:
                    raise RuntimeError("Could not interpret model output shape: %s" % (out.shape,))
            return prob

def run_one_study(study_id, cfg, model_wrapper):
    volume_path = os.path.join(cfg['volumes_dir'], f"{study_id}.npy")
    if not os.path.exists(volume_path):
        raise FileNotFoundError(volume_path)
    vol = load_volume(volume_path)
    cam = load_cam_for_study(cfg['cam_dir'], study_id)
    cam = normalize_cam(cam)
    mask = compute_mask_from_cam(cam, mode=cfg.get('mask_mode','topk'),
                                 topk=float(cfg.get('topk', 0.1)),
                                 threshold=cfg.get('threshold', None),
                                 erode=cfg.get('erode', 0))
    orig_prob = model_wrapper.predict_prob(vol)
    kept_vol = apply_mask_keep(vol, mask)
    kept_prob = model_wrapper.predict_prob(kept_vol)
    prob_retained = kept_prob / (orig_prob + 1e-12) if orig_prob != 0 else float(kept_prob)
    result = {
        'study_id': study_id,
        'orig_prob': float(orig_prob),
        'kept_prob': float(kept_prob),
        'prob_retained': float(prob_retained)
    }
    return result, vol, cam, mask, kept_vol

def save_visuals(study_id, out_dir, vol, cam, mask, kept_vol):
    vis_dir = os.path.join(out_dir, 'visuals')
    os.makedirs(vis_dir, exist_ok=True)
    z = vol.shape[0]
    center = z // 2
    offsets = list(range(-4,5))
    sel = [min(max(0, center + o), z-1) for o in offsets]
    rows = 3; cols = 6
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()
    idx = 0
    for s in sel:
        img = vol[s]
        cam_s = cam[s] if cam.ndim==3 and cam.shape[0]==vol.shape[0] else resize(cam if cam.ndim==2 else cam[0], img.shape)
        mask_s = mask[s] if mask.ndim==3 and mask.shape[0]==vol.shape[0] else resize(mask if mask.ndim==2 else mask[0], img.shape) >= 0.5
        axs[idx].imshow(img, cmap='gray')
        axs[idx].imshow(cam_s, cmap='jet', alpha=0.4, vmin=0, vmax=1)
        axs[idx].axis('off')
        idx += 1
        axs[idx].imshow(kept_vol[s], cmap='gray')
        axs[idx].axis('off')
        idx += 1
    plt.suptitle(f"{study_id} — left: orig+CAM, right: kept-only")
    p = os.path.join(vis_dir, f"{study_id}_sufficiency.png")
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close(fig)

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/m9.yaml', help='config yaml')
    parser.add_argument('--model', type=str, help='override model path')
    parser.add_argument('--cam_dir', type=str, help='override cam dir')
    parser.add_argument('--volumes_dir', type=str, help='override preprocessed volumes dir')
    parser.add_argument('--study_list', type=str, help='optional text file with study ids, one per line')
    parser.add_argument('--out_dir', type=str, default='outputs/m9', help='output directory')
    parser.add_argument('--topk', type=float, help='override topk fraction')
    parser.add_argument('--threshold', type=float, help='override threshold')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)
    if args.model:
        cfg['model_path'] = args.model
    if args.cam_dir:
        cfg['cam_dir'] = args.cam_dir
    if args.volumes_dir:
        cfg['volumes_dir'] = args.volumes_dir
    cfg['out_dir'] = args.out_dir
    if args.topk is not None:
        cfg['topk'] = args.topk
    if args.threshold is not None:
        cfg['threshold'] = args.threshold
    cfg['device'] = args.device

    os.makedirs(cfg['out_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['out_dir'], 'visuals'), exist_ok=True)

    setup_logging()
    logging.info("Starting sufficiency run with config: %s", cfg)

    if args.study_list:
        with open(args.study_list, 'r') as f:
            studies = [l.strip() for l in f if l.strip()]
    else:
        studies = find_studies_from_dir(cfg['volumes_dir'])

    model = ModelWrapper(cfg['model_path'], device=cfg['device'])

    results = []
    csv_path = os.path.join(cfg['out_dir'], 'sufficiency_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['study_id','orig_prob','kept_prob','prob_retained'])
        writer.writeheader()
        for s in tqdm(studies, desc="Studies"):
            try:
                res, vol, cam, mask, kept_vol = run_one_study(s, cfg, model)
                writer.writerow(res)
                results.append(res)
                with open(os.path.join(cfg['out_dir'], f"{s}_sufficiency.json"), 'w') as jf:
                    json.dump(res, jf, indent=2)
                try:
                    save_visuals(s, cfg['out_dir'], vol, cam, mask, kept_vol)
                except Exception as e:
                    logging.warning("Failed to save visual for %s: %s", s, str(e))
            except Exception as e:
                logging.exception("Failed for study %s: %s", s, str(e))
                continue

    retained = np.array([r['prob_retained'] for r in results if 'prob_retained' in r])
    summary = {
        'n_studies': len(results),
        'mean_retained': float(np.mean(retained)) if retained.size else None,
        'median_retained': float(np.median(retained)) if retained.size else None,
        'std_retained': float(np.std(retained)) if retained.size else None,
    }
    with open(os.path.join(cfg['out_dir'], 'sufficiency_scores.json'), 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)

    logging.info("Done. Summary: %s", summary)

if __name__ == "__main__":
    main_cli()
