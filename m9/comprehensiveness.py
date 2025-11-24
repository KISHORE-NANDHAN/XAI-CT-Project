#!/usr/bin/env python3
"""
comprehensiveness.py

Mask OUT the top salient regions (from CAM) and measure the drop in model probability.
Saves per-study results to outputs/m9/comprehensiveness_scores.json and visuals to outputs/m9/visuals/.

Usage:
    python scripts/m9/comprehensiveness.py --config config/m9.yaml
    python scripts/m9/comprehensiveness.py --model models/fusion_model.pt \
        --cam_dir outputs/m7/gradcam_2d --volumes_dir data/preprocessed --out_dir outputs/m9 --topk 0.1

Notes / expectations:
- Model loader supports:
  * Torchscript (.pt with torch.jit.load)
  * Standard PyTorch state_dict if you provide a model factory callable (not included here)
  * Or a simple wrapper that loads a torch.nn.Module saved with torch.save(model.state_dict())
- CAMs can be .npy (recommended) or image files (.png/.jpg). If .npy, expected shape (H,W) per slice or (Z,H,W) for 3D.
- Volumes are expected as numpy .npy arrays (Z,H,W) and pre-windowed to the model expectation.
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
import torchvision.transforms as T
import csv
from scipy.ndimage import zoom, binary_dilation
from skimage.transform import resize

# ---------------------------
# Helpers
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
    # Expect one file per study named <study_id>.npy
    vols = sorted(glob(os.path.join(volumes_dir, f"*{ext}")))
    studies = [Path(v).stem for v in vols]
    return studies

def load_volume(volume_path):
    return np.load(volume_path)  # shape (Z,H,W)

def load_cam_for_study(cam_dir, study_id):
    """
    Look for <study_id>.npy, else <study_id>_cam.npy, else directory with slice cams.
    Returns:
      cam: np.array either (Z,H,W) or (H,W) for single-slice
    """
    candidates = [
        os.path.join(cam_dir, f"{study_id}.npy"),
        os.path.join(cam_dir, f"{study_id}_cam.npy"),
        os.path.join(cam_dir, f"{study_id}.npz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.load(c)
    # fallback: look for slice files like studyid_slice_000.png
    pattern = os.path.join(cam_dir, f"{study_id}*")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CAM found for {study_id} in {cam_dir}")
    # if we have many image files, load and stack
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
    # cam: (Z,H,W) or (H,W)
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

def compute_mask_from_cam(cam, mode='topk', topk=0.1, threshold=None, dilate=3):
    """
    Return binary mask (same shape as cam) where True indicates salient pixels.
    mode: 'topk' -> keep top fraction topk (0..1)
          'thresh' -> threshold absolute value (0..1)
    dilate: morphological dilation radius to expand salient region
    """
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
        if dilate > 0:
            mask = binary_dilation(mask, iterations=dilate)
        return mask.astype(np.bool_)
    else:
        masks = []
        for i in range(cam.shape[0]):
            masks.append(compute_mask_from_cam(cam[i], mode=mode, topk=topk, threshold=threshold, dilate=dilate))
        return np.stack(masks, axis=0)

def apply_mask_out(volume, mask):
    """Mask OUT salient regions -> set salient voxels to zero (or to mean background)"""
    vol = volume.copy()
    if mask.shape != vol.shape:
        # attempt to resize mask to vol
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
    # replace salient voxels by median of non-salient voxels to avoid distribution shift to zeros
    non_salient = ~mask
    if non_salient.sum() == 0:
        fill = 0.0
    else:
        fill = float(np.median(vol[non_salient]))
    out = vol.copy()
    out[mask] = fill
    return out

# ---------------------------
# Model wrapper (best-effort)
# ---------------------------
class ModelWrapper:
    def __init__(self, model_path, device='cpu', input_mode='volume'):
        """
        input_mode: 'volume' expects (Z,H,W) numpy and model returns prob for positive class
        """
        self.device = device
        self.model_path = model_path
        self.input_mode = input_mode
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path):
        # try torchscript first
        try:
            m = torch.jit.load(path, map_location=self.device)
            logging.info("Loaded TorchScript model from %s", path)
            return m
        except Exception:
            logging.info("Not a TorchScript model, attempting torch.load(...)")
        # try normal torch.load
        try:
            state = torch.load(path, map_location=self.device)
            # If it's a dict with 'state_dict' or key 'model', user must provide a factory - we cannot instantiate arbitrary architectures.
            if isinstance(state, dict) and ('state_dict' in state or 'model' in state):
                raise RuntimeError("Model file appears to be a state_dict. For full automation you must provide a script that instantiates the module architecture and loads the state_dict. "
                                   "Save a TorchScript model instead (torch.jit.trace / torch.jit.script) or provide a model factory.")
            # If it's an nn.Module saved with torch.save(model), it will load as module here.
            if hasattr(state, 'eval'):
                logging.info("Loaded nn.Module via torch.load")
                return state
            else:
                raise RuntimeError("Unrecognized torch object in file.")
        except Exception as e:
            logging.error("Failed to load model: %s", str(e))
            raise

    def predict_prob(self, volume_np):
        """
        volume_np: numpy array (Z,H,W) or (C,Z,H,W) depending on model expectation.
        Returns probability scalar for positive class.
        """
        # convert to torch tensor with batch dim
        if not isinstance(volume_np, np.ndarray):
            volume_np = np.array(volume_np)
        x = torch.from_numpy(volume_np.astype(np.float32)).unsqueeze(0)  # shape (1, ...)
        # If model expects channel-first and single channel, expand
        if x.ndim == 4:  # (1,Z,H,W)
            # assume model expects (B,1,Z,H,W) or (B,C,H,W)
            # Try to add channel dim
            x = x.unsqueeze(1)  # (1,1,Z,H,W)
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            # out can be logits tensor shape (1,2) or (1,) etc.
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            out = out.detach().cpu().numpy()
            # Interpret outputs:
            if out.ndim == 2 and out.shape[1] == 2:
                # assume logit for two classes
                prob = float(1.0 / (1.0 + np.exp(-out[0,1])))
            elif out.ndim == 2 and out.shape[1] == 1:
                prob = float(1.0 / (1.0 + np.exp(-out[0,0])))
            elif out.ndim == 1:
                # if single value
                prob = float(1.0 / (1.0 + np.exp(-out[0])))
            else:
                # fallback: softmax first row
                try:
                    p = np.exp(out[0]) / np.sum(np.exp(out[0]))
                    prob = float(p[-1])
                except Exception:
                    raise RuntimeError("Could not interpret model output shape: %s" % (out.shape,))
            return prob

# ---------------------------
# Main logic
# ---------------------------

def run_one_study(study_id, cfg, model_wrapper):
    volume_path = os.path.join(cfg['volumes_dir'], f"{study_id}.npy")
    if not os.path.exists(volume_path):
        raise FileNotFoundError(volume_path)
    vol = load_volume(volume_path)  # (Z,H,W)
    cam = load_cam_for_study(cfg['cam_dir'], study_id)  # (Z,H,W) or (H,W)
    cam = normalize_cam(cam)
    mask = compute_mask_from_cam(cam, mode=cfg.get('mask_mode','topk'),
                                 topk=float(cfg.get('topk', 0.1)),
                                 threshold=cfg.get('threshold', None),
                                 dilate=cfg.get('dilate', 3))
    # original prob
    orig_prob = model_wrapper.predict_prob(vol)
    # masked out volume
    masked_vol = apply_mask_out(vol, mask)
    masked_prob = model_wrapper.predict_prob(masked_vol)
    prob_drop = orig_prob - masked_prob
    result = {
        'study_id': study_id,
        'orig_prob': float(orig_prob),
        'masked_prob': float(masked_prob),
        'prob_drop': float(prob_drop)
    }
    return result, vol, cam, mask, masked_vol

def save_visuals(study_id, out_dir, vol, cam, mask, masked_vol):
    vis_dir = os.path.join(out_dir, 'visuals')
    os.makedirs(vis_dir, exist_ok=True)
    # create a simple montage: central 9 slices before & after with CAM overlay on original
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
        axs[idx].imshow(masked_vol[s], cmap='gray')
        axs[idx].axis('off')
        idx += 1
    plt.suptitle(f"{study_id} — left: orig+CAM, right: masked-out")
    p = os.path.join(vis_dir, f"{study_id}_comprehensiveness.png")
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
    # CLI overrides
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
    logging.info("Starting comprehensiveness run with config: %s", cfg)

    # study list
    if args.study_list:
        with open(args.study_list, 'r') as f:
            studies = [l.strip() for l in f if l.strip()]
    else:
        studies = find_studies_from_dir(cfg['volumes_dir'])

    model = ModelWrapper(cfg['model_path'], device=cfg['device'])

    results = []
    csv_path = os.path.join(cfg['out_dir'], 'comprehensiveness_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['study_id','orig_prob','masked_prob','prob_drop'])
        writer.writeheader()
        for s in tqdm(studies, desc="Studies"):
            try:
                res, vol, cam, mask, masked_vol = run_one_study(s, cfg, model)
                writer.writerow(res)
                results.append(res)
                # save json per-study
                with open(os.path.join(cfg['out_dir'], f"{s}_comprehensiveness.json"), 'w') as jf:
                    json.dump(res, jf, indent=2)
                # save visual
                try:
                    save_visuals(s, cfg['out_dir'], vol, cam, mask, masked_vol)
                except Exception as e:
                    logging.warning("Failed to save visual for %s: %s", s, str(e))
            except Exception as e:
                logging.exception("Failed for study %s: %s", s, str(e))
                continue

    # aggregate
    drops = np.array([r['prob_drop'] for r in results if 'prob_drop' in r])
    summary = {
        'n_studies': len(results),
        'mean_drop': float(np.mean(drops)) if drops.size else None,
        'median_drop': float(np.median(drops)) if drops.size else None,
        'std_drop': float(np.std(drops)) if drops.size else None,
    }
    with open(os.path.join(cfg['out_dir'], 'comprehensiveness_scores.json'), 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)

    logging.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main_cli()
