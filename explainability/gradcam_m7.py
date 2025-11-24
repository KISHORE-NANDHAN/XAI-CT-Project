#!/usr/bin/env python3
"""
gradcam_m7.py

Finalized Grad-CAM tool for M7 (Explainability & Attention Visualization)

- Primary 2D target: Slice encoder (SliceClassifier from models/slice_model.py or get_model(cfg))
- Primary 3D target: 3D encoder (models/resnet3d.py)
- Robust loading of state_dicts (handles OrderedDict, nested dicts, prefixes)
- Saves overlays (.png) and raw CAM arrays (.npy) per study and manifest_all.json

Usage example (Windows PowerShell / cmd):
python explainability/gradcam_m7.py ^
  --slice_checkpoint "d:/xai-ct-project - Copy/outputs/checkpoints/best_model.pt" ^
  --slice_model_file "d:/xai-ct-project - Copy/models/slice_model.py" ^
  --slice_model_class "SliceClassifier" ^
  --checkpoint3d "d:/xai-ct-project - Copy/outputs/m5/checkpoints/best.pth" ^
  --model_class_file3d "d:/xai-ct-project - Copy/models/resnet3d.py" ^
  --model_class3d "ResNet3D" ^
  --preprocessed_root "d:/xai-ct-project - Copy/data/preprocessed" ^
  --out_root "d:/xai-ct-project - Copy/outputs/m7/attention_heatmaps" ^
  --top_k 5 ^
  --device cuda:0
"""
import os
import sys
import glob
import json
import argparse
import importlib.util
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2

# nibabel optional for NIfTI
try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

# --------------------------
# Utilities
# --------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_nifti(path: str) -> np.ndarray:
    if not _HAS_NIB:
        raise RuntimeError("nibabel not installed. Install nibabel to load NIfTI files.")
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)

def load_npy(path: str) -> np.ndarray:
    return np.load(path)

def window_image(img: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    """
    Apply HU window and normalize to 0–255 in uint8.
    Handles several input types:
      - already uint8 in 0..255
      - floats in 0..1
      - HU-like floats in other ranges (apply window)
    """
    # Already 0..255 uint8
    if img.dtype == np.uint8 and img.max() <= 255 and img.min() >= 0:
        return img
    # Already normalized float 0..1
    if img.dtype in (np.float32, np.float64) and img.max() <= 1.0 and img.min() >= 0.0:
        return (img * 255.0).astype(np.uint8)

    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(img, low, high)
    norm = (clipped - low) / (high - low)
    return (np.uint8(np.clip(norm, 0.0, 1.0) * 255.0))

def save_overlay(base_img: np.ndarray, cam: np.ndarray, out_path: str, alpha: float = 0.45):
    """Save an overlay of cam (0..1) on base_img. Resizes cam to base image if needed."""
    try:
        # Ensure base_img is uint8 HxW or HxWx3
        if base_img.dtype != np.uint8:
            base_img = np.uint8(np.clip(base_img, 0, 255))
        if base_img.ndim == 2:
            base_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        elif base_img.ndim == 3 and base_img.shape[2] == 3:
            base_bgr = base_img
        else:
            # fallback: take first channel
            base_bgr = cv2.cvtColor(base_img[..., 0], cv2.COLOR_GRAY2BGR)

        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam_min, cam_max = float(np.nanmin(cam)), float(np.nanmax(cam))
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam, dtype=np.float32)

        H, W = base_bgr.shape[:2]
        if cam_norm.shape != (H, W):
            cam_norm = cv2.resize(cam_norm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

        heat = cv2.applyColorMap(np.uint8(cam_norm * 255.0), cv2.COLORMAP_JET)
        out = cv2.addWeighted(heat, alpha, base_bgr, 1.0 - alpha, 0)
        cv2.imwrite(out_path, out)
    except Exception as e:
        print(f"[save_overlay] Failed to write overlay {out_path}: {e}")

# --------------------------
# Grad-CAM implementations
# --------------------------
class HookManager:
    def __init__(self):
        self.handles = []
        self.activations = None
        self.gradients = None

    def clear(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self.activations = None
        self.gradients = None

class GradCAM2D:
    def __init__(self, model: nn.Module, device: torch.device, target_layer: Optional[str] = None):
        self.model = model.to(device)
        self.device = device
        self.hooks = HookManager()
        self.target_layer = target_layer
        self._register_hooks()

    def _find_conv2d_name(self):
        # iterate reversed — deepest conv wins
        for name, m in reversed(list(self.model.named_modules())):
            if isinstance(m, nn.Conv2d):
                return name
        raise RuntimeError("No Conv2D layer found in model for Grad-CAM.")

    def _register_hooks(self):
        name = self.target_layer or self._find_conv2d_name()
        module = dict(self.model.named_modules()).get(name, None)
        if module is None:
            raise RuntimeError(f"Target layer '{name}' not found in model modules.")
        def fwd(_, __, out):
            self.hooks.activations = out.detach()
        def bwd(_, grad_in, grad_out):
            # some backward hooks provide gradients differently; guard access
            g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
            self.hooks.gradients = g.detach()
        self.hooks.handles.append(module.register_forward_hook(fwd))
        try:
            self.hooks.handles.append(module.register_backward_hook(bwd))
        except Exception:
            if hasattr(module, "register_full_backward_hook"):
                self.hooks.handles.append(module.register_full_backward_hook(lambda m, gi, go: bwd(m, gi, go)))
            else:
                raise
        print(f"[GradCAM2D] Registered hooks on {name}")

    def compute_cam(self, x: torch.Tensor, class_idx: Optional[int] = None, upsample_size: Optional[Tuple[int,int]] = None) -> np.ndarray:
        self.model.zero_grad()
        x = x.to(self.device)
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        A = self.hooks.activations.cpu().numpy()[0]  # C x h x w
        G = self.hooks.gradients.cpu().numpy()[0]    # C x h x w
        weights = np.mean(G, axis=(1,2))
        cam = np.zeros(A.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * A[i]
        cam = np.maximum(cam, 0)
        if upsample_size is not None:
            cam = cv2.resize(cam, (upsample_size[1], upsample_size[0]), interpolation=cv2.INTER_LINEAR)
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
        return cam

    def cleanup(self):
        self.hooks.clear()

class GradCAM3D:
    def __init__(self, model: nn.Module, device: torch.device, target_layer: Optional[str] = None):
        self.model = model.to(device)
        self.device = device
        self.hooks = HookManager()
        self.target_layer = target_layer
        self._register_hooks()

    def _find_conv3d_name(self):
        for name, m in reversed(list(self.model.named_modules())):
            if isinstance(m, nn.Conv3d):
                return name
        raise RuntimeError("No Conv3D layer found in model for Grad-CAM.")

    def _register_hooks(self):
        name = self.target_layer or self._find_conv3d_name()
        module = dict(self.model.named_modules()).get(name, None)
        if module is None:
            raise RuntimeError(f"Target layer '{name}' not found in model modules.")
        def fwd(_, __, out):
            self.hooks.activations = out.detach()
        def bwd(_, grad_in, grad_out):
            g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
            self.hooks.gradients = g.detach()
        self.hooks.handles.append(module.register_forward_hook(fwd))
        try:
            self.hooks.handles.append(module.register_backward_hook(bwd))
        except Exception:
            if hasattr(module, "register_full_backward_hook"):
                self.hooks.handles.append(module.register_full_backward_hook(lambda m, gi, go: bwd(m, gi, go)))
            else:
                raise
        print(f"[GradCAM3D] Registered hooks on {name}")

    def compute_cam_volume(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        x = x.to(self.device)
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        A = self.hooks.activations.cpu().numpy()[0]  # C x D x H x W
        G = self.hooks.gradients.cpu().numpy()[0]    # C x D x H x W
        weights = np.mean(G, axis=(1,2,3))
        cam_vol = np.zeros(A.shape[1:], dtype=np.float32)  # D x H x W
        for i, w in enumerate(weights):
            cam_vol += w * A[i]
        cam_vol = np.maximum(cam_vol, 0)
        if cam_vol.max() > 0:
            cam_vol = cam_vol - cam_vol.min()
            cam_vol = cam_vol / (cam_vol.max() + 1e-8)
        return cam_vol

    def project_mip(self, cam_vol: np.ndarray, axis: int = 0) -> np.ndarray:
        mip = np.max(cam_vol, axis=axis)
        if mip.max() > 0:
            mip = mip - mip.min()
            mip = mip / (mip.max() + 1e-8)
        return mip

    def cleanup(self):
        self.hooks.clear()

# --------------------------
# Dynamic import helpers
# --------------------------
def dynamic_import_from_file(path: str, symbol: Optional[str] = None):
    """
    Import a module from a .py file and optionally return a symbol (class/function).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    spec = importlib.util.spec_from_file_location("user_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if symbol:
        if not hasattr(module, symbol):
            raise AttributeError(f"Symbol {symbol} not found in {path}")
        return getattr(module, symbol)
    return module

# --------------------------
# Robust state_dict loader
# --------------------------
def robust_load_state_dict_into(model: nn.Module, state_obj: Any, strict: bool = False):
    """
    Accepts:
      - OrderedDict (state_dict)
      - dict with 'state_dict' key
      - full saved checkpoint dict
    Attempts to load state by handling prefixes like 'module.' or top-level 'model.'.
    """
    if isinstance(state_obj, dict) and "state_dict" in state_obj:
        state = state_obj["state_dict"]
    else:
        state = state_obj

    try:
        model.load_state_dict(state, strict=strict)
        return
    except Exception as e1:
        fixed = {}
        for k, v in state.items():
            newk = k
            if newk.startswith("module."):
                newk = newk[7:]
            if newk.startswith("model."):
                newk = newk[len("model."):]
            fixed[newk] = v
        try:
            model.load_state_dict(fixed, strict=strict)
            return
        except Exception as e2:
            stripped2 = {}
            for k, v in fixed.items():
                parts = k.split(".", 1)
                if len(parts) == 2:
                    stripped2[parts[1]] = v
            try:
                model.load_state_dict(stripped2, strict=strict)
                return
            except Exception as e3:
                raise RuntimeError(f"Failed to load state_dict into model. Errors:\n1) {e1}\n2) {e2}\n3) {e3}")

# --------------------------
# Model loaders (slice CNN + 3D)
# --------------------------
def load_slice_model(slice_model_file: str,
                     slice_model_class: Optional[str],
                     slice_model_cfg: Optional[str],
                     device: torch.device,
                     slice_checkpoint: str):
    """
    Loads the slice encoder for 2D Grad-CAM.
    Supports:
      - SliceClassifier returned by get_model(cfg)
      - Any class defined in slice_model.py
    """
    module = dynamic_import_from_file(slice_model_file)
    model = None

    if slice_model_class:
        if not hasattr(module, slice_model_class):
            raise AttributeError(f"{slice_model_class} not found in {slice_model_file}")
        cls_or_func = getattr(module, slice_model_class)

        if isinstance(cls_or_func, type) and issubclass(cls_or_func, nn.Module):
            try:
                model = cls_or_func()
            except Exception as e:
                raise RuntimeError(f"Could not instantiate class {slice_model_class}: {e}")
        elif callable(cls_or_func):
            try:
                import ast
                if slice_model_cfg:
                    cfg = ast.literal_eval(slice_model_cfg)
                else:
                    cfg = {"name": "resnet50", "pretrained": False, "num_classes": 2}
                model = cls_or_func(cfg)
            except Exception as e:
                raise RuntimeError(f"Could not call {slice_model_class}(cfg): {e}")

    if model is None and hasattr(module, "get_model"):
        get_model_func = getattr(module, "get_model")
        if callable(get_model_func):
            import ast
            try:
                if slice_model_cfg:
                    cfg = ast.literal_eval(slice_model_cfg)
                else:
                    cfg = {"name": "resnet50", "pretrained": False, "num_classes": 2}
                model = get_model_func(cfg)
            except Exception as e:
                raise RuntimeError(f"Fallback get_model(cfg) failed: {e}")

    if model is None:
        raise RuntimeError(f"Failed to construct 2D slice model from {slice_model_file}.")

    # -----------------------------
    # Load weights
    # -----------------------------
    ckpt = torch.load(slice_checkpoint, map_location='cpu')
    try:
        robust_load_state_dict_into(model, ckpt, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load slice checkpoint: {e}")

    model.to(device).eval()
    print(f"[slice] ✓ Loaded 2D model from {slice_model_file} + weights")
    return model

def try_instantiate_3d_model_from_module(module_path):
    """
    Try to import common ResNet3D classes from the given models.resnet3d module.
    """
    module = dynamic_import_from_file(module_path)
    candidates = []
    for name in dir(module):
        attr = getattr(module, name)
        try:
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                candidates.append((name, attr))
        except Exception:
            continue
    preferred = ["ResNet3D", "ResNet3D18", "ResNet3D_18", "resnet3d", "resnet18_3d", "ResNet"]
    for p in preferred:
        for name, cls in candidates:
            if p.lower() in name.lower():
                try:
                    model = cls()
                    return model
                except Exception:
                    continue
    if candidates:
        name, cls = candidates[0]
        try:
            model = cls()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate any candidate class from {module_path}: {e}")
    raise RuntimeError(f"No suitable nn.Module class found in {module_path}")

def load_3d_model(checkpoint3d: str, device: torch.device, model_class_file3d: Optional[str], model_class3d: Optional[str]):
    ckpt = torch.load(checkpoint3d, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        ckpt.to(device).eval()
        return ckpt

    state = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]

    if model_class_file3d and model_class3d:
        cls = dynamic_import_from_file(model_class_file3d, model_class3d)
        try:
            model = cls()
            robust_load_state_dict_into(model, state, strict=False)
            model.to(device).eval()
            print(f"[3d] Loaded 3D model {model_class3d} from {model_class_file3d}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate/load 3D model {model_class3d}: {e}")

    module_guess = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "resnet3d.py"))
    if os.path.exists(module_guess):
        model = try_instantiate_3d_model_from_module(module_guess)
        try:
            robust_load_state_dict_into(model, state, strict=False)
            model.to(device).eval()
            print(f"[3d] Loaded 3D model via fallback from {module_guess}")
            return model
        except Exception as e:
            raise RuntimeError(f"[3d] Fallback load failed: {e}")

    raise RuntimeError("Could not load 3D model. Provide --model_class_file3d and --model_class3d.")

# --------------------------
# Processing pipelines
# --------------------------

def process_study_2d(model: nn.Module, gradcam: GradCAM2D,
                     study_path: str, out_dir: str, top_k: int, device: torch.device):

    # ------------------------------------------
    # STRICT INPUT FILTERING (NEW)
    # ------------------------------------------
    slice_dir = os.path.join(study_path, "slices")
    png_slices = sorted(glob.glob(os.path.join(slice_dir, "*.png")))

    vol_path = os.path.join(study_path, "volume.npy")

    slices = None

    if len(png_slices) > 0:
        # Use ONLY slices/*.png
        slices = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                  for f in png_slices]

    elif os.path.exists(vol_path):
        # Fallback to full volume
        vol = load_npy(vol_path)
        slices = [vol[i].astype(np.float32) for i in range(vol.shape[0])]

    else:
        print(f"[2D] No valid slices for {study_path}")
        return None

    # ------------------------------------------
    # Continue with your original pipeline
    # ------------------------------------------
    ensure_dir(out_dir)
    L = len(slices)
    if L == 0:
        print(f"[2D] No readable slices for {study_path}")
        return None

    # pick evenly-spaced indices
    idxs = list(np.linspace(0, L - 1, min(L, top_k)).astype(int))

    results = []

    for idx in idxs:
        s = slices[idx]
        s_win = window_image(s)

        # robust tensor construction
        t = torch.tensor(s_win.astype(np.float32) / 255.0)

        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        elif t.ndim == 3 and t.shape[2] in (1,3):
            t = t.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        else:
            t = t.unsqueeze(0)  # fallback

        t = t.float().to(device)

        # run Grad-CAM
        cam = gradcam.compute_cam(t, upsample_size=(s_win.shape[0], s_win.shape[1]))

        out_png = os.path.join(out_dir, f"slice_{idx:03d}_cam.png")
        save_overlay(s_win, cam, out_png)
        np.save(out_png.replace(".png", ".npy"), cam)

        results.append({
            "slice_index": int(idx),
            "cam_png": out_png,
            "cam_npy": out_png.replace(".png", ".npy")
        })

    return results


def process_study_3d(model: nn.Module, gradcam: GradCAM3D,
                     study_path: str, out_dir: str, top_k: int, device: torch.device):

    # ---------------------------------------------
    # 1) STRICT VOLUME DETECTION
    # ---------------------------------------------
    vol_path = None

    # Prefer volume.npy
    if os.path.exists(os.path.join(study_path, "volume.npy")):
        vol_path = os.path.join(study_path, "volume.npy")

    # Otherwise look for volume_*.npy
    if vol_path is None:
        candidates = sorted(glob.glob(os.path.join(study_path, "volume_*.npy")))
        if len(candidates) > 0:
            vol_path = candidates[0]

    # Try NIfTI only if above missing
    if vol_path is None:
        nii_candidates = (
            glob.glob(os.path.join(study_path, "*.nii")) +
            glob.glob(os.path.join(study_path, "*.nii.gz"))
        )
        if len(nii_candidates) > 0:
            vol_path = nii_candidates[0]

    if vol_path is None:
        print(f"[3D] No valid volume found for {study_path}; skipping")
        return None

    # ---------------------------------------------
    # 2) LOAD VOLUME SAFELY
    # ---------------------------------------------
    if vol_path.endswith(".npy"):
        vol = load_npy(vol_path)
    else:
        vol = load_nifti(vol_path)

    vol = np.nan_to_num(vol).astype(np.float32, copy=False)

    # ---------------------------------------------
    # 3) FIX ORIENTATION (AUTO)
    # Desired = D x H x W
    # ---------------------------------------------
    if vol.ndim != 3:
        print(f"[3D] Invalid shape {vol.shape} in {study_path}")
        return None

    # Check if shape is H x W x D
    if vol.shape[0] == vol.shape[1] and vol.shape[2] < 10:  # weird tiny depth → ignore
        pass

    if vol.shape[0] == 224 and vol.shape[1] == 224 and vol.shape[2] > 10:
        # (H,W,D) → (D,H,W)
        vol = np.transpose(vol, (2, 0, 1))

    D, H, W = vol.shape

    # ---------------------------------------------
    # 4) APPLY WINDOWING (FAST)
    # ---------------------------------------------
    vol_win = window_image(vol)   # window_image supports vector input

    # ---------------------------------------------
    # 5) BUILD 3D TENSOR: shape (1,1,D,H,W)
    # ---------------------------------------------
    t = torch.from_numpy(vol_win.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(device)  # 1x1xD x H x W

    # ---------------------------------------------
    # 6) RUN 3D GRAD-CAM
    # ---------------------------------------------
    cam_vol = gradcam.compute_cam_volume(t)

    ensure_dir(out_dir)
    cam_npy = os.path.join(out_dir, "cam_volume.npy")
    np.save(cam_npy, cam_vol)

    # ---------------------------------------------
    # 7) Generate Volume MIP Overlay
    # ---------------------------------------------
    mip = gradcam.project_mip(cam_vol, axis=0)  # H x W
    mid_slice = vol_win[D // 2]                 # use middle slice
    mip_png = os.path.join(out_dir, "cam_volume_mip.png")
    save_overlay(mid_slice, mip, mip_png)

    # ---------------------------------------------
    # 8) Top-K slices
    # ---------------------------------------------
    slice_scores = cam_vol.max(axis=(1, 2))
    topk = min(top_k, D)
    top_idx = np.argsort(slice_scores)[-topk:][::-1]

    per_slice = []
    for i in top_idx:
        s_img = vol_win[i]
        s_cam = cam_vol[i]

        out_png = os.path.join(out_dir, f"slice_{int(i):03d}_cam.png")
        save_overlay(s_img, s_cam, out_png)

        np.save(out_png.replace(".png", ".npy"), s_cam)

        per_slice.append({
            "slice_index": int(i),
            "cam_png": out_png,
            "cam_npy": out_png.replace(".png", ".npy")
        })

    return {
        "cam_volume_npy": cam_npy,
        "mip_png": mip_png,
        "per_slice": per_slice
    }

# --------------------------
# CLI & main
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM M7 unified tool (2D slice encoder + 3D encoder)")

    # Slice encoder (2D)
    p.add_argument("--slice_checkpoint", type=str, help="Slice encoder checkpoint path (state_dict or checkpoint)", required=True)
    p.add_argument("--slice_model_file", type=str, help="Python file path for slice model (e.g. models/slice_model.py)", required=True)
    p.add_argument("--slice_model_class", type=str, help="Class or function name in slice_model file (e.g. SliceClassifier or get_model)")
    p.add_argument("--slice_model_cfg", type=str,help="Python dict string for get_model(cfg). Example: \"{'name':'resnet50','pretrained':False,'num_classes':2}\"")

    # 3D encoder (M5)
    p.add_argument("--checkpoint3d", type=str, help="3D model checkpoint path", required=False)
    p.add_argument("--model_class_file3d", type=str, help="3D model .py file path (optional)")
    p.add_argument("--model_class3d", type=str, help="3D model class name (optional)")

    # Data + outputs
    p.add_argument("--preprocessed_root", type=str, required=True, help="Root with per-study folders containing .npy/.nii/.png slices")
    p.add_argument("--out_root", type=str, required=True, help="Output root for CAMs")
    p.add_argument("--top_k", type=int, default=5, help="Top-k slices for visualization")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target_layer2d", type=str, help="Explicit conv layer name for 2D Grad-CAM (optional)")
    p.add_argument("--target_layer3d", type=str, help="Explicit conv layer name for 3D Grad-CAM (optional)")

    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    ensure_dir(args.out_root)

    study_dirs = sorted([d for d in glob.glob(os.path.join(args.preprocessed_root, "*")) if os.path.isdir(d)])
    if len(study_dirs) == 0:
        print("No study directories found under preprocessed_root:", args.preprocessed_root)
        sys.exit(1)

    # -------------- 2D slice encoder setup --------------
    gradcam2d = None
    model2d = None
    try:
        model2d = load_slice_model(
            slice_model_file=args.slice_model_file,
            slice_model_class=args.slice_model_class,
            slice_model_cfg=args.slice_model_cfg,
            device=device,
            slice_checkpoint=args.slice_checkpoint,
        )

        gradcam2d = GradCAM2D(model2d, device, target_layer=args.target_layer2d)
    except Exception as e:
        print("[ERROR] Could not load slice encoder for 2D Grad-CAM:", e)
        print("Please verify --slice_checkpoint, --slice_model_file and --slice_model_class.")
        sys.exit(1)

    # -------------- 3D model setup (optional) --------------
    gradcam3d = None
    model3d = None
    if args.checkpoint3d:
        try:
            model3d = load_3d_model(args.checkpoint3d, device, args.model_class_file3d, args.model_class3d)
            gradcam3d = GradCAM3D(model3d, device, target_layer=args.target_layer3d)
        except Exception as e:
            print("[ERROR] Could not load 3D model for Grad-CAM:", e)
            gradcam3d = None

    # -------------- Process studies --------------
    manifest = []
    for sd in study_dirs:
        study_name = os.path.basename(sd.rstrip('/\\'))
        print(f"[main] Processing study: {study_name}")
        study_out_root = os.path.join(args.out_root, study_name)
        ensure_dir(study_out_root)
        study_result = {"study": study_name, "2d": None, "3d": None}

        if gradcam2d is not None:
            out2d = os.path.join(study_out_root, "2d")
            ensure_dir(out2d)
            try:
                res2d = process_study_2d(model2d, gradcam2d, sd, out2d, args.top_k, device)
                study_result["2d"] = res2d
            except Exception as e:
                print(f"[main] Error generating 2D CAM for {study_name}: {e}")

        if gradcam3d is not None:
            out3d = os.path.join(study_out_root, "3d")
            ensure_dir(out3d)
            try:
                res3d = process_study_3d(model3d, gradcam3d, sd, out3d, args.top_k, device)
                study_result["3d"] = res3d
            except Exception as e:
                print(f"[main] Error generating 3D CAM for {study_name}: {e}")

        manifest.append(study_result)
        with open(os.path.join(study_out_root, "manifest.json"), "w") as f:
            json.dump(study_result, f, indent=2)

    with open(os.path.join(args.out_root, "manifest_all.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if gradcam2d:
        gradcam2d.cleanup()
    if gradcam3d:
        gradcam3d.cleanup()

    print("[main] Done. Results saved to:", args.out_root)

if __name__ == "__main__":
    main()
