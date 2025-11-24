#!/usr/bin/env python3
"""
prepare_3d_volumes.py
Convert CT3D NIfTI (.nii/.nii.gz) or .npy volumes into:
  - volume.npy  (float32, original orientation after moveaxis -> D,H,W)
  - slices/     (PNG files per axial slice, windowed & resized)

Heuristics:
 - Detect which axis is depth (D) and reorder to (D, H, W)
 - If data values look like HU (min < -500 or max > 2000) we apply HU window.
 - If data looks already in 0..255 we use that directly (still normalise/rescale).
"""

import os
import sys
import argparse
import numpy as np
import cv2

try:
    import nibabel as nib
except ImportError:
    print("❌ nibabel not installed. Install with: pip install nibabel")
    sys.exit(1)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def window_hu_to_uint8(slice_arr, center=-600.0, width=1500.0):
    low = center - width / 2.0
    high = center + width / 2.0
    s = np.clip(slice_arr, low, high)
    s = (s - low) / (high - low)
    s = np.clip(s, 0.0, 1.0)
    return (s * 255.0).astype(np.uint8)


def normalize_uint8(slice_arr):
    # Input expected numeric; scale to 0..255
    mn, mx = float(np.min(slice_arr)), float(np.max(slice_arr))
    if mx <= mn:
        return np.zeros_like(slice_arr, dtype=np.uint8)
    s = (slice_arr - mn) / (mx - mn)
    s = np.clip(s, 0.0, 1.0)
    return (s * 255.0).astype(np.uint8)


def load_volume(path):
    path = str(path)
    if path.lower().endswith(".npy"):
        vol = np.load(path)
        vol = np.nan_to_num(vol).astype(np.float32)
        return vol
    # nifti
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    vol = np.nan_to_num(vol)
    return vol


def detect_depth_axis(shape):
    # shape is a tuple, typical 3D volumes: (D,H,W) or (H,W,D)
    if len(shape) != 3:
        return None
    a, b, c = shape
    # If last axis is much smaller than first two -> likely depth
    if c < min(a, b):
        return 2
    if a < min(b, c):
        return 0
    if b < min(a, c):
        return 1
    # fallback: choose axis with value < 512 (prefer smaller one)
    min_axis = int(np.argmin(shape))
    return min_axis


def process_single_volume(vol_path, out_dir, img_size=224, center=-600.0, width=1500.0, verbose=True):
    if verbose:
        print(f"[process] {vol_path}")

    vol = load_volume(vol_path)  # numpy array

    if vol.ndim == 4:
        # could be (C,H,W,D) or (T,H,W,C) etc. Try to reduce
        # If first dim is small (<=4) assume channels: collapse by taking first channel
        if vol.shape[0] in (1, 3, 4):
            vol = vol[0]
        else:
            # try to squeeze singleton dims
            vol = np.squeeze(vol)
            if vol.ndim != 3:
                raise RuntimeError(f"Unsupported 4D shape after squeeze: {vol.shape}")

    if vol.ndim != 3:
        raise RuntimeError(f"Unsupported volume ndim != 3: {vol.shape}")

    # detect depth axis and move it to axis 0 -> (D, H, W)
    depth_axis = detect_depth_axis(vol.shape)
    if depth_axis is None:
        raise RuntimeError(f"Could not detect depth axis for shape {vol.shape}")
    if depth_axis != 0:
        vol = np.moveaxis(vol, depth_axis, 0)  # now D,H,W

    D, H, W = vol.shape
    if verbose:
        print(f"  -> Reordered shape (D,H,W): {vol.shape}")

    ensure_dir(out_dir)
    out_vol = os.path.join(out_dir, "volume.npy")
    np.save(out_vol, vol.astype(np.float32))
    if verbose:
        print(f"  [save] volume.npy -> {out_vol}")

    # Determine whether values are HU-like or already 0..255
    vmin, vmax = float(vol.min()), float(vol.max())
    hu_like = (vmin < -500.0) or (vmax > 2000.0)
    if verbose:
        print(f"  value range: {vmin:.1f} .. {vmax:.1f}  -> {'HU-like' if hu_like else '8-bit-like'}")

    slices_dir = os.path.join(out_dir, "slices")
    ensure_dir(slices_dir)

    for i in range(D):
        sl = vol[i]  # H x W

        if hu_like:
            img = window_hu_to_uint8(sl, center=center, width=width)
        else:
            # already in 0..255 or small-range; normalize to 0..255
            img = normalize_uint8(sl)

        # resize if needed
        if (img_size is not None) and (img.shape[0] != img_size or img.shape[1] != img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        out_png = os.path.join(slices_dir, f"slice_{i:03d}.png")
        cv2.imwrite(out_png, img)

    if verbose:
        print(f"  [save] {D} PNG slices -> {slices_dir}\n")


def parse_args():
    p = argparse.ArgumentParser(description="CT3D -> volume.npy + PNG slices converter (robust axis detection)")
    p.add_argument("--root", type=str, required=True, help="Input CT3D dataset root (folder with classes: CAP,COVID,...)")
    p.add_argument("--out_root", type=str, required=True, help="Output preprocessed directory")
    p.add_argument("--img_size", type=int, default=224, help="Target slice image size (square)")
    p.add_argument("--center", type=float, default=-600.0, help="HU window center (only used if volume looks like HU)")
    p.add_argument("--width", type=float, default=1500.0, help="HU window width")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()

    root = args.root
    out_root = args.out_root
    ensure_dir(out_root)

    categories = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    print(f"[main] Categories found: {categories}")

    for cat in categories:
        cat_in = os.path.join(root, cat)
        cat_out = os.path.join(out_root, cat)
        ensure_dir(cat_out)

        # collect nifti and npy files (not directories)
        vols = [f for f in os.listdir(cat_in) if f.lower().endswith((".nii", ".nii.gz", ".npy"))]
        if not vols:
            print(f"[main]  - No valid volumes found in {cat_in} (skipping)")
            continue

        for v in sorted(vols):
            name_no_ext = os.path.splitext(os.path.basename(v))[0]
            # handle .nii.gz two extensions
            if name_no_ext.endswith(".nii"):
                name_no_ext = os.path.splitext(name_no_ext)[0]
            case_dir = os.path.join(cat_out, name_no_ext)
            ensure_dir(case_dir)
            try:
                process_single_volume(
                    os.path.join(cat_in, v),
                    case_dir,
                    img_size=args.img_size,
                    center=args.center,
                    width=args.width,
                    verbose=args.verbose
                )
            except Exception as e:
                print(f"[ERROR] Failed to process {v}: {e}")

    print("\n✅ All volumes processed. Use --preprocessed_root <out_root> for M7 Grad-CAM.")


if __name__ == "__main__":
    main()
