# ============================================================
# File: scripts/m5/dataset_loader_3d.py
# 4-class loader + faster settings
# ============================================================

import os
import warnings
from glob import glob

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _load_raw_volume(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        vol = np.load(path)
    elif path.endswith(".nii.gz"):
        vol = nib.load(path).get_fdata()
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return vol.astype(np.float32)


def _window_and_normalize(vol: np.ndarray, win=(-1000.0, 400.0)) -> np.ndarray:
    lo, hi = win
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-6)
    return vol.astype(np.float32)


def _to_chw_tensor(vol: np.ndarray) -> torch.Tensor:
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W], got {vol.shape}")
    return torch.from_numpy(np.expand_dims(vol, 0)).contiguous()  # [1,D,H,W]


def _spatial_resize(x: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    _, D, H, W = x.shape
    x = x.unsqueeze(0)  # [1,1,D,H,W]
    x = F.interpolate(x, size=(D, th, tw), mode="trilinear", align_corners=False)
    return x.squeeze(0)  # [1,D,th,tw]


def _depth_crop_or_pad(x: torch.Tensor, td: int) -> torch.Tensor:
    _, D, _, _ = x.shape
    if D == td:
        return x
    if D > td:
        s = (D - td) // 2
        return x[:, s:s + td]
    pad = td - D
    return F.pad(x, (0, 0, 0, 0, 0, pad), value=0.0)


def load_volume(path: str, target_d=96, target_h=160, target_w=160, window=(-1000, 400)) -> torch.Tensor:
    vol = _load_raw_volume(path)
    vol = _window_and_normalize(vol, window)
    ten = _to_chw_tensor(vol)                  # [1,D,H,W]
    ten = _spatial_resize(ten, target_h, target_w)
    ten = _depth_crop_or_pad(ten, target_d)
    return ten  # float32 in [0,1]


class CT3DDataset(Dataset):
    def __init__(self, data_dir: str, split="train",
                 target_d=96, target_h=160, target_w=160, max_items=None):
        self.root = os.path.join(data_dir, split)
        self.target_d, self.target_h, self.target_w = int(target_d), int(target_h), int(target_w)

        # discover classes by subfolders
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Split folder not found: {self.root}")
        class_dirs = [d for d in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, d))]
        if len(class_dirs) < 2:
            raise RuntimeError(f"Expected ≥2 class folders under {self.root}, found: {class_dirs}")

        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        samples, labels = [], []
        for c in class_dirs:
            cdir = os.path.join(self.root, c)
            files = sorted(glob(os.path.join(cdir, "**", "*.npy"), recursive=True) +
                           glob(os.path.join(cdir, "**", "*.nii.gz"), recursive=True))
            for f in files:
                samples.append(f)
                labels.append(self.class_to_idx[c])

        if len(samples) == 0:
            raise RuntimeError(f"No .npy/.nii.gz files found under class folders in {self.root}")

        if max_items is not None:
            samples, labels = samples[:int(max_items)], labels[:int(max_items)]

        self.samples, self.labels = samples, labels
        print(f"[INFO] {split}: classes={self.class_to_idx}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x = load_volume(self.samples[idx], self.target_d, self.target_h, self.target_w)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def get_3d_dataloaders(data_dir: str, batch_size=2, num_workers=4,
                       target_d=96, target_h=160, target_w=160, max_items_debug=None):
    train_ds = CT3DDataset(data_dir, "train", target_d, target_h, target_w, max_items_debug)
    val_ds   = CT3DDataset(data_dir, "val",   target_d, target_h, target_w, max_items_debug)
    test_ds  = CT3DDataset(data_dir, "test",  target_d, target_h, target_w, max_items_debug)

    print(f"[INFO] Found {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test volumes")
    print(f"[INFO] Target shape = [1, {target_d}, {target_h}, {target_w}]")

    use_workers = max(0, int(num_workers))
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=use_workers, pin_memory=pin,
                  persistent_workers=(use_workers > 0))
    if use_workers > 0:
        common["prefetch_factor"] = 4

    train_dl = DataLoader(train_ds, shuffle=True,  **common)
    val_dl   = DataLoader(val_ds,   shuffle=False, **common)
    test_dl  = DataLoader(test_ds,  shuffle=False, **common)
    return train_dl, val_dl, test_dl
