# ============================================================
# File: scripts/prepare_mil_data.py
# Description: Merge embeddings from all datasets for M4
# ============================================================

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Dataset class for study embeddings
# ============================================================

class StudyEmbeddingDataset(Dataset):
    def __init__(self, root: str, max_slices: int = 256, pad_value: float = 0.0):
        self.root = Path(root)
        self.max_slices = max_slices
        self.pad_value = pad_value

        self.entries = []
        if not self.root.exists():
            print(f"⚠️ Missing root: {self.root}")
            return

        for p in self.root.rglob("embeddings.npy"):
            self.entries.append(p.parent)

        print(f"✅ Found {len(self.entries)} study embeddings under {self.root}")

    def _load_entry(self, p: Path):
        emb_file = p / "embeddings.npy"
        meta_file = p / "metadata.json"
        emb = np.load(emb_file)
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return emb, meta

    def __getitem__(self, idx):
        p = self.entries[idx]
        emb, meta = self._load_entry(p)
        N, D = emb.shape

        # Clip or pad
        if N > self.max_slices:
            indices = np.linspace(0, N - 1, num=self.max_slices, dtype=int)
            emb = emb[indices]
            slc_indices = indices
        else:
            slc_indices = np.arange(N)
            pad = self.max_slices - N
            if pad > 0:
                emb = np.concatenate(
                    [emb, np.full((pad, D), self.pad_value, dtype=emb.dtype)], axis=0
                )

        label = int(meta.get("label", 0))
        study_uid = meta.get("study_uid", p.stem)
        return torch.from_numpy(emb).float(), label, study_uid, torch.from_numpy(slc_indices).long()

    def __len__(self):
        return len(self.entries)


def collate_fn(batch):
    embs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    uids = [b[2] for b in batch]
    slc_indices = [b[3] for b in batch]
    return embs, labels, uids, slc_indices


# ============================================================
# Merge multiple datasets
# ============================================================

def prepare_mil_data(mil_root: str, out_root: str):
    mil_root = Path(mil_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = ["COVIDx", "SARS-CoV2", "COVID-CT-MD"]
    splits = ["train", "val", "test"]

    for split in splits:
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        total = 0

        for dataset in datasets:
            split_path = mil_root / dataset / split
            if not split_path.exists():
                print(f"⚠️ Missing: {split_path}")
                continue

            for study_dir in split_path.iterdir():
                if not study_dir.is_dir():
                    continue

                dest = split_out / f"{dataset}_{study_dir.name}"
                shutil.copytree(study_dir, dest, dirs_exist_ok=True)
                total += 1

        print(f"✅ Merged {total} studies into {split_out}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare unified MIL dataset")
    parser.add_argument("--mil_root", type=str, required=True, help="Path to per-dataset MIL root (data/mil)")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory for unified dataset (data/mil_prepared)")
    args = parser.parse_args()

    prepare_mil_data(args.mil_root, args.out_root)
