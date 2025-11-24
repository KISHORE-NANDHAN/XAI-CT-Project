"""
Extract per-slice embeddings from the trained M3 model for all datasets (COVIDx, SARS-CoV2, COVID-CT-MD)
using manifest files that include file_path and label fields.
Embeddings are grouped per study for use in M4 (MIL Transformer Aggregator).
"""

import os, sys, json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slice_model import get_model


# ===============================================================
# Dataset Loader (Manifest-Based)
# ===============================================================
class ManifestDataset(Dataset):
    """
    Loads slices from a manifest JSON file.
    Each manifest entry may contain:
    {"file_path": "...", "label": "COVID" or "non-COVID"} or
    {"image": "...", "label": 0/1}.
    """

    def __init__(self, manifest_path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.samples = []

        if not self.manifest_path.exists():
            print(f"⚠️ Manifest not found: {self.manifest_path}")
            return

        data = json.loads(self.manifest_path.read_text())
        for item in data:
            # Accept either key name
            img_path = item.get("file_path") or item.get("image")
            if not img_path or not os.path.exists(img_path):
                continue

            raw_label = str(item.get("label", "")).lower()

            # Map textual labels → numeric
            if "covid" in raw_label or "pneumonia" in raw_label or "cap" in raw_label:
                label = 1
            else:
                label = 0

            study_id = Path(img_path).parent.name
            self.samples.append((img_path, label, study_id))

        print(f"✅ Loaded {len(self.samples)} slices from {self.manifest_path.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, study_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, study_id


# ===============================================================
# Embedding Extraction Function
# ===============================================================
def extract_embeddings_for_dataset(model, dataset_root, out_root, device, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_name = Path(dataset_root).name
    print(f"\n🚀 Processing dataset: {dataset_name}")

    for split in ["train", "val", "test"]:
        manifest_path = Path(dataset_root) / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"⚠️ Manifest not found: {manifest_path}")
            continue

        ds = ManifestDataset(manifest_path, transform)
        if len(ds) == 0:
            continue

        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        study_dict = {}

        with torch.no_grad():
            for imgs, labels, study_ids in tqdm(dl, desc=f"Extracting {dataset_name}/{split}"):
                imgs = imgs.to(device)
                logits, feats = model(imgs)
                feats = feats.cpu().numpy()  # [B, D]

                for i, sid in enumerate(study_ids):
                    label = int(labels[i].item())
                    vec = feats[i]
                    if sid not in study_dict:
                        study_dict[sid] = {"embeddings": [], "label": label}
                    study_dict[sid]["embeddings"].append(vec)

        # Save per study
        split_out = Path(out_root) / dataset_name / split
        split_out.mkdir(parents=True, exist_ok=True)

        for sid, data in study_dict.items():
            study_dir = split_out / sid
            study_dir.mkdir(parents=True, exist_ok=True)
            emb = np.array(data["embeddings"])
            np.save(study_dir / "embeddings.npy", emb)
            meta = {"label": data["label"], "study_uid": sid}
            (study_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        print(f"✅ Saved {len(study_dict)} studies → {split_out}")


# ===============================================================
# Main Entry
# ===============================================================
def main(ckpt_path, preprocessed_root, out_root, device="cuda", batch_size=16):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"🔹 Loading model from {ckpt_path}")
    cfg = {"name": "resnet50", "pretrained": False, "num_classes": 2}
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    datasets = ["COVIDx", "SARS-CoV2", "COVID-CT-MD"]
    for dataset in datasets:
        dataset_root = Path(preprocessed_root) / dataset
        if not dataset_root.exists():
            print(f"⚠️ Dataset folder missing: {dataset_root}")
            continue
        extract_embeddings_for_dataset(model, dataset_root, out_root, device, batch_size)

    print(f"\n🎯 All embeddings generated successfully under: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from trained M3 model for all datasets (manifest-based)")
    parser.add_argument("--ckpt", required=True, help="Path to trained slice model checkpoint (M3)")
    parser.add_argument("--preprocessed_root", required=True, help="Root of preprocessed datasets (e.g., data/preprocessed)")
    parser.add_argument("--out_root", required=True, help="Output directory for MIL embeddings (e.g., data/mil)")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    main(args.ckpt, args.preprocessed_root, args.out_root, args.device, args.batch_size)
