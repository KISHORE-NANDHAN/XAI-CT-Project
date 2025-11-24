# ============================================================
# File: scripts/slice_dataset.py
# Description: Unified loader for multiple CT slice datasets (M3)
# ============================================================

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CovidSliceDataset(Dataset):
    def __init__(self, dataset_roots, split="train", img_size=224):
        """
        dataset_roots: list of dataset directories (e.g. [COVIDx, SARS-CoV2, COVID-CT-MD])
        split: 'train' or 'val'
        """
        self.img_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        for root in dataset_roots:
            manifest_path = os.path.join(root, f"{split}_manifest.json")
            if os.path.exists(manifest_path):
                print(f"📄 Loading manifest from {manifest_path}")
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                # If manifest is a dict with "images" → extract that
                if isinstance(manifest, dict) and "images" in manifest:
                    manifest = manifest["images"]

                # Now manifest should be a list of dicts
                for entry in manifest:
                    file_path = entry.get("file_path") or entry.get("path") or entry.get("image") or ""
                    label = entry.get("label", "")
                    img_path = file_path.replace("\\", "/")  # handle Windows slashes

                    # Fix relative paths
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(os.getcwd(), img_path)

                    # Binary label mapping
                    label_lower = label.lower()
                    if "covid" in label_lower:
                        y = 1
                    elif "non" in label_lower or "normal" in label_lower:
                        y = 0
                    else:
                        # group everything else (e.g. pneumonia, CAP) as non-COVID
                        y = 0

                    if os.path.exists(img_path):
                        self.img_paths.append(img_path)
                        self.labels.append(y)
                    else:
                        print(f"⚠️ Missing file: {img_path}")
            else:
                # Fallback — scan folders manually
                for subdir, dirs, files in os.walk(root):
                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            full_path = os.path.join(subdir, file)
                            label = os.path.basename(os.path.dirname(full_path))
                            y = 1 if "covid" in label.lower() else 0
                            self.img_paths.append(full_path)
                            self.labels.append(y)

        print(f"✅ Loaded {len(self.img_paths)} slices for split='{split}'")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
