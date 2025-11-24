# ============================================================
# File: scripts/merge_ct3d_split_multiclass.py
# Description: Split multi-class 3D dataset into train/val/test
# Classes: COVID, NORMAL, PNEUMONIA, CAP
# ============================================================

import os
import shutil
import random
from glob import glob

random.seed(42)

source_root = "data/ct3d_dataset_raw"
target_root = "data/ct3d_dataset"

splits = ["train", "val", "test"]
ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

classes = ["COVID", "NORMAL", "PNEUMONIA", "CAP"]

# Create folders
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

def split_files(file_list):
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(ratios["train"] * n)
    n_val = int(ratios["val"] * n)
    return {
        "train": file_list[:n_train],
        "val": file_list[n_train:n_train+n_val],
        "test": file_list[n_train+n_val:]
    }

for cls in classes:
    files = glob(os.path.join(source_root, cls, "*.nii.gz"))
    if len(files) == 0:
        print(f"⚠️ No files found for class: {cls}")
        continue

    print(f"📦 Splitting {cls}: {len(files)} files")
    split_sets = split_files(files)

    for split, paths in split_sets.items():
        out_dir = os.path.join(target_root, split, cls)
        for p in paths:
            shutil.copy(p, out_dir)

print("\n✅ Multi-class dataset split successfully!")
print(f"📦 Output structure: {target_root}/[train|val|test]/[class]/")
