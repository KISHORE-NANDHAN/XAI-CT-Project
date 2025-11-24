# ============================================================
# File: scripts/create_nii_volumes_all_flat_multiclass.py
# Description: Merge 2D slices (flat folders) from multiple datasets
#              into 3D .nii.gz volumes (per patient)
# Supports: COVID-CT-MD, COVIDx, SARS-CoV2
# Outputs multi-class folders: COVID, NORMAL, PNEUMONIA, CAP
# ============================================================

import os
import re
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from collections import defaultdict

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
input_roots = [
    "data/preprocessed/COVID-CT-MD",
    "data/preprocessed/COVIDx",
    "data/preprocessed/SARS-CoV2"
]
output_root = "data/ct3d_dataset_raw"
os.makedirs(output_root, exist_ok=True)

# regex pattern to extract patient ID before last underscore + number
patient_id_pattern = re.compile(r"^(.*?)(?:[_-]?\d+)\.[a-zA-Z]+$")

# ---------------------------------------------
# Helper: build a 3D volume
# ---------------------------------------------
def build_volume(slice_paths, save_path):
    files_sorted = sorted(slice_paths, key=lambda x: int(re.findall(r"(\d+)", os.path.basename(x))[-1]))
    imgs = []
    for f in files_sorted:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imgs.append(img)
    if len(imgs) < 3:
        return False
    vol = np.stack(imgs, axis=-1)
    vol = vol.astype(np.int16)
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), save_path)
    return True

# ---------------------------------------------
# Helper: process one folder (class)
# ---------------------------------------------
def process_flat_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    grouped = defaultdict(list)

    for path in glob(os.path.join(input_folder, "*")):
        fname = os.path.basename(path)
        match = patient_id_pattern.match(fname)
        if not match:
            continue
        pid = match.group(1)
        grouped[pid].append(path)

    print(f"📁 [{os.path.basename(input_folder)}] Found {len(grouped)} patient groups.")
    saved_count = 0

    for pid, slice_paths in tqdm(grouped.items(), desc=f"Building volumes → {os.path.basename(input_folder)}"):
        save_path = os.path.join(output_folder, f"{pid}.nii.gz")
        if build_volume(slice_paths, save_path):
            saved_count += 1

    print(f"✅ Saved {saved_count} volumes to {output_folder}")

# ---------------------------------------------
# STEP 1 - Loop through datasets & detect classes
# ---------------------------------------------
for dataset in input_roots:
    if not os.path.exists(dataset):
        print(f"⚠️ Skipping missing dataset: {dataset}")
        continue

    # Define class keyword lists (case-insensitive)
    covid_names = ["COVID", "COVID19", "COVID-19", "SARS", "POSITIVE"]
    pneumonia_names = ["PNEUMONIA"]
    normal_names = ["NORMAL"]
    cap_names = ["CAP"]
    noncovid_names = ["NONCOVID", "NON-COVID", "NEGATIVE"]

    subdirs = glob(os.path.join(dataset, "*"))
    for cls_dir in subdirs:
        cls_name = os.path.basename(cls_dir).upper()

        # Determine target subfolder
        if any(k in cls_name for k in covid_names):
            target_folder = os.path.join(output_root, "COVID")
        elif any(k in cls_name for k in pneumonia_names):
            target_folder = os.path.join(output_root, "PNEUMONIA")
        elif any(k in cls_name for k in normal_names):
            target_folder = os.path.join(output_root, "NORMAL")
        elif any(k in cls_name for k in cap_names):
            target_folder = os.path.join(output_root, "CAP")
        elif any(k in cls_name for k in noncovid_names):
            target_folder = os.path.join(output_root, "nonCOVID")
        else:
            print(f"⚠️ Skipping unknown class folder: {cls_dir}")
            continue

        process_flat_folder(cls_dir, target_folder)

print("\n✅ All datasets processed successfully!")
print("📦 Final multi-class output organized under:", output_root)
