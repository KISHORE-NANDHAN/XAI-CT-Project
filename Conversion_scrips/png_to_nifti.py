# ============================================================
# File: scripts/png_to_nifti.py
# XAI-CT PROJECT — Universal PNG→NIfTI Converter (M1 → M2)
# Handles both 2D (SARS-CoV2, COVIDx) and 3D (COVID-CT-MD) datasets
# Stacks 2D slices into 3D NIfTI volumes per class or study.
# ============================================================

import os
import yaml
import json
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ------------------------------------------------------------
# 1️⃣ Load Config
# ------------------------------------------------------------
cfg = yaml.safe_load(open("config/preprocess.yaml"))

DATASETS = cfg["datasets"]["available"]
RAW_PATHS = cfg["paths"]["raw_data"]
OUT_ROOT = Path("data/raw_nifti")
LOG_PATH = Path(cfg["logging"]["file"])

HU_MIN, HU_MAX = cfg["preprocess"]["hu_window"]
TARGET_SIZE = tuple(cfg["preprocess"]["image_size"])

# ------------------------------------------------------------
# 2️⃣ Setup Logging
# ------------------------------------------------------------
OUT_ROOT.mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=getattr(logging, cfg["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if cfg["logging"]["console"]:
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, cfg["logging"]["level"]))
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

logging.info("=== XAI-CT Universal PNG→NIfTI Converter Started ===")

# ------------------------------------------------------------
# 3️⃣ Helper Functions
# ------------------------------------------------------------
def normalize_hu(arr, hu_min, hu_max):
    """Clip HU values and normalize to [0, 1]."""
    arr = np.clip(arr, hu_min, hu_max)
    return (arr - hu_min) / (hu_max - hu_min)

def resize_image(arr, target_size):
    """Resize 2D image array to target dimensions."""
    img = Image.fromarray(arr)
    img = img.resize(target_size, resample=Image.BILINEAR)
    return np.array(img, dtype=np.float32)

def save_nifti(volume, out_path):
    """Save numpy array as NIfTI file."""
    if volume.size == 0:
        raise ValueError("Volume is empty, cannot save.")
    nifti_img = sitk.GetImageFromArray(volume)
    sitk.WriteImage(nifti_img, str(out_path))

# ------------------------------------------------------------
# 4️⃣ Conversion Logic
# ------------------------------------------------------------
def convert_dataset(dataset_name, input_root):
    input_dir = Path(input_root)
    if not input_dir.exists():
        logging.warning(f"⚠️ Dataset folder not found: {input_root}")
        return []

    logging.info(f"Converting dataset: {dataset_name}")
    qc_data = []

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        out_dir = OUT_ROOT / dataset_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Processing class: {class_name}")

        subdirs = [d for d in class_dir.iterdir() if d.is_dir()]
        slice_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))

        # Case 1️⃣: 2D dataset (flat folder of PNGs)
        if slice_files and not subdirs:
            try:
                all_slices = []
                for s in tqdm(sorted(slice_files), desc=f"{dataset_name}/{class_name} (Stacking 2D→3D)"):
                    img = Image.open(s).convert("L")
                    arr = np.array(img, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    arr = resize_image(arr, TARGET_SIZE)
                    arr = normalize_hu(arr, HU_MIN, HU_MAX)
                    all_slices.append(arr)

                if len(all_slices) == 0:
                    logging.warning(f"⚠️ No valid slices found in {class_dir}")
                    continue

                volume = np.stack(all_slices, axis=0)  # (depth, H, W)
                out_path = out_dir / f"{class_name}_volume.nii.gz"
                save_nifti(volume, out_path)

                qc_data.append({
                    "dataset": dataset_name,
                    "class": class_name,
                    "file": out_path.name,
                    "num_slices": len(all_slices),
                    "shape": list(volume.shape),
                    "mean": float(np.mean(volume)),
                    "std": float(np.std(volume)),
                })
                logging.info(f"✅ Saved stacked 3D volume: {out_path.name}")

            except Exception as e:
                logging.error(f"❌ Error stacking {class_dir.name}: {e}")
            continue

        # Case 2️⃣: 3D dataset (patient/study subfolders)
        for study_dir in tqdm(subdirs, desc=f"{dataset_name}/{class_name} (3D Studies)"):
            slice_paths = list(study_dir.glob("*.png")) + list(study_dir.glob("*.jpg"))
            if not slice_paths:
                continue
            try:
                slices = []
                for sp in sorted(slice_paths):
                    img = Image.open(sp).convert("L")
                    arr = np.array(img, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    arr = resize_image(arr, TARGET_SIZE)
                    slices.append(arr)

                if len(slices) == 0:
                    logging.warning(f"⚠️ No valid slices in {study_dir.name}")
                    continue

                volume = np.stack(slices, axis=0)
                volume = normalize_hu(volume, HU_MIN, HU_MAX)
                out_path = out_dir / f"{class_name}_{study_dir.name}.nii.gz"
                save_nifti(volume, out_path)

                qc_data.append({
                    "dataset": dataset_name,
                    "class": class_name,
                    "file": study_dir.name,
                    "num_slices": len(slices),
                    "shape": list(volume.shape),
                    "mean": float(np.mean(volume)),
                    "std": float(np.std(volume)),
                })
                logging.info(f"✅ Saved 3D study: {out_path.name}")

            except Exception as e:
                logging.error(f"❌ Error converting {study_dir.name}: {e}")

    return qc_data

# ------------------------------------------------------------
# 5️⃣ Main Execution
# ------------------------------------------------------------
def run_conversion():
    all_qc = []
    for ds_name, ds_info in DATASETS.items():
        raw_path = RAW_PATHS.get(ds_name)
        if raw_path and Path(raw_path).exists():
            qc = convert_dataset(ds_name, raw_path)
            all_qc.extend(qc)
        else:
            logging.warning(f"⚠️ Dataset path missing or invalid: {raw_path}")

    qc_report_path = OUT_ROOT / "png_to_nifti_qc.json"
    with open(qc_report_path, "w") as f:
        json.dump(all_qc, f, indent=2)

    logging.info(f"✅ Conversion complete. QC saved to {qc_report_path}")

# ------------------------------------------------------------
# 6️⃣ Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    run_conversion()
