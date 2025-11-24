# ============================================================
# File: scripts/preprocess.py
# XAI-CT PROJECT — M2: Preprocessing & Harmonization
# ============================================================
# Standardizes voxel spacing, applies HU windowing,
# generates lung masks (optional), performs QC, and saves
# preprocessed 3D CT volumes for downstream training.
# ============================================================

import os
import json
import yaml
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------
# 1️⃣ Load Configuration
# ------------------------------------------------------------
CFG_PATH = Path("config/preprocess.yaml")
cfg = yaml.safe_load(open(CFG_PATH))

TARGET_SPACING = cfg["preprocess"]["voxel_spacing"]
HU_MIN, HU_MAX = cfg["preprocess"]["hu_window"]
RAW_PATHS = cfg["paths"]["raw_nifti_data"]
OUT_PATH = Path(cfg["paths"]["curated_data"])
MASK_PATH = Path(cfg["paths"]["masks"])
LOG_PATH = Path(cfg["logging"]["file"])

# Create directories
OUT_PATH.mkdir(parents=True, exist_ok=True)
MASK_PATH.mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2️⃣ Logging Setup
# ------------------------------------------------------------
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

logging.info("=== XAI-CT Preprocessing Pipeline Initialized ===")

# ------------------------------------------------------------
# 3️⃣ Helper Functions
# ------------------------------------------------------------
def resample_volume(img, new_spacing):
    """Resample a SimpleITK image to new voxel spacing."""
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    return resample.Execute(img)

def window_hu(arr, hu_min, hu_max):
    """Clip HU values and normalize to [0, 1]."""
    arr = np.clip(arr, hu_min, hu_max)
    return (arr - hu_min) / (hu_max - hu_min)

def qc_metrics(arr, spacing, dataset, file_name):
    """Compute QC statistics for the volume."""
    if arr.size == 0:
        return {
            "dataset": dataset,
            "file_name": file_name,
            "status": "skipped_empty"
        }
    return {
        "dataset": dataset,
        "file_name": file_name,
        "mean_intensity": float(np.mean(arr)),
        "std_intensity": float(np.std(arr)),
        "min_intensity": float(np.min(arr)),
        "max_intensity": float(np.max(arr)),
        "voxel_spacing": [float(s) for s in spacing],
        "shape": arr.shape,
        "status": "processed"
    }

def generate_mask(volume):
    """Generate lung mask placeholder (U-Net or fallback)."""
    if not cfg["preprocess"]["mask_generation"]["enable"]:
        return None

    method = cfg["preprocess"]["mask_generation"]["method"]
    if method == "otsu":
        arr = sitk.GetArrayFromImage(volume)
        otsu = sitk.OtsuThreshold(sitk.GetImageFromArray(arr.astype(np.float32)))
        mask = sitk.GetArrayFromImage(otsu)
        return mask
    elif method == "unet_pretrained":
        logging.warning("U-Net segmentation not yet implemented; skipping mask.")
        return None
    else:
        logging.warning(f"Unknown mask method: {method}")
        return None

# ------------------------------------------------------------
# 4️⃣ Dataset Preprocessing
# ------------------------------------------------------------
def preprocess_dataset(dataset_name, dataset_path):
    dataset_path = Path(dataset_path)
    logging.info(f"Scanning dataset path: {dataset_path}")

    all_vols = sorted(dataset_path.rglob("*.nii*"))
    qc_results = []

    if not all_vols:
        logging.error(f"No NIfTI files found in {dataset_path}. Skipping dataset.")
        return qc_results

    for f in tqdm(all_vols, desc=f"Preprocessing {dataset_name}"):
        try:
            vol = sitk.ReadImage(str(f))
            arr = sitk.GetArrayFromImage(vol)

            # Skip empty/invalid volumes
            if arr.size == 0 or np.all(arr == 0):
                logging.warning(f"⚠️ Skipping empty or invalid volume: {f.name}")
                continue

            # Resample to target spacing
            resampled = resample_volume(vol, TARGET_SPACING)
            arr = sitk.GetArrayFromImage(resampled)
            if arr.size == 0 or np.all(arr == 0):
                logging.warning(f"⚠️ Skipping invalid resampled array: {f.name}")
                continue

            # Apply HU windowing
            windowed = window_hu(arr, HU_MIN, HU_MAX)

            # Optional mask generation
            mask = generate_mask(resampled)
            if mask is not None:
                np.save(MASK_PATH / f"{dataset_name}_{f.stem}_mask.npy", mask)

            # Save preprocessed volume
            np.save(OUT_PATH / f"{dataset_name}_{f.stem}.npy", windowed)

            qc = qc_metrics(windowed, TARGET_SPACING, dataset_name, f.name)
            qc_results.append(qc)
            logging.info(f"✅ [{dataset_name}] Saved: {f.stem}.npy")

        except Exception as e:
            logging.error(f"❌ [{dataset_name}] Error processing {f.name}: {e}")
            continue

    return qc_results

# ------------------------------------------------------------
# 5️⃣ Run All Datasets
# ------------------------------------------------------------
def run_preprocessing_all():
    all_qc_results = []

    for ds_name, ds_path in RAW_PATHS.items():
        if not Path(ds_path).exists():
            logging.warning(f"⚠️ Dataset path missing or invalid: {ds_path}")
            continue

        qc = preprocess_dataset(ds_name, ds_path)
        all_qc_results.extend(qc)

    # Save combined QC report
    qc_report_path = Path(cfg["paths"]["qc_reports"]) / "preprocess_qc_all.json"
    qc_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qc_report_path, "w") as f:
        json.dump(all_qc_results, f, indent=2)

    logging.info(f"✅ All dataset QC saved to {qc_report_path}")
    logging.info("=== All Dataset Preprocessing Completed ===")

# ------------------------------------------------------------
# 6️⃣ Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing_all()
