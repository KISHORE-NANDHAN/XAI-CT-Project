# ============================================
# XAI-CT PROJECT DATA PREPROCESSING
# ============================================

import os, json, cv2, yaml, pydicom, nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = Path("config/preprocess.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIRS = cfg["paths"]["raw_data"]
CURATED_DIR = Path(cfg["paths"]["curated_data"])
LOG_DIR = Path(cfg["paths"]["logs"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    filename=LOG_DIR / "preprocessing_all_datasets.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
if cfg["logging"].get("console", True):
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

logging.info("🚀 Starting preprocessing pipeline for all datasets")

# -----------------------------
# Image loaders
# -----------------------------
def load_ct_image(path):
    ext = path.suffix.lower()
    try:
        if ext in [".png", ".jpg", ".jpeg"]:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        elif ext == ".dcm":
            ds = pydicom.dcmread(str(path), force=True)
            img = ds.pixel_array.astype(np.float32)
        elif ext in [".nii", ".nii.gz"]:
            img = nib.load(str(path)).get_fdata()
            if img.ndim == 3:
                img = np.mean(img, axis=2)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        img = None
    return img

# -----------------------------
# Preprocessing functions
# -----------------------------
def apply_hu_window(img, hu_min=-1000, hu_max=400):
    """
    Clip CT image to HU window and scale to 0-255 safely for Python 3.12.
    """
    # Ensure float computation to prevent integer overflow
    img = img.astype(np.float32)

    # Clip values to the HU window
    img = np.clip(img, hu_min, hu_max)

    # Scale to 0-255
    img = (img - hu_min) / (hu_max - hu_min) * 255.0

    return img.astype(np.uint8)



def preprocess_image(img, size=(224, 224)):
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(img.astype(np.uint8))

def denoise(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

# -----------------------------
# Process single image
# -----------------------------
def process_image(img_path, label, dataset_name, out_dir):
    img = load_ct_image(img_path)
    if img is None:
        return None

    # Apply HU window first if specified
    if cfg["preprocess"].get("hu_window", None):
        img = apply_hu_window(img, *cfg["preprocess"]["hu_window"])

    # Convert to grayscale if needed
    if cfg["preprocess"].get("convert_to_grayscale", True) and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise if requested
    if cfg["preprocess"].get("remove_noise", False):
        img = denoise(img)

    # Resize and normalize
    img = preprocess_image(img, tuple(cfg["preprocess"]["image_size"]))

    # Save processed image
    out_path = out_dir / label / img_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {"file_path": str(out_path), "label": label, "dataset": dataset_name}


# -----------------------------
# Process a dataset
# -----------------------------
def process_dataset(dataset_name):
    raw_dir = Path(RAW_DIRS[dataset_name])
    out_dir = CURATED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = cfg["datasets"]["available"][dataset_name]
    labels = dataset_info["classes"]

    # Dynamically detect any additional folders
    existing_labels = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    labels_to_process = [l for l in labels if l in existing_labels]

    metadata = []
    for label in labels_to_process:
        img_paths = list((raw_dir / label).glob("*"))
        logging.info(f"{dataset_name} - {label}: {len(img_paths)} images")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(lambda p: process_image(p, label, dataset_name, out_dir), img_paths),
                                total=len(img_paths), desc=f"{dataset_name} - {label}"))

        for record in results:
            if record:
                metadata.append(record)

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"✅ Metadata saved for {dataset_name}: {meta_path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    for dataset in cfg["datasets"]["available"]:
        logging.info(f"💾 Processing dataset: {dataset}")
        process_dataset(dataset)
    logging.info("🏁 All datasets processed successfully")
