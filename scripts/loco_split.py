# ============================================
# Dataset Card + Split Generator for all datasets
# ============================================

import json
import os
import yaml
import logging
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np

# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
CONFIG_PATH = Path("config/preprocess.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATASETS = CONFIG["datasets"]["available"].keys()
CURATED_ROOT = Path(CONFIG["paths"]["curated_data"])
CARD_OUT_ROOT = Path(CONFIG["paths"]["dataset_card"]).parent
FIGURE_DIR = Path(CONFIG["paths"]["figures"])
LOG_DIR = Path(CONFIG["paths"]["logs"])
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CARD_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Setup logging
LOG_FILE = LOG_DIR / "dataset_processing.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------
# Helper: generate dataset card
# ---------------------------------------------------------
def generate_dataset_card(dataset_name):
    CURATED_DIR = CURATED_ROOT / dataset_name
    META_FILE = CURATED_DIR / CONFIG["metadata"]["output_file"]
    CARD_OUT = CARD_OUT_ROOT / f"{dataset_name}_card.json"

    if not META_FILE.exists():
        logging.warning(f"Metadata not found: {META_FILE}")
        return

    meta = json.load(open(META_FILE))
    total = len(meta)
    labels = [m["label"] for m in meta]
    counts = Counter(labels)

    # Optional intensity stats
    intensities = [m["intensity_range"] for m in meta if "intensity_range" in m]
    intensity_stats = {}
    if intensities:
        intensities = np.array(intensities)
        intensity_stats = {
            "min": float(np.min(intensities[:, 0])),
            "max": float(np.max(intensities[:, 1])),
            "mean": float(np.mean(intensities[:, 0])),
            "std": float(np.std(intensities[:, 1]))
        }

    stats = {
        "project": CONFIG["project"]["name"],
        "version": CONFIG["project"]["version"],
        "dataset": dataset_name,
        "total_images": total,
        "class_distribution": dict(counts),
        "image_size": CONFIG["preprocess"]["image_size"],
        "augmentation": CONFIG["preprocess"]["augmentations"],
        "normalization": CONFIG["preprocess"]["normalization"],
        "source": CONFIG["datasets"]["available"][dataset_name]["source"],
        "intensity_stats": intensity_stats
    }

    with open(CARD_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Dataset card saved: {CARD_OUT}")
    print(f"✅ Dataset card saved: {CARD_OUT}")

    # Plot class distribution
    plt.figure(figsize=(6, 6))
    classes = CONFIG["datasets"]["available"][dataset_name]["classes"]
    plt.pie(
        counts.values(),
        labels=[classes[k] if isinstance(k,int) else k for k in counts.keys()],
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.Set3.colors
    )
    plt.title(f"{dataset_name} - Class Distribution")
    plt.tight_layout()
    pie_path = FIGURE_DIR / f"{dataset_name}_class_distribution.png"
    plt.savefig(pie_path, dpi=200)
    plt.close()
    logging.info(f"Class distribution figure saved: {pie_path}")
    print(f"📊 Class distribution figure saved: {pie_path}")


# ---------------------------------------------------------
# Helper: generate train/val/test splits
# ---------------------------------------------------------
def generate_splits(dataset_name):
    CURATED_DIR = CURATED_ROOT / dataset_name
    META_FILE = CURATED_DIR / CONFIG["metadata"]["output_file"]

    if not META_FILE.exists():
        logging.warning(f"Metadata not found: {META_FILE}")
        return

    meta = json.load(open(META_FILE))
    labels = [m["label"] for m in meta]

    if CONFIG["split"]["method"].lower() == "loco":
        logging.info(f"Using LOCO split for {dataset_name}")
        groups = [m.get("site", "site1") for m in meta]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=CONFIG["split"]["test_ratio"],
            random_state=CONFIG["split"]["seed"]
        )
        train_idx, test_idx = next(splitter.split(meta, y=labels, groups=groups))
        train_temp = [meta[i] for i in train_idx]
        test = [meta[i] for i in test_idx]
        val_size = CONFIG["split"]["val_ratio"] / (CONFIG["split"]["train_ratio"] + CONFIG["split"]["val_ratio"])
        train, val = train_test_split(
            train_temp,
            test_size=val_size,
            stratify=[m["label"] for m in train_temp],
            random_state=CONFIG["split"]["seed"]
        )
    else:
        logging.info(f"Using stratified split for {dataset_name}")
        train, temp = train_test_split(
            meta,
            test_size=(CONFIG["split"]["val_ratio"] + CONFIG["split"]["test_ratio"]),
            stratify=labels,
            random_state=CONFIG["split"]["seed"]
        )
        val_size = CONFIG["split"]["test_ratio"] / (CONFIG["split"]["val_ratio"] + CONFIG["split"]["test_ratio"])
        val, test = train_test_split(
            temp,
            test_size=val_size,
            stratify=[m["label"] for m in temp],
            random_state=CONFIG["split"]["seed"]
        )

    # Save manifests
    splits = {"train": train, "val": val, "test": test}
    for split_name, split_data in splits.items():
        split_path = CURATED_DIR / f"{split_name}_manifest.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        logging.info(f"{split_name.capitalize()} manifest created: {split_path}")
        print(f"✅ {split_name.capitalize()} manifest created: {split_path}")


# ---------------------------------------------------------
# Main loop over all datasets
# ---------------------------------------------------------
if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"\n💾 Processing dataset: {dataset}")
        logging.info(f"Processing dataset: {dataset}")

        generate_dataset_card(dataset)
        generate_splits(dataset)

    logging.info("🏁 All datasets processed successfully")
    print("\n🏁 All datasets processed successfully")
