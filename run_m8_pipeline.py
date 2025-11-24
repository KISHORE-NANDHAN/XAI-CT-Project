#!/usr/bin/env python3
"""
run_m8_pipeline.py  (PATCHED VERSION)

Fixes:
- Uses enhanced fused manifest for prototype tiles
- Uses correct CAM root & image root paths
- More robust YAML reads
"""

import subprocess
import yaml
from pathlib import Path
import sys

CONFIG_PATH = Path("config/m8_config.yaml")
PYTHON = sys.executable

def run(cmd):
    print("\n=======================================================")
    print(">>> Running:", " ".join(cmd))
    print("=======================================================\n")
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print("\n❌ ERROR: Command failed:", " ".join(cmd))
        sys.exit(1)
    print("\n✔ DONE:", " ".join(cmd))


def main():
    # Load config
    if not CONFIG_PATH.exists():
        print(f"❌ Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    print("\n🚀 Starting M8 Pipeline using config:", CONFIG_PATH)

    # ---------------------------------------------------------
    # 1. Prepare SHAP background
    # ---------------------------------------------------------
    run([
        PYTHON, "shap/m8_prepare_background.py",
        "--embeddings", cfg["m6"]["embeddings"],
        "--manifest",   cfg["m6"]["manifest"],
        "--out_dir",    cfg["shap"]["out_dir"],
        "--method",     cfg["params"]["shap"].get("background_method", "stratified"),
        "--n_samples",  str(cfg["params"]["shap"].get("background_n", 50)),
        "--seed",       "42"
    ])

    # ---------------------------------------------------------
    # 2. Compute SHAP
    # ---------------------------------------------------------
    run([
        PYTHON, "shap/m8_compute_shap.py",
        "--embeddings", cfg["m6"]["embeddings"],
        "--manifest",   cfg["m6"]["manifest"],
        "--model",      cfg["m6"]["fusion_head"],
        "--background", cfg["shap"]["background_samples"],
        "--out_dir",    cfg["shap"]["out_dir"],
        "--method",     cfg["params"]["shap"]["method"],
        "--nsamples",   str(cfg["params"]["shap"]["kernel_nsamples"]),
        "--subset",     str(cfg["params"]["shap"]["subset_for_quick"])
    ])

    # ---------------------------------------------------------
    # 3. Extract prototypes
    # ---------------------------------------------------------
    run([
        PYTHON, "shap/m8_prototype_extractor.py",
        "--embeddings", cfg["m6"]["embeddings"],
        "--manifest",   cfg["m6"]["manifest"],
        "--out_dir",    cfg["prototypes"]["out_dir"],
        "--per_class",  str(cfg["params"]["prototypes"]["per_class"]),
        "--method",     cfg["params"]["prototypes"]["method"],
        "--seed",       "42"
    ])

    # ---------------------------------------------------------
    # 4. Generate prototype tiles  (PATCHED)
    # ---------------------------------------------------------
    enhanced_manifest = cfg["m6"].get("enhanced_manifest")
    if enhanced_manifest is None:
        print("❌ ERROR: You must set m6.enhanced_manifest in config/m8_config.yaml")
        sys.exit(1)

    run([
        PYTHON, "shap/m8_prototype_tiles.py",
        "--prototypes", cfg["prototypes"]["vectors"],
        "--metadata",   cfg["prototypes"]["metadata"],
        "--embeddings", cfg["m6"]["embeddings"],
        "--manifest",   enhanced_manifest,     # ✔ FIXED: use enhanced manifest
        "--img_root",   cfg["params"]["paths"]["img_root"],
        "--cam_root",   cfg["params"]["paths"]["cam_root"],
        "--out_dir",    cfg["prototypes"]["tiles_dir"],
        "--k_nearest",  str(cfg["params"]["prototypes"]["k_nearest"])
    ])

    # ---------------------------------------------------------
    # 5. Generate Final M8 Report
    # ---------------------------------------------------------
    run([
        PYTHON, "shap/m8_generate_report.py",
        "--config", str(CONFIG_PATH)
    ])

    print("\n🎉 M8 Pipeline Completed Successfully!")
    print("📄 Report available at:", cfg["m8"]["report_pdf"])
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
