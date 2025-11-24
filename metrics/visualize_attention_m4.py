# ============================================================
# File: metrics/visualize_attention_m4.py
# Description: Auto-detect and visualize top-k attended CT slices (M4)
# ============================================================

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


def visualize_attention(results_json, preprocessed_root, save_root, top_k=5):
    results = json.loads(Path(results_json).read_text())
    preprocessed_root = Path(preprocessed_root)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Loaded {len(results)} study attention results")

    for r in tqdm(results, desc="Rendering"):
        study_uid = r["study_uid"]
        label = r.get("label", -1)
        topk_indices = r["topk_indices"]
        scores = r["topk_scores"]

        # --- Try exact subfolder match ---
        found_paths = list(preprocessed_root.rglob(f"{study_uid}"))
        if not found_paths:
            # Try to match partial folder names (COVID, CAP, etc.)
            found_paths = list(preprocessed_root.rglob(f"*{study_uid}*"))

        if not found_paths:
            print(f"⚠️ Could not locate study folder for {study_uid}")
            continue

        # Pick the first valid path containing PNG slices
        study_path = None
        for path in found_paths:
            if path.is_dir() and list(path.glob("*.png")):
                study_path = path
                break

        if not study_path:
            print(f"⚠️ No PNG slices found for {study_uid}")
            continue

        # --- Load slices ---
        slice_paths = sorted(study_path.glob("*.png"))
        if not slice_paths:
            print(f"⚠️ No slices found for {study_uid} → {study_path}")
            continue

        # --- Select top-k slices ---
        topk_indices = [i for i in topk_indices if i < len(slice_paths)]
        topk_slices = [slice_paths[i] for i in topk_indices]
        topk_scores = [scores[i] if i < len(scores) else 0 for i in range(len(topk_indices))]

        # --- Plot ---
        ncols = len(topk_slices)
        fig, axs = plt.subplots(1, ncols, figsize=(3 * ncols, 3))
        if ncols == 1:
            axs = [axs]

        for ax, path, sc in zip(axs, topk_slices, topk_scores):
            img = np.array(Image.open(path).convert("L"))
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{Path(path).stem}\nScore={sc:.3f}", fontsize=8)
            ax.axis("off")

        plt.suptitle(f"{study_uid} (Label={label})", fontsize=10)
        out_path = save_root / f"{study_uid}_topk.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"\n✅ Saved all visualizations → {save_root}")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize top-k attended CT slices per study (M4)")
    parser.add_argument("--results", required=True, help="Path to attention_results.json")
    parser.add_argument("--preprocessed_root", required=True, help="Path to preprocessed dataset root")
    parser.add_argument("--out_root", default="outputs/m4/attention_vis", help="Output directory for results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top slices to visualize")
    args = parser.parse_args()

    visualize_attention(args.results, args.preprocessed_root, args.out_root, args.top_k)
