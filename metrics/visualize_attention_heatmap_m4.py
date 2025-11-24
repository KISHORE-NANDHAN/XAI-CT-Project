# ============================================================
# File: metrics/visualize_attention_heatmap_m4.py
# Description: Overlay Transformer attention on top-K CT slices (M4)
# ============================================================

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def apply_heatmap(img_gray, attn_value, cmap='jet', alpha=0.5):
    """
    Apply attention heatmap on top of grayscale image.
    attn_value: float in [0, 1]
    """
    img_gray = np.array(img_gray).astype(np.float32) / 255.0
    img_gray_rgb = np.stack([img_gray] * 3, axis=-1)

    cmap_func = plt.get_cmap(cmap)
    heat_color = cmap_func(attn_value)[:, :3] if isinstance(attn_value, np.ndarray) else cmap_func(attn_value)[:3]

    # if attn_value is scalar, broadcast
    if not isinstance(attn_value, np.ndarray):
        heat_color = np.ones_like(img_gray_rgb) * heat_color

    blended = (1 - alpha) * img_gray_rgb + alpha * heat_color
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


def visualize_attention_heatmaps(results_json, preprocessed_root, save_root, top_k=5):
    results = json.loads(Path(results_json).read_text())
    preprocessed_root = Path(preprocessed_root)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"🔥 Loaded {len(results)} attention results — generating overlays...")

    for r in tqdm(results, desc="Rendering Heatmaps"):
        study_uid = r["study_uid"]
        label = r.get("label", -1)
        topk_indices = r["topk_indices"]
        scores = r["topk_scores"]

                # --- Detect dataset + class ---
        dataset_prefixes = ["COVIDx", "SARS-CoV2", "COVID-CT-MD"]
        dataset = None
        for prefix in dataset_prefixes:
            if study_uid.startswith(prefix):
                dataset = prefix
                break

        # 🔹 Fallback auto-mapping for short names
        if dataset is None:
            if study_uid in ["CAP", "COVID", "non-COVID"]:
                dataset = "COVID-CT-MD"
            elif study_uid in ["NORMAL", "PNEUMONIA"]:
                dataset = "COVIDx"
            else:
                print(f"⚠️ Unknown dataset prefix in {study_uid}")
                continue

        # Extract class name (everything after dataset_)
        cls = study_uid.replace(dataset + "_", "")
        study_path = preprocessed_root / dataset / cls


        if not study_path.exists():
            print(f"⚠️ Study path not found: {study_path}")
            continue

        slice_paths = sorted(study_path.glob("*.png"))
        if not slice_paths:
            print(f"⚠️ No PNG slices in {study_path}")
            continue

        topk_indices = [i for i in topk_indices if i < len(slice_paths)]
        norm_scores = np.array(scores[:len(topk_indices)])
        if len(norm_scores) > 0:
            norm_scores = (norm_scores - norm_scores.min()) / (norm_scores.max() - norm_scores.min() + 1e-8)

        for i, idx in enumerate(topk_indices):
            img_path = slice_paths[idx]
            score = norm_scores[i] if i < len(norm_scores) else 0.0

            img_gray = Image.open(img_path).convert("L")
            overlay = apply_heatmap(img_gray, score, cmap="jet", alpha=0.5)

            out_dir = save_root / study_uid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(img_path).stem}_overlay.png"

            Image.fromarray(overlay).save(out_path)

    print(f"\n✅ Saved all attention heatmaps → {save_root}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps per study (M4)")
    parser.add_argument("--results", required=True, help="Path to attention_results.json")
    parser.add_argument("--preprocessed_root", required=True, help="Path to preprocessed dataset root")
    parser.add_argument("--out_root", default="outputs/m4/attention_heatmaps", help="Output directory for overlays")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top slices to visualize")
    args = parser.parse_args()

    visualize_attention_heatmaps(args.results, args.preprocessed_root, args.out_root, args.top_k)
