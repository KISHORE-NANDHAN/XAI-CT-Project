#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import pandas as pd

def collect_test_files(test_root):
    """Collect all test .nii/.nii.gz paths with class name."""
    items = []
    for cls in sorted(os.listdir(test_root)):
        cls_dir = os.path.join(test_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        for f in os.listdir(cls_dir):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                stem = f.replace(".nii.gz", "").replace(".nii", "")
                items.append((cls, stem))
    return items

def load_m7_manifest(m7_path):
    with open(m7_path, "r") as f:
        arr = json.load(f)

    m7_map = {}
    for entry in arr:
        study = entry["study"]
        m7_map[study] = entry
    return m7_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m6_manifest", required=True)
    ap.add_argument("--m5_csv", required=True)
    ap.add_argument("--m7_manifest", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load M6 + M5
    df_m6 = pd.read_csv(args.m6_manifest)
    df_m5 = pd.read_csv(args.m5_csv)

    # Fix id column
    if "id" not in df_m6.columns:
        df_m6["id"] = df_m6.index.astype(str)
    df_m6["id"] = df_m6["id"].astype(str)

    # Load test file list
    test_root = Path(args.dataset_root) / "test"
    test_items = collect_test_files(test_root)     # list of (CLASS, STEM)
    if len(test_items) != len(df_m5):
        print(f"WARNING: test dataset has {len(test_items)} volumes but M5 has {len(df_m5)} rows.")

    # Load M7 manifest
    m7_map = load_m7_manifest(args.m7_manifest)
    m7_keys = set(m7_map.keys())

    # Build enhanced manifest
    enhanced = []
    missing = 0

    for idx, row in df_m6.iterrows():
        fid = row["id"]
        label = row["label"]

        # Get corresponding test file info (same order as dataset loader)
        if idx >= len(test_items):
            enhanced.append({
                "id": fid, "label": label,
                "study": None, "cam_pngs": "[]", "cam_npys": "[]", "slice_indices": "[]"
            })
            missing += 1
            continue

        cls, stem = test_items[idx]

        # Reconstruct the exact M7 folder name
        m7_name = f"test_{cls}_{stem}"

        if m7_name not in m7_keys:
            # Try fallback: strip class mismatches
            # (rare edge case)
            fallback = None
            for key in m7_keys:
                if key.endswith(stem):
                    fallback = key
                    break
            if fallback:
                m7_name = fallback
            else:
                enhanced.append({
                    "id": fid, "label": label,
                    "study": m7_name,
                    "cam_pngs": "[]", "cam_npys": "[]", "slice_indices": "[]"
                })
                missing += 1
                continue

        entry = m7_map[m7_name]
        slices2d = entry.get("2d", [])

        cam_pngs = [s["cam_png"] for s in slices2d]
        cam_npys = [s["cam_npy"] for s in slices2d]
        slice_idxs = [s["slice_index"] for s in slices2d]

        enhanced.append({
            "id": fid,
            "label": label,
            "study": m7_name,
            "cam_pngs": json.dumps(cam_pngs),
            "cam_npys": json.dumps(cam_npys),
            "slice_indices": json.dumps(slice_idxs)
        })

    # Write outputs
    df_out = pd.DataFrame(enhanced)
    df_out.to_csv(out_dir / "fused_manifest_enhanced.csv", index=False)
    df_out.to_json(out_dir / "fused_manifest_enhanced.json", orient="records", indent=2)

    print("\n===================================")
    print(f"Enhanced manifest written to: {out_dir}")
    print(f"Missing M7 studies: {missing}")
    print("===================================")

if __name__ == "__main__":
    main()
