#!/usr/bin/env python3
"""
Final robust enhanced manifest builder for M6 -> M8 (test-split aware).

Behavior:
 - Uses the M5 CSV (rows correspond exactly to the test split ordering)
 - Scans ONLY dataset_root/test/** for .nii or .nii.gz (matching your CT3D dataloader)
 - Matches each test filename stem to an M7 study key (pattern: test_{CLASS}_{STUDYNAME})
 - Extracts all 2D CAM entries (cam_png, cam_npy, slice_index) from the M7 manifest
 - Writes outputs:
     - <out_dir>/fused_manifest_enhanced.csv
     - <out_dir>/fused_manifest_enhanced.json
     - <out_dir>/enhanced_manifest_debug.txt

Defaults tuned to your repo:
 - default m7_manifest -> /mnt/data/manifest_all.json (the uploaded file)
 - default dataset_root -> "d:/xai-ct-project - Copy/data/ct3d_dataset"
 - default m5_csv -> "d:/xai-ct-project - Copy/outputs/m5/eval/test_predictions.csv"
 - default m6_manifest -> "d:/xai-ct-project - Copy/outputs/m6/fused_manifest.csv"
 - default out_dir -> "d:/xai-ct-project - Copy/outputs/m6/results"

Usage example (PowerShell):
    python utils_scripts/build_enhanced_manifest.py `
        --m6_manifest "d:/xai-ct-project - Copy/outputs/m6/fused_manifest.csv" `
        --m5_csv "d:/xai-ct-project - Copy/outputs/m5/eval/test_predictions.csv" `
        --m7_manifest "/mnt/data/manifest_all.json" `
        --dataset_root "d:/xai-ct-project - Copy/data/ct3d_dataset" `
        --out_dir "d:/xai-ct-project - Copy/outputs/m6/results"

"""
from pathlib import Path
import argparse, json, glob, os, pprint, difflib, sys
import pandas as pd

DEFAULT_M7_MANIFEST = "/mnt/data/manifest_all.json"  # uploaded manifest_all.json

def find_test_nii_files(dataset_root: Path):
    """Find .nii / .nii.gz only under the test split and return deterministic list sorted by (class, filename)."""
    test_root = dataset_root / "test"
    if not test_root.exists():
        raise RuntimeError(f"Test split folder not found: {test_root}")
    classes = sorted([p.name for p in test_root.iterdir() if p.is_dir()])
    files = []
    for cls in classes:
        cls_dir = test_root / cls
        # pattern: all nii/nigz files under class (non-recursive except nested allowed)
        patterns = [str(cls_dir / "**" / "*.nii"), str(cls_dir / "**" / "*.nii.gz")]
        found = []
        for patt in patterns:
            found += glob.glob(patt, recursive=True)
        found = [Path(f) for f in sorted(found, key=lambda p: tuple(Path(p).parts))]
        # keep deterministic order per-class
        files.extend(found)
    return files

def load_m7_manifest(path: Path):
    if not path.exists():
        print(f"WARNING: m7_manifest not found at {path}", file=sys.stderr)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = json.loads(path.read_text(encoding="latin1"))
    m7 = {}
    for entry in raw:
        key = entry.get("study") or entry.get("study_uid") or entry.get("id") or entry.get("study_id")
        if not key:
            continue
        m7[key] = entry
    return m7

def stem_of(pathlike: str):
    if not isinstance(pathlike, str) or pathlike.strip() == "":
        return None
    p = pathlike.replace("\\", "/").strip()
    return Path(p).stem

def best_m7_for_stem(stem: str, m7_keys):
    """Try direct, suffix, substring then fuzzy match (case-insensitive)."""
    if stem is None:
        return None
    s_low = stem.lower()
    # exact match or endswith '_stem' or contains '_stem'
    for k in m7_keys:
        kl = k.lower()
        if kl == s_low or kl.endswith("_" + s_low) or ("_" + s_low + "_") in kl or s_low in kl:
            return k
    # fuzzy as fallback
    matches = difflib.get_close_matches(stem, m7_keys, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m6_manifest", type=str, default=r"d:/xai-ct-project - Copy/outputs/m6/fused_manifest.csv")
    p.add_argument("--m5_csv", type=str, default=r"d:/xai-ct-project - Copy/outputs/m5/eval/test_predictions.csv")
    p.add_argument("--m7_manifest", type=str, default=DEFAULT_M7_MANIFEST)
    p.add_argument("--dataset_root", type=str, default=r"d:/xai-ct-project - Copy/data/ct3d_dataset")
    p.add_argument("--out_dir", type=str, default=r"d:/xai-ct-project - Copy/outputs/m6/results")
    args = p.parse_args()

    m6_manifest = Path(args.m6_manifest)
    m5_csv = Path(args.m5_csv)
    m7_manifest = Path(args.m7_manifest)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dbg = []
    dbg.append(f"m6_manifest: {m6_manifest}")
    dbg.append(f"m5_csv: {m5_csv}")
    dbg.append(f"m7_manifest: {m7_manifest}")
    dbg.append(f"dataset_root: {dataset_root.resolve()}")
    dbg.append(f"out_dir: {out_dir.resolve()}")

    if not m6_manifest.exists():
        print("ERROR: m6_manifest not found.", file=sys.stderr); sys.exit(1)
    df_m6 = pd.read_csv(m6_manifest)
    if "id" not in df_m6.columns:
        df_m6.insert(0, "id", df_m6.index.astype(str))
    df_m6["id"] = df_m6["id"].astype(str)
    dbg.append(f"Loaded m6 rows: {len(df_m6)}")

    if not m5_csv.exists():
        print("ERROR: m5_csv not found.", file=sys.stderr); sys.exit(1)
    df_m5 = pd.read_csv(m5_csv, dtype=str)
    dbg.append(f"Loaded m5 rows: {len(df_m5)} columns: {list(df_m5.columns)}")

    # Discover test files in deterministic order (class-sorted, filename-sorted)
    test_files = find_test_nii_files(dataset_root)
    dbg.append(f"Found {len(test_files)} test .nii/.nii.gz files (test split).")
    if len(test_files) < len(df_m5):
        dbg.append("Warning: discovered test files < m5 rows; check dataset_root/test contents.")
    # derive stems (same order as dataloader)
    stems = [p.stem for p in test_files]

    # id order: prefer 'id' column if present in df_m5 else use row indices
    if "id" in df_m5.columns:
        id_order = df_m5["id"].astype(str).tolist()
    else:
        id_order = [str(i) for i in range(len(df_m5))]

    # Load m7 manifest mapping (uploaded)
    m7_map = load_m7_manifest(m7_manifest)
    m7_keys = list(m7_map.keys())
    dbg.append(f"Loaded M7 manifest entries: {len(m7_keys)}")

    # Now map: we expect len(stems) >= len(m5 rows) and test_files includes only test-set files
    mapping = {}
    missing = 0
    # If exact counts equal, map by index (most reliable)
    if len(stems) >= len(id_order):
        dbg.append("Mapping by index order: stems list used in dataloader order -> m5 row order.")
        for i, sid in enumerate(id_order):
            stem = stems[i] if i < len(stems) else None
            chosen = best_m7_for_stem(stem, m7_keys)
            mapping[str(sid)] = chosen
            if chosen is None:
                missing += 1
    else:
        # fewer stems found than m5 rows — try best-effort per-row lookup
        dbg.append("Less stems than m5 rows; performing best-effort matching per row.")
        for idx, sid in enumerate(id_order):
            # try scanning df_m5 row for any path-like value
            chosen = None
            for col in df_m5.columns:
                val = str(df_m5.iloc[idx][col])
                if val and ("/" in val or "\\" in val or ".nii" in val.lower()):
                    chosen = best_m7_for_stem(Path(val).stem, m7_keys)
                    if chosen:
                        break
            # fallback to stem list by index if exists
            if not chosen and idx < len(stems):
                chosen = best_m7_for_stem(stems[idx], m7_keys)
            mapping[str(sid)] = chosen
            if chosen is None:
                missing += 1

    # Build enhanced manifest with CAM lists
    enhanced_rows = []
    for _, r in df_m6.iterrows():
        fid = str(r["id"])
        label = int(r["label"]) if "label" in r and str(r["label"]).strip() != "" else None
        chosen_key = mapping.get(fid)
        cam_pngs, cam_npys, slice_idxs = [], [], []
        if chosen_key and chosen_key in m7_map:
            entry = m7_map[chosen_key]
            two = entry.get("2d") or entry.get("2d_slices") or entry.get("slices2d") or []
            for s in two:
                cam_pngs.append(s.get("cam_png"))
                cam_npys.append(s.get("cam_npy"))
                slice_idxs.append(s.get("slice_index"))
        else:
            # leave empty lists if no mapping
            pass
        enhanced_rows.append({
            "id": fid,
            "label": label,
            "m5_stem": (stems[int(fid)] if int(fid) < len(stems) else None),
            "m7_study_key": chosen_key,
            "cam_pngs": json.dumps(cam_pngs, ensure_ascii=False),
            "cam_npys": json.dumps(cam_npys, ensure_ascii=False),
            "slice_indices": json.dumps(slice_idxs)
        })

    out_csv = out_dir / "fused_manifest_enhanced.csv"
    out_json = out_dir / "fused_manifest_enhanced.json"
    pd.DataFrame(enhanced_rows).to_csv(out_csv, index=False)
    pd.DataFrame(enhanced_rows).to_json(out_json, orient="records", indent=2, force_ascii=False)

    dbg_path = out_dir / "enhanced_manifest_debug.txt"
    with open(dbg_path, "w", encoding="utf-8") as f:
        f.write("DEBUG LOG\n===========\n")
        for line in dbg:
            f.write(line + "\n")
        f.write("\nSample mapping (first 60):\n")
        sample = list(mapping.items())[:60]
        pprint.pprint(sample, stream=f)
        f.write(f"\nMissing (no m7 match) count: {missing}\n")

    print("Wrote enhanced manifest:", out_csv)
    print("Wrote JSON copy:", out_json)
    print("Debug log at:", dbg_path)
    print(f"Missing M7 matches: {missing}.")
    if missing > 0:
        print("If missing > 0, inspect debug log and ensure manifest_all.json and dataset test split match exactly.")

if __name__ == "__main__":
    main()
