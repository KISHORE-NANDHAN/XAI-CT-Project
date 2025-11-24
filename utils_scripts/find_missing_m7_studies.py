#!/usr/bin/env python3
import json
from pathlib import Path

# EDIT THESE PATHS IF NEEDED
TEST_ROOT = Path(r"d:/xai-ct-project - Copy/data/ct3d_dataset/test")
M7_MANIFEST = Path(r"d:/xai-ct-project - Copy/outputs/m7/ct3d_dataset/attention_heatmaps/manifest_all.json")
OUT_PATH = Path(r"d:/xai-ct-project - Copy/outputs/m7/missing_studies.txt")

def collect_test_studies():
    studies = []
    for cls in sorted(TEST_ROOT.iterdir()):
        if cls.is_dir():
            for p in cls.rglob("*.nii.gz"):
                studies.append(p.stem)
            for p in cls.rglob("*.nii"):
                studies.append(p.stem)
    return sorted(studies)

def collect_m7_studies():
    with open(M7_MANIFEST, "r") as f:
        data = json.load(f)
    keys = [d["study"] for d in data]
    return sorted(keys)

def extract_study_stem(m7_key):
    # test_CLASS_studyname → studyname
    # remove prefix "test_*_"
    parts = m7_key.split("_", 2)
    if len(parts) < 3:
        return None
    return parts[2]  # study name part

def main():
    test_stems = collect_test_studies()
    print(f"Test studies found: {len(test_stems)}")

    m7_keys = collect_m7_studies()
    print(f"M7 studies in manifest: {len(m7_keys)}")

    m7_stems = set()
    for k in m7_keys:
        stem = extract_study_stem(k)
        if stem:
            m7_stems.add(stem)

    missing = [s for s in test_stems if s not in m7_stems]

    print(f"\n❗ Missing M7 CAM studies: {len(missing)}\n")

    for s in missing[:30]:
        print(" -", s)
    if len(missing) > 30:
        print(" ... (more)")

    OUT_PATH.write_text("\n".join(missing), encoding="utf-8")
    print(f"\nSaved missing study list to:\n{OUT_PATH}")

if __name__ == "__main__":
    main()
