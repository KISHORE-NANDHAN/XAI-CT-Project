#!/usr/bin/env python3
"""
flatten_preprocessed_ct3d.py

Purpose:
--------
Flatten CT3D preprocessed dataset so that each study becomes a single folder.

Input structure (current):
    root/
        train/
            CAP/
                cap008/
                    volume.npy
                    slices/*.png
        val/
        test/

Output structure (flattened):
    out_root/
        train_CAP_cap008/
            volume.npy
            slices/*.png
        val_COVID_case01/
        test_NORMAL_xxxxx/
"""

import os
import shutil
import argparse

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser(description="Flatten CT3D preprocessed folders")
    p.add_argument("--root", type=str, required=True, help="Input preprocessed_ct3d directory")
    p.add_argument("--out_root", type=str, required=True, help="Output flattened directory")
    return p.parse_args()

def main():
    args = parse_args()

    root = args.root
    out_root = args.out_root

    ensure_dir(out_root)

    print(f"[main] Flattening dataset from:\n   {root}\nto:\n   {out_root}\n")

    # expects: root/<split>/<class>/<case>/
    splits = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    if not splits:
        print("❌ No valid split folders (train/val/test) found in:", root)
        return

    for split in splits:
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            continue

        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        print(f"[split] {split}: classes = {classes}")

        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            cases = [d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))]

            for case in cases:
                case_path = os.path.join(cls_path, case)

                new_name = f"{split}_{cls}_{case}".replace(" ", "_")
                out_case = os.path.join(out_root, new_name)
                ensure_dir(out_case)

                # copy all contents
                print(f"[copy] {case_path} → {out_case}")
                shutil.rmtree(out_case, ignore_errors=True)
                shutil.copytree(case_path, out_case)

    print("\n✅ Flatten complete!")
    print("Use this path for M7:")
    print(f"   --preprocessed_root \"{out_root}\"")


if __name__ == "__main__":
    main()
