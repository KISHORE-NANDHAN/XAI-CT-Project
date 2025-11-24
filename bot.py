# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("Number of CUDA devices:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# import numpy as np
# import nibabel as nib


# v = np.load("data/mil_prepared/train/COVIDx_COVID/embeddings.npy")
# y = np.load("data/mil/COVIDx/train/COVID/embeddings.npy")

# img = nib.load("data/raw_nifti/COVID-CT-MD/CAP/CAP_volume.nii.gz")
# x = img.get_fdata()

# print(v.shape)
# print(x.shape)
# print(y.shape)

# import glob, os
# for cls in ["COVID", "NORMAL", "PNEUMONIA", "CAP", "nonCOVID"]:
#     count = len(glob.glob(f"data/ct3d_dataset_raw/{cls}/*.nii.gz"))
#     print(f"{cls}: {count}")

# from collections import Counter
# from scripts.m5.dataset_loader_3d import CT3DDataset
# root="data/ct3d_dataset"
# for split in ["train","val","test"]:
#     ds = CT3DDataset(root, split=split)
#     print(split, "counts:", dict(Counter(ds.labels)))


# import torch

# ckpt_path = r"d:/xai-ct-project - Copy/outputs/checkpoints/best_model.pt"
# ckpt = torch.load(ckpt_path, map_location="cpu")

# print(type(ckpt))
# if isinstance(ckpt, dict):
#     print("Keys:", ckpt.keys())


import numpy as np

cam = np.load("d:/xai-ct-project - Copy/outputs/m6/fused_embeddings.npy")
print(cam.shape, cam.dtype)
#print(cam)

vol = np.load("outputs/m7/preprocessed/attention_heatmaps/COVID-CT-MD/2d/slice_000_cam.npy")
print(vol.shape)

# import os

# root = r"d:/xai-ct-project - Copy/data/preprocessed_ct3d_flat"

# for split in ["train", "val", "test"]:
#     d = os.path.join(root, split)
#     print("\n=== Checking:", d, "===")
#     for cls in ["CAP", "COVID", "NORMAL", "PNEUMONIA"]:
#         cdir = os.path.join(d, cls)
#         print(" > Class:", cls)
#         if not os.path.exists(cdir):
#             print("    ❌ Folder missing")
#             continue

#         files = [f for f in os.listdir(cdir)
#                  if f.lower().endswith((".nii", ".nii.gz", ".npy"))]

#         if not files:
#             print("    ❌ NO volume files found")
# #         else:
# #             for f in files[:10]:  # print up to 10
# #                 print("    ✔", f)

# # import numpy as np
# # import nibabel as nib
# # import os

# # path = r"d:/xai-ct-project - Copy/data/ct3d_dataset/train/CAP/cap008.nii.gz"

# # nii = nib.load(path)
# # vol = nii.get_fdata()

# # print("File:", path)
# # print("Shape:", vol.shape)
# # print("Min/Max:", np.min(vol), np.max(vol))
# import os

# test_root = r"d:/xai-ct-project - Copy/data/ct3d_dataset/test"

# count = 0
# studies = []

# for cls in sorted(os.listdir(test_root)):
#     cls_path = os.path.join(test_root, cls)
#     if not os.path.isdir(cls_path):
#         continue

#     for root, dirs, files in os.walk(cls_path):
#         for f in files:
#             if f.endswith(".nii") or f.endswith(".nii.gz"):
#                 count += 1
#                 studies.append(os.path.splitext(f)[0])

# print("TOTAL TEST STUDIES:", count)
# print("FIRST 20:", studies[:20])
