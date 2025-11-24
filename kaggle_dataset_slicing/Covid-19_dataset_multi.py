# ===============================================================
# 📦 COVIDx CT-3A Subset Builder (Multi-Class, Kaggle-Ready)
# ===============================================================
# Run this Kaggle notebook
import os, shutil, random
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------
# 1️⃣ Paths
# ---------------------------------------------------------------
DATASET_PATH = "/kaggle/input/covidxct"
IMAGE_DIR = os.path.join(DATASET_PATH, "3A_images")
OUTPUT_DIR = "/kaggle/working/covidxct_subset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Output directory:", OUTPUT_DIR)

# ---------------------------------------------------------------
# 2️⃣ Load train/val/test lists
# ---------------------------------------------------------------
def load_split(txt_file):
    df = pd.read_csv(
        os.path.join(DATASET_PATH, txt_file),
        sep=" ", header=None,
        names=["filename", "class", "xmin", "ymin", "xmax", "ymax"]
    )
    return df

train_df = load_split("train_COVIDx_CT-3A.txt")
val_df   = load_split("val_COVIDx_CT-3A.txt")
test_df  = load_split("test_COVIDx_CT-3A.txt")

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print("✅ Combined metadata shape:", all_df.shape)
print(all_df.head())

# ---------------------------------------------------------------
# 3️⃣ Split into 3 classes
# ---------------------------------------------------------------
normal_df = all_df[all_df["class"] == 0]
pneumonia_df = all_df[all_df["class"] == 1]
covid_df = all_df[all_df["class"] == 2]

print(f"🫁 Normal: {len(normal_df)} | 🌫️ Pneumonia: {len(pneumonia_df)} | 🦠 COVID: {len(covid_df)}")

# ---------------------------------------------------------------
# 4️⃣ Sample up to 1000 per class (or all if smaller)
# ---------------------------------------------------------------
def sample_df(df, n=1000):
    return df.sample(n=n, random_state=42) if len(df) > n else df

normal_sample = sample_df(normal_df, 1000)
pneumonia_sample = sample_df(pneumonia_df, 1000)
covid_sample = sample_df(covid_df, 1000)

print(f"✅ Sampled {len(normal_sample)} Normal, {len(pneumonia_sample)} Pneumonia, {len(covid_sample)} COVID")

# ---------------------------------------------------------------
# 5️⃣ Copy images
# ---------------------------------------------------------------
def copy_images(df, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    copied = 0
    for fname in df["filename"]:
        src = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(label_dir, fname))
            copied += 1
    return copied

normal_dir = os.path.join(OUTPUT_DIR, "NORMAL")
pneumonia_dir = os.path.join(OUTPUT_DIR, "PNEUMONIA")
covid_dir = os.path.join(OUTPUT_DIR, "COVID")

normal_copied = copy_images(normal_sample, normal_dir)
pneumonia_copied = copy_images(pneumonia_sample, pneumonia_dir)
covid_copied = copy_images(covid_sample, covid_dir)

print(f"✅ Copied: {normal_copied} Normal | {pneumonia_copied} Pneumonia | {covid_copied} COVID images")

# ---------------------------------------------------------------
# 6️⃣ Zip for download
# ---------------------------------------------------------------
subset_zip = shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
print("📦 Subset zipped successfully!")
print("🔗 ZIP path:", subset_zip)
print("🎉 Done! You can download covidxct_subset.zip from the Output tab.")
