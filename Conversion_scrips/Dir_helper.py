import os, shutil, random
from glob import glob

# All your datasets
sources = [
    "data/raw_nifti/COVID-CT-MD",
    "data/raw_nifti/COVIDx",
    "data/raw_nifti/SARS-CoV2"
]

target_root = "data/ct3d_dataset"

splits = ["train", "val", "test"]
classes = {"COVID": ["COVID"], "nonCOVID": ["CAP", "non-COVID", "NORMAL", "PNEUMONIA"]}

# Create directories
for s in splits:
    for c in classes.keys():
        os.makedirs(os.path.join(target_root, s, c), exist_ok=True)

# Gather all files
covid_files, noncovid_files = [], []
for src in sources:
    for c in classes["COVID"]:
        covid_files += glob(os.path.join(src, c, "*.nii.gz"))
    for c in classes["nonCOVID"]:
        noncovid_files += glob(os.path.join(src, c, "*.nii.gz"))

print(f"Found {len(covid_files)} COVID and {len(noncovid_files)} nonCOVID files")

# Shuffle and split (80/10/10) with minimum 1 per split
def split_data(files):
    random.shuffle(files)
    n = len(files)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    n_test = max(1, n - n_train - n_val)

    # adjust to ensure all files are used
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}

covid_split = split_data(covid_files)
noncovid_split = split_data(noncovid_files)

# Copy files
for s in splits:
    for f in covid_split[s]:
        dest = os.path.join(target_root, s, "COVID")
        shutil.copy(f, dest)
    for f in noncovid_split[s]:
        dest = os.path.join(target_root, s, "nonCOVID")
        shutil.copy(f, dest)

print("✅ Merged dataset organized under", target_root)
for s in splits:
    print(f"{s}: {len(os.listdir(os.path.join(target_root, s, 'COVID')))} COVID, "
          f"{len(os.listdir(os.path.join(target_root, s, 'nonCOVID')))} nonCOVID")
