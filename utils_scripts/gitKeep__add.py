import os

def add_gitkeep(root_dir: str, overwrite: bool = False):
    """
    Inserts a .gitKeep file in every folder under root_dir.
    
    :param root_dir: The directory to process
    :param overwrite: If True, overwrite existing .gitKeep files
    """
    for folder, subfolders, files in os.walk(root_dir):
        gitkeep_path = os.path.join(folder, ".gitKeep")

        if os.path.exists(gitkeep_path):
            if overwrite:
                with open(gitkeep_path, "w") as f:
                    pass
                print(f"Overwritten: {gitkeep_path}")
            else:
                print(f"Skipped (already exists): {gitkeep_path}")
        else:
            with open(gitkeep_path, "w") as f:
                pass
            print(f"Created: {gitkeep_path}")


if __name__ == "__main__":
    # ---- EDIT THIS ----
    directory = r"D:\xai-ct-project - Copy\data"
    overwrite_existing = False  # Change to True if you want to overwrite

    add_gitkeep(directory, overwrite_existing)
