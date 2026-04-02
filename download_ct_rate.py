import os
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LAPTOP-FRIENDLY CONFIGURATION ---
REPO_ID = "ibrahimhamamci/CT-RATE"
DEST_DIR = "./data/ct_rate_subset"
HF_TOKEN = os.getenv("HF_TOKEN") 

# Set this to 5 for initial testing on a laptop to save space (~1.5GB total)
# Set to 50+ only if you have 20GB+ free space.
VOLUME_LIMIT = 5 

def check_disk_space(required_gb=2.0):
    """Checks if the destination drive has enough space."""
    # Use the absolute path since relative path anchor returns an empty string
    path_to_check = Path(DEST_DIR).resolve() if Path(DEST_DIR).exists() else Path(".").resolve()
    total, used, free = shutil.disk_usage(path_to_check)
    free_gb = free / (2**30)
    if free_gb < required_gb:
        print(f"Warning: Only {free_gb:.2f} GB free. You might need more space!")
        return False
    return True

def download_subset():
    """
    Downloads a minimal subset of CT-RATE for development on limited storage.
    """
    Path(DEST_DIR).mkdir(parents=True, exist_ok=True)
    
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        return

    api = HfApi(token=HF_TOKEN)
    print(f"Connecting to {REPO_ID}...")
    
    try:
        all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"Error: Access denied. Have you accepted the terms on Hugging Face?\n{e}")
        return

    all_files_set = set(all_files)

    # 1. Essential Metadata — try valid split first, fall back to train
    valid_meta_patterns = [
        "dataset/metadata/valid_metadata.csv",
        "dataset/radiology_text_reports/valid_reports.csv",
    ]
    train_meta_patterns = [
        "dataset/metadata/train_metadata.csv",
        "dataset/radiology_text_reports/train_reports.csv",
    ]

    # Prefer valid CSVs (matches the valid volumes we download below).
    # Fall back to train CSVs only if valid ones are absent from the repo.
    if all(p in all_files_set for p in valid_meta_patterns):
        meta_to_download = valid_meta_patterns
        volume_prefix = "dataset/valid_fixed/"
        print("Using valid split CSVs.")
    else:
        meta_to_download = train_meta_patterns
        volume_prefix = "dataset/train_fixed/"
        print(f"Valid CSVs not found in repo — falling back to train split.")
        print(f"Available metadata files: {[f for f in all_files if 'metadata' in f or 'reports' in f]}")

    # 2. 3D Volumes — match the same split as the CSVs
    volume_files = [
        f for f in all_files
        if f.startswith(volume_prefix) and f.endswith(".nii.gz")
    ]

    files_to_download = meta_to_download + volume_files[:VOLUME_LIMIT]

    print(f"Plan: Download {len(meta_to_download)} metadata files and {min(len(volume_files), VOLUME_LIMIT)} volumes.")
    check_disk_space(required_gb=VOLUME_LIMIT * 0.4)  # Estimate ~400MB per volume

    for filename in tqdm(files_to_download, desc="Downloading Files"):
        local_path = Path(DEST_DIR) / filename
        if local_path.exists():
            continue

        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )

    print(f"\nSetup Complete! Subset available at: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    download_subset()
