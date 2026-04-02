import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi

# Load variables from .env file
load_dotenv()

# --- CONFIG ---
# The correct repo for the model weights
REPO_ID = "ibrahimhamamci/CT-CLIP" 
# Common filenames for research weights in this repo
FILENAME = "CT-CLIP.pt" 
SAVE_DIR = "./models/ct_clip"

# Access the token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

def download_ct_clip():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in .env file.")
        print("Please create a .env file with: HF_TOKEN=your_huggingface_token_here")
        return

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    api = HfApi(token=HF_TOKEN)
    
    print(f"Connecting to {REPO_ID}...")
    try:
        # Check available files if the default one fails
        files = api.list_repo_files(repo_id=REPO_ID)
        print(f"Available files in repo: {files}")
        
        # If CT-CLIP.pt isn't there, try to find the most likely candidate
        target_file = FILENAME
        if target_file not in files:
            candidates = [f for f in files if f.endswith(('.pt', '.bin', '.ckpt'))]
            if candidates:
                target_file = candidates[0]
                print(f"{FILENAME} not found. Trying best candidate: {target_file}")
            else:
                print(f"No weight files (.pt, .bin, .ckpt) found in {REPO_ID}")
                return

        print(f"Downloading {target_file}...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=target_file,
            token=HF_TOKEN,
            local_dir=SAVE_DIR
        )
        print(f"Model saved to: {path}")
        
    except Exception as e:
        print(f"Operation failed: {e}")

if __name__ == "__main__":
    download_ct_clip()
