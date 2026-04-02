"""
preprocess_and_extract.py
Preprocesses chest CT volumes and extracts visual features using CT-CLIP.

Setup (one-time, before running):
    1. Clone CT-CLIP source into this repo root:
           git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
       Then install both sub-packages:
           pip install -e CT-CLIP/transformer_maskgit
           pip install -e CT-CLIP/CT_CLIP
    2. Ensure HF_TOKEN is set in your .env file (gated model access).

Run:
    python preprocess_and_extract.py

Output:
    ./data/processed_features/<VolumeName>.pt
    Each file is a dict: {"features": Tensor(NUM_VISUAL_TOKENS, 512), "label": str}
"""
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged, Resized,
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---- CONFIGURATION ----
DATA_DIR    = Path("./data/ct_rate_subset")
OUTPUT_DIR  = Path("./data/processed_features")
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# CT-CLIP's expected spatial input (H, W, D).
# H and W must be divisible by CTViT patch_size (20).
# D must be divisible by CTViT temporal_patch_size (10).
# Full-resolution (model training size): (480, 480, 240) — slow on CPU.
# Reduced size for testing (faster, ~8x fewer tokens):  (160, 160, 80).
VOLUME_SIZE = (480, 480, 240)
# Number of patch tokens to keep after spatial pooling (passed to the LLM).
NUM_VISUAL_TOKENS = 256
VISUAL_DIM  = 512   # CTViT encoder output dimension


# ---- PREPROCESSING PIPELINE ----
# Normalises raw NIfTI volumes into the [0, 1] range that CT-CLIP expects.
ct_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    # Canonical orientation: Right-Anterior-Superior
    Orientationd(keys=["image"], axcodes="RAS"),
    # Isotropic-ish voxel spacing in mm
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    # Chest CT Lung Window (HU -1000 → +400) → [0, 1]
    ScaleIntensityRanged(
        keys=["image"], a_min=-1000, a_max=400,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    Resized(keys=["image"], spatial_size=VOLUME_SIZE),
])


# ---- CT-CLIP ENCODER LOADING ----
def build_ctclip_encoder(device: str) -> nn.Module:
    """
    Downloads the CT-CLIP checkpoint from HuggingFace (CT-RATE dataset repo)
    and returns the CTViT vision encoder with pretrained weights loaded.

    Checkpoint source: ibrahimhamamci/CT-RATE  →  models/CT-CLIP-Related/CT-CLIP_v2.pt
    The full CTCLIP checkpoint stores the vision encoder under the
    "visual_transformer.*" key prefix.
    """
    from transformer_maskgit import CTViT

    ckpt_path = hf_hub_download(
        repo_id="ibrahimhamamci/CT-RATE",
        filename="models/CT-CLIP-Related/CT-CLIP_v2.pt",
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir="./model_weights",
    )

    # Initialise CTViT with the same hparams used during CT-CLIP training.
    # use_vgg_and_gan=False disables the discriminator head (not needed for inference).
    encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        use_vgg_and_gan=False,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)

    # The CTCLIP checkpoint stores CTViT weights under "visual_transformer.*"
    visual_sd = {
        k[len("visual_transformer."):]: v
        for k, v in sd.items()
        if k.startswith("visual_transformer.")
    }
    if not visual_sd:
        raise RuntimeError(
            "Could not find 'visual_transformer.*' keys in checkpoint. "
            f"Top-level keys: {list(sd.keys())[:10]}"
        )

    missing, unexpected = encoder.load_state_dict(visual_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading CTViT: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return encoder.to(device).eval()


class CTCLIPEncoderWrapper(nn.Module):
    """
    Thin wrapper around the CT-CLIP vision encoder that:
      1. Runs the volume through the encoder → (B, N, 768) raw patch tokens.
      2. Applies adaptive average pooling → (B, NUM_VISUAL_TOKENS, 768).

    The fixed token count lets the training collator build uniform batches
    without padding, and keeps the sequence length manageable for the LLM.
    """
    def __init__(self, encoder: nn.Module, num_output_tokens: int = NUM_VISUAL_TOKENS):
        super().__init__()
        self.encoder = encoder
        self.num_output_tokens = num_output_tokens

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W, D) float32, values in [0, 1]  — MONAI channel-first format
        Returns:
            tokens: (B, num_output_tokens, VISUAL_DIM) float32
        """
        # CTViT expects (B, C, T, H, W) — depth is the temporal dimension.
        # MONAI gives (B, C, H, W, D), so permute D to position 2.
        x = x.permute(0, 1, 4, 2, 3)                   # (B, 1, D, H, W)

        # CTViT returns (B, t, h, w, dim) with return_encoded_tokens=True
        tokens = self.encoder(x, return_encoded_tokens=True)  # (B, t, h, w, dim)

        # Flatten spatial/temporal patch dims → (B, N, dim)
        B, t, h, w, dim = tokens.shape
        tokens = tokens.reshape(B, t * h * w, dim)      # (B, N, dim)

        # Pool to a fixed token count for the LLM
        tokens = tokens.permute(0, 2, 1)                # (B, dim, N)
        tokens = F.adaptive_avg_pool1d(tokens, self.num_output_tokens)  # (B, dim, T)
        tokens = tokens.permute(0, 2, 1)                # (B, T, dim)
        return tokens


# ---- PATH HELPERS ----
def volume_name_to_path(volume_name: str, data_dir: Path) -> Path:
    """
    Reconstructs the nested CT-RATE directory path from a bare volume filename.

    CT-RATE hierarchy:
        {split}_fixed / {split}_{num} / {split}_{num}_{letter} / {volume_name}

    Example:
        'valid_1_a_1.nii.gz'
        → data_dir/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz
    """
    stem = volume_name.replace(".nii.gz", "")   # 'valid_1_a_1'
    parts = stem.split("_")                      # ['valid', '1', 'a', '1']
    split      = parts[0]                        # 'valid' | 'train'
    patient_id = f"{split}_{parts[1]}"           # 'valid_1'
    series_id  = f"{split}_{parts[1]}_{parts[2]}"  # 'valid_1_a'
    return (
        data_dir / "dataset" / f"{split}_fixed"
        / patient_id / series_id / volume_name
    )


def load_split_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Loads the first available (metadata, reports) CSV pair.
    Tries the validation split first since that is what is downloaded locally.
    """
    candidates = [
        ("valid_metadata.csv", "valid_reports.csv"),
        ("train_metadata.csv", "train_reports.csv"),
    ]
    for meta_file, report_file in candidates:
        meta_path   = data_dir / "dataset" / "metadata" / meta_file
        report_path = data_dir / "dataset" / "radiology_text_reports" / report_file
        if meta_path.exists() and report_path.exists():
            meta_df   = pd.read_csv(meta_path)
            report_df = pd.read_csv(report_path)
            # Normalise column name: CT-RATE uses 'Findings_EN'
            report_df = report_df.rename(columns={"Findings_EN": "Findings"})
            df = pd.merge(
                meta_df[["VolumeName"]],
                report_df[["VolumeName", "Findings"]],
                on="VolumeName",
            )
            print(f"Loaded {len(df)} records from '{meta_file}' + '{report_file}'.")
            return df
    raise FileNotFoundError(
        "No (metadata, reports) CSV pair found under "
        f"{data_dir}/dataset/.  Run download_ct_rate.py first."
    )


# ---- MAIN ----
def run_extraction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_split_dataframe(DATA_DIR)

    # Resolve local file paths and drop volumes without findings text
    data_list = []
    for _, row in df.iterrows():
        file_path = volume_name_to_path(row["VolumeName"], DATA_DIR)
        if file_path.exists() and pd.notna(row["Findings"]) and row["Findings"].strip():
            data_list.append({
                "image": str(file_path),
                "label": row["Findings"].strip(),
                "id":    row["VolumeName"],
            })

    print(f"Found {len(data_list)} local volumes with valid findings text.")
    if not data_list:
        raise RuntimeError(
            "No matching volumes found.  "
            "Verify that download_ct_rate.py ran successfully."
        )

    # Load CT-CLIP encoder
    print("Loading CT-CLIP encoder…")
    raw_encoder = build_ctclip_encoder(DEVICE)
    encoder = CTCLIPEncoderWrapper(raw_encoder, num_output_tokens=NUM_VISUAL_TOKENS)
    encoder = encoder.to(DEVICE).eval()

    ds = Dataset(data=data_list, transform=ct_transforms)
    dl = DataLoader(
        ds, batch_size=1, num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )

    skipped = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting Features"):
            volume_id = batch["id"][0]
            out_path  = OUTPUT_DIR / f"{volume_id}.pt"

            if out_path.exists():
                continue  # Resume support: skip already-processed files

            try:
                image  = batch["image"].to(DEVICE)  # (1, 1, H, W, D)
                tokens = encoder(image)              # (1, NUM_VISUAL_TOKENS, 768)
                torch.save(
                    {
                        # Store as float16 to halve disk usage
                        "features": tokens.squeeze(0).cpu().half(),
                        "label":    batch["label"][0],
                    },
                    out_path,
                )
            except Exception as exc:
                import traceback
                print(f"[WARN] Skipped '{volume_id}': {exc}")
                traceback.print_exc()
                skipped += 1

    saved = len(data_list) - skipped
    print(f"\nDone. {saved} feature files saved to '{OUTPUT_DIR}'. Skipped: {skipped}.")


if __name__ == "__main__":
    run_extraction()
