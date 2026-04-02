"""
train.py
Fine-tunes a small LLM (Phi-3-mini or Llama-3-8B) on CT radiology report generation
using pre-extracted CT-CLIP features and PEFT / LoRA.

Architecture
------------
  [CT-CLIP features]  (NUM_VISUAL_TOKENS, 512)
       ↓  MedicalProjector (768 → LLM_DIM)
  [Visual embeddings] (NUM_VISUAL_TOKENS, LLM_DIM)
       ↓  concat with text embeddings
  [LLM + LoRA]        loss masked to text positions only
       ↓
  "Findings: <generated report>"

Multi-GPU usage (2 × GPU)
--------------------------
  # One-time setup:
  accelerate config          # choose "multi-GPU", 2 processes, bf16
  # Then every run:
  accelerate launch train.py

Single-GPU / CPU
-----------------
  python train.py

Requirements
------------
  pip install transformers peft accelerate bitsandbytes pandas tqdm
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model for local testing. Swap back to "microsoft/Phi-3-mini-4k-instruct" (LLM_DIM=3072)
# or "meta-llama/Meta-Llama-3-8B" (LLM_DIM=4096) for production training on a GPU machine.
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_DIM      = 2048   # TinyLlama hidden dim.

# Feature dimensions (must match preprocess_and_extract.py)
VISUAL_DIM        = 512   # CTViT encoder output dimension
NUM_VISUAL_TOKENS = 256

# Data paths
FEATURES_DIR = Path("./data/processed_features")
REPORTS_CSV  = Path("./data/ct_rate_subset/dataset/radiology_text_reports/train_reports.csv")

# Training
OUTPUT_DIR        = Path("./checkpoints")
EPOCHS            = 1
BATCH_SIZE        = 1      # per GPU
GRAD_ACCUM_STEPS  = 1      # effective batch = BATCH_SIZE × num_gpus × GRAD_ACCUM_STEPS
LR                = 2e-4
WEIGHT_DECAY      = 0.01
WARMUP_FRACTION   = 0.05   # fraction of total steps used for LR warm-up
MAX_GRAD_NORM     = 1.0
MAX_TEXT_LEN      = 512    # max tokens for the radiology report text

# LoRA
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
# Applies LoRA to all standard attention + MLP projection matrices.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LOG_EVERY = 50   # steps between loss prints


# ============================================================================
# PROJECTOR
# ============================================================================

class MedicalProjector(nn.Module):
    """
    Two-layer MLP with GELU that maps CT-CLIP patch tokens (VISUAL_DIM)
    into the LLM's embedding space (LLM_DIM).

    Trained from scratch; LoRA is applied separately to the LLM.
    """
    def __init__(self, visual_dim: int = VISUAL_DIM, llm_dim: int = LLM_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(visual_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, NUM_VISUAL_TOKENS, visual_dim) → (B, NUM_VISUAL_TOKENS, llm_dim)"""
        return self.mlp(self.norm(x))


# ============================================================================
# DATASET
# ============================================================================

class CTReportDataset(Dataset):
    """
    Loads pre-extracted CT-CLIP feature files and pairs them with findings text.

    Each feature file (produced by preprocess_and_extract.py) is a dict:
        {"features": Tensor(NUM_VISUAL_TOKENS, 768, fp16), "label": str}

    Falls back to the reports CSV for volumes whose feature files do not yet
    contain an embedded label (older format compatibility).
    """
    def __init__(
        self,
        features_dir: Path,
        reports_csv: Path,
    ):
        self.features_dir = features_dir
        self.samples: list[dict] = []

        # Build an optional volume→findings lookup from the CSV
        csv_lookup: dict[str, str] = {}
        if reports_csv.exists():
            df = pd.read_csv(reports_csv)
            # Normalise column name
            findings_col = "Findings_EN" if "Findings_EN" in df.columns else "Findings"
            for _, row in df.iterrows():
                text = str(row.get(findings_col, "")).strip()
                if text:
                    csv_lookup[row["VolumeName"]] = text

        # Collect feature files that have a label
        for pt_path in sorted(features_dir.glob("*.pt")):
            volume_name = pt_path.stem + ".nii.gz"  # reconstruct VolumeName
            # Try embedded label first (written by new extractor)
            data  = torch.load(pt_path, map_location="cpu", weights_only=False)
            label = data.get("label") or csv_lookup.get(volume_name, "")
            if label:
                self.samples.append({"pt_path": pt_path, "label": label})

        print(f"CTReportDataset: {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data     = torch.load(s["pt_path"], map_location="cpu", weights_only=False)
        features = data["features"].float()  # (NUM_VISUAL_TOKENS, 768) fp16→fp32
        return features, s["label"]


# ============================================================================
# COLLATE + MASKED LABELS
# ============================================================================

def build_collate_fn(tokenizer, num_visual_tokens: int, max_text_len: int):
    """
    Returns a collate function that builds:

      inputs_embeds : built inside forward() — NOT returned here
      input_ids     : (B, max_text_len)  — used to look up text embeddings
      attention_mask: (B, num_visual_tokens + max_text_len) — 1 everywhere valid
      labels        : (B, num_visual_tokens + max_text_len)
                      -100 for visual positions (never predicted)
                      -100 for padding positions
                      real token ids for text positions

    The LLM's built-in causal-LM loss computes:
        loss = CE(logits[:, :-1], labels[:, 1:])
    so masking visual labels to -100 ensures the model only learns to predict
    the radiology report words, not any visual tokens.
    """
    PROMPT_PREFIX = "Findings: "
    prefix_ids = tokenizer(
        PROMPT_PREFIX, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].squeeze(0)  # (P,)
    P = prefix_ids.shape[0]

    def collate(batch):
        visual_feats, texts = zip(*batch)
        visual_feats = torch.stack(visual_feats)  # (B, V, visual_dim)
        B = len(texts)

        # Tokenise findings text (leave room for prefix)
        encoding = tokenizer(
            list(texts),
            max_length=max_text_len - P,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_ids  = encoding["input_ids"]          # (B, T-P)
        text_mask = encoding["attention_mask"]      # (B, T-P)

        # Prepend prompt prefix to every sequence
        prefix_batch = prefix_ids.unsqueeze(0).expand(B, -1)           # (B, P)
        input_ids    = torch.cat([prefix_batch, text_ids], dim=1)       # (B, T)
        prefix_mask  = torch.ones(B, P, dtype=torch.long)
        text_attn    = torch.cat([prefix_mask, text_mask], dim=1)       # (B, T)

        # ── Attention mask: visual slots + text slots ──
        visual_attn  = torch.ones(B, num_visual_tokens, dtype=torch.long)
        full_attn    = torch.cat([visual_attn, text_attn], dim=1)       # (B, V+T)

        # ── Labels ──
        # Visual positions: never predict → -100
        visual_labels = torch.full((B, num_visual_tokens), -100, dtype=torch.long)
        # Text positions: real ids where not padding, else -100
        text_labels   = input_ids.clone()                               # (B, T)
        text_labels[text_attn == 0] = -100
        full_labels = torch.cat([visual_labels, text_labels], dim=1)   # (B, V+T)

        return visual_feats, input_ids, full_attn, full_labels

    return collate


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    # Accelerator handles device placement, DDP wrapping, mixed precision, and
    # gradient accumulation for free — no manual dist.init_process_group needed.
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    is_main = accelerator.is_main_process

    if is_main:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base LLM ───────────────────────────────────────────────────────────
    # bfloat16 on CUDA/CPU; float16 on MPS (bfloat16 unsupported on Apple Silicon).
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        dtype=model_dtype,
        trust_remote_code=False,
    )

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    if is_main:
        model.print_trainable_parameters()

    # ── Projector ──────────────────────────────────────────────────────────
    projector = MedicalProjector(visual_dim=VISUAL_DIM, llm_dim=LLM_DIM)

    # ── Dataset + DataLoader ───────────────────────────────────────────────
    dataset = CTReportDataset(FEATURES_DIR, REPORTS_CSV)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No feature files found in '{FEATURES_DIR}'. "
            "Run preprocess_and_extract.py first."
        )

    collate_fn = build_collate_fn(tokenizer, NUM_VISUAL_TOKENS, MAX_TEXT_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # ── Optimiser ──────────────────────────────────────────────────────────
    # Projector params are fully trained; LoRA params are the LLM's trainable subset.
    trainable_params = (
        list(projector.parameters())
        + [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=LR, weight_decay=WEIGHT_DECAY
    )

    total_steps = (len(dataloader) * EPOCHS) // GRAD_ACCUM_STEPS
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate preparation ─────────────────────────────────────────────
    # prepare() wraps models in DDP, moves everything to the right device,
    # and hooks into the dataloader for distributed sampling automatically.
    model, projector, optimizer, dataloader, scheduler = accelerator.prepare(
        model, projector, optimizer, dataloader, scheduler
    )

    # ── Training ───────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        projector.train()
        running_loss = 0.0
        running_steps = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            disable=not is_main,
        )

        for batch in progress:
            visual_feats, input_ids, attention_mask, labels = batch
            # visual_feats : (B, V, visual_dim)
            # input_ids    : (B, T)
            # attention_mask: (B, V+T)
            # labels       : (B, V+T)   -100 for visual + padding positions

            with accelerator.accumulate(model):
                # 1. Project CT features → LLM embedding space
                #    Cast projector output to the LLM's working dtype (bfloat16).
                visual_embeds = projector(visual_feats).to(model.dtype)  # (B, V, D)

                # 2. Look up text token embeddings from the LLM's embedding table.
                #    We call get_input_embeddings() on the *unwrapped* model so this
                #    works identically in single-GPU and DDP contexts.
                embed_layer  = accelerator.unwrap_model(model).get_input_embeddings()
                text_embeds  = embed_layer(input_ids).to(model.dtype)    # (B, T, D)

                # 3. Concatenate: [visual tokens | text tokens]
                #    The LLM sees the visual context first, then the report text.
                inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, V+T, D)

                # 4. Forward pass.
                #    HuggingFace causal-LM internally computes:
                #        loss = CE(logits[:, :-1], labels[:, 1:])
                #    Because labels[:, :V] == -100, the visual token positions
                #    contribute zero gradient — only the report text is learned.
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss  += loss.detach().float()
            running_steps += 1
            global_step   += 1

            if global_step % LOG_EVERY == 0 and is_main:
                avg = running_loss / running_steps
                progress.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                running_loss  = 0.0
                running_steps = 0

        # ── Checkpoint after each epoch ────────────────────────────────────
        accelerator.wait_for_everyone()
        if is_main:
            epoch_dir = OUTPUT_DIR / f"epoch_{epoch + 1}"
            epoch_dir.mkdir(exist_ok=True)

            # Save LoRA adapter weights only (much smaller than full model)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(epoch_dir / "lora_adapter")
            tokenizer.save_pretrained(epoch_dir / "lora_adapter")

            # Save projector weights
            torch.save(
                accelerator.unwrap_model(projector).state_dict(),
                epoch_dir / "projector.pt",
            )
            print(f"[Epoch {epoch + 1}] Checkpoint saved → {epoch_dir}")

    accelerator.end_training()


# ============================================================================
# INFERENCE HELPER  (for quick manual testing after training)
# ============================================================================

@torch.no_grad()
def generate_report(
    feature_pt: str | Path,
    lora_dir:   str | Path,
    projector_pt: str | Path,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    """
    Generates a Findings report from a pre-extracted feature file.

    Usage:
        text = generate_report(
            feature_pt   = "./data/processed_features/valid_1_a_1.nii.gz.pt",
            lora_dir     = "./checkpoints/epoch_3/lora_adapter",
            projector_pt = "./checkpoints/epoch_3/projector.pt",
        )
        print(text)
    """
    from peft import PeftModel

    # Load model + projector
    tokenizer  = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, dtype=torch.bfloat16, trust_remote_code=False
    ).to(device)
    model = PeftModel.from_pretrained(base_model, lora_dir).eval().to(device)

    projector = MedicalProjector(VISUAL_DIM, LLM_DIM)
    projector.load_state_dict(torch.load(projector_pt, map_location=device))
    projector = projector.to(device).eval()

    # Load features
    data     = torch.load(feature_pt, map_location=device, weights_only=True)
    features = data["features"].float().unsqueeze(0).to(device)  # (1, V, 768)

    # Build prompt
    prefix    = "Findings: "
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)

    visual_embeds = projector(features).to(model.dtype)           # (1, V, D)
    embed_layer   = model.get_input_embeddings()
    prefix_embeds = embed_layer(prefix_ids)                       # (1, P, D)
    inputs_embeds = torch.cat([visual_embeds, prefix_embeds], dim=1)  # (1, V+P, D)

    out_ids = model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train()
