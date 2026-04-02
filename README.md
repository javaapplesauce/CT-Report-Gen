CT-Gen: 3D Chest CT Radiology Report Generation

This project implements a generative Vision-Language Model (VLM) for automatic radiology report generation using the CT-RATE dataset. It transitions the original CT-CLIP (contrastive/classification) architecture into an Encoder-Projector-Decoder framework.

🚀 The Strategy: "Feature Alignment"

To handle the 21TB scale of CT-RATE on limited hardware, we use a two-stage approach:

Frozen Vision Expertise: We use the pretrained 3D Swin-Transformer from CT-CLIP to extract visual features.

Multimodal Bridge: We train a 2-layer MLP (Projector) to map these 3D features into the embedding space of a causal LLM (e.g., Llama-3-8B or Phi-3-mini).

📂 Project Structure

download_ct_rate.py: Script to download a manageable subset of NIfTI volumes and metadata.

preprocess_and_extract.py: MONAI-based pipeline to standardize 3D volumes (1.5mm spacing, Lung Windowing) and save lightweight feature tensors.

train_gen.py: (Next Step) QLoRA fine-tuning of the LLM using the extracted features.

data/:

ct_rate_subset/: Raw NIfTI files and CSVs.

processed_features/: Extracted .pt tensors (the actual training data).

🩺 Medical Preprocessing Specs

Standardization is critical for 3D medical AI. Our pipeline ensures:

Orientation: RAS (Right, Anterior, Superior).

Spacing: Resampled to $(1.5, 1.5, 2.0)$ mm.

Windowing: Hounsfield Units (HU) clipped to $[-1000, 400]$ to highlight lung parenchyma.

Resolution: Spatial size fixed to $(128, 128, 64)$ for consistent token length.

📊 Evaluation Metrics

Since standard NLP metrics (BLEU/ROUGE) don't capture clinical truth, we evaluate using:

Clinical F1: Extracting the 18 CT-RATE abnormalities from generated text and comparing them to ground truth labels.

RadGraph F1: Measuring the overlap of clinical entities and relations.

ROUGE-L / METEOR: For general linguistic fluency.

🛠️ Usage

Download: python download_ct_rate.py (Default: 5 volumes for laptop testing).

Extract: python preprocess_and_extract.py (Converts NIfTIs to tiny tensors).

Train: (Upcoming) Fine-tune the LLM projector and LoRA adapters.

Note: Access to the CT-RATE dataset is gated. Ensure you have accepted the terms on Hugging Face and set your HF_TOKEN.