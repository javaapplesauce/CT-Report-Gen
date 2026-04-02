#!/usr/bin/env bash
# setup_env.sh
# Creates a Python virtual environment and installs all project dependencies.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh

set -e

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

echo "==> Checking Python version..."
$PYTHON --version

echo "==> Creating virtual environment at '$VENV_DIR'..."
$PYTHON -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing core dependencies..."
pip install \
    torch torchvision torchaudio \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    pandas \
    tqdm \
    python-dotenv \
    huggingface_hub \
    monai \
    nibabel

echo "==> Cloning CT-CLIP (vision encoder)..."
if [ ! -d "CT-CLIP" ]; then
    git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
fi

echo "==> Installing CT-CLIP sub-packages..."
pip install -e CT-CLIP/transformer_maskgit
pip install -e CT-CLIP/CT_CLIP

echo ""
echo "==> Setup complete."
echo ""
echo "    Activate the environment with:"
echo "        source $VENV_DIR/bin/activate"
echo ""
echo "    Then run:"
echo "        python download_ct_rate.py      # download data"
echo "        python preprocess_and_extract.py # extract CT-CLIP features"
echo "        python train.py                  # start training"
