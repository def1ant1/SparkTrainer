
#!/usr/bin/env bash
set -euo pipefail
nvidia-smi || { echo "CUDA not available or driver missing"; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install deepspeed==0.14.4 transformers datasets accelerate einops sentencepiece tensorboard wandb loguru opencv-python-headless soundfile librosa pillow pytest
echo "Environment ready."
