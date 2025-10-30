
#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
torchrun --standalone --nproc_per_node=${NUM_GPUS:-1} train/train.py --config configs/training.yaml --stage pretrain
