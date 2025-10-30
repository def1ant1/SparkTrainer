
#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
NUM_GPUS=${NUM_GPUS:-8}
deepspeed --num_gpus ${NUM_GPUS} train/train.py --config configs/training.yaml --ds_config configs/deepspeed_config.json --stage pretrain --out_dir outputs/checkpoints/pretrain
deepspeed --num_gpus ${NUM_GPUS} train/train.py --config configs/training.yaml --ds_config configs/deepspeed_config.json --stage dag      --out_dir outputs/checkpoints/dag
deepspeed --num_gpus ${NUM_GPUS} train/train.py --config configs/training.yaml --ds_config configs/deepspeed_config.json --stage policy   --out_dir outputs/checkpoints/policy
