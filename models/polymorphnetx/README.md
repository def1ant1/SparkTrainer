
# PolymorphNet-X (Hardened + SPM + Multimodal)

- Deterministic byte tokenizer (baseline) + **SentencePiece** wrapper with fixed special IDs
- **Causal mask** wired in
- **Portable checkpoints** (state_dict) + DeepSpeed engine checkpoints
- **Metrics**: loss, perplexity, token accuracy, control-token F1
- **Multimodal**: Image & Audio adapters with `[IMG]...[/IMG]`, `[AUD]...[/AUD]`

## Setup
```bash
bash scripts/setup.sh
```

## Train SPM
```bash
python train/spm_train.py --input data/corpus.txt --model_prefix configs/spm --vocab_size 50000
```

## Train (DeepSpeed multi-GPU)
```bash
export NUM_GPUS=8
bash scripts/run_deepspeed.sh
```

## Multimodal demo
```bash
python train/mm_toy_train.py
```

## Evaluate
```bash
python scripts/quick_eval.py outputs/checkpoints/policy/pnetx_state_step_1000.pt
```

### v2 Improvements
- **MoE**: token-level Top‑K routing with auxiliary load-balance loss & capacity factor.
- **Modality embeddings** across text/image/audio.
- **Grad checkpointing** hook in core model.
- **EMA**, **cosine LR** with warmup, **grad clipping**.
