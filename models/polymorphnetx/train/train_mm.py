
import os, json, argparse
from pathlib import Path
from loguru import logger
import torch, torch.nn as nn
from torch.utils.data import DataLoader

try:
    import deepspeed
except Exception:
    deepspeed = None

from model.config import CONF
from model.multimodal_model_v2 import PolymorphNetXMultimodalV2
from train.mm_manifest import MMJsonlDataset, mm_collate
from train.metrics import compute_metrics
from train.checkpoint import save_state_dict

def _eval(model, val_loader, device):
    agg = {"loss":0.0,"perplexity":0.0,"token_acc":0.0,"control_f1":0.0}
    n = 0
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            logits = model(batch)
            m = compute_metrics(logits, batch["labels"])
            for k in agg: agg[k] += m[k]
            n += 1
        model.train()
    for k in agg: agg[k] /= max(n,1)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--ds_config", default="configs/deepspeed_config.json")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--out_dir", default="outputs/checkpoints_mm")
    args = ap.parse_args()

    ds = MMJsonlDataset(args.manifest)
    val_len = max(1, len(ds)//20)
    ds_val = torch.utils.data.Subset(ds, range(val_len))

    model = PolymorphNetXMultimodalV2(dim=CONF.d_model)

    if deepspeed is not None and torch.cuda.is_available():
        with open(args.ds_config, "r") as f:
            ds_conf = json.load(f)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_conf)
        device = engine.device
        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=mm_collate, pin_memory=True)
        val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=mm_collate, pin_memory=True)
        step = 0
        for epoch in range(args.epochs):
            for batch in train_loader:
                batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                logits = engine(batch)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100
                )
                engine.backward(loss); engine.step(); step += 1
                if step % 50 == 0: logger.info(f"epoch={epoch} step={step} loss={loss.item():.4f}")
                if step % args.eval_every == 0:
                    m = _eval(engine.module, val_loader, device)
                    logger.info(f"[eval] step={step} " + " ".join(f"{k}={v:.4f}" for k,v in m.items()))
                if step % args.save_every == 0:
                    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
                    engine.save_checkpoint(args.out_dir + f"/engine_step_{step}")
                    sd_path = save_state_dict(engine.module, args.out_dir, step)
                    logger.info(f"Saved engine + state_dict: {sd_path}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9,0.95))
        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=mm_collate)
        val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=mm_collate)
        step = 0
        for epoch in range(args.epochs):
            for batch in train_loader:
                batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                opt.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100
                )
                loss.backward(); opt.step(); step += 1
                if step % 50 == 0: logger.info(f"epoch={epoch} step={step} loss={loss.item():.4f}")
                if step % args.eval_every == 0:
                    m = _eval(model, val_loader, device)
                    logger.info(f"[eval] step={step} " + " ".join(f"{k}={v:.4f}" for k,v in m.items()))
                if step % args.save_every == 0:
                    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
                    sd_path = save_state_dict(model, args.out_dir, step)
                    logger.info(f"Saved state_dict: {sd_path}")

if __name__ == "__main__":
    main()
