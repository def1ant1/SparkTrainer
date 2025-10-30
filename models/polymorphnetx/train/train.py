
import os, json, yaml, argparse, math
from pathlib import Path
from loguru import logger
import torch, torch.nn as nn
from torch.utils.data import DataLoader
try:
    import deepspeed
except Exception:
    deepspeed = None

from model.model import PolymorphNetX
from model.config import CONF
from train.data import JsonlDataset, collate
from train.curriculum import filter_fn
from train.metrics import compute_metrics
from train.checkpoint import save_state_dict

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

def load_yaml(path): return yaml.safe_load(open(path, "r"))

def split_curriculum(train_path, stage):
    import json, tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    keep = 0
    with open(train_path, "r", encoding="utf-8") as f, open(tmp.name, "w", encoding="utf-8") as out:
        for line in f:
            obj = json.loads(line)
            if filter_fn(obj, stage):
                out.write(json.dumps(obj) + "\\n")
                keep += 1
    logger.info(f"Stage={stage}: kept {keep} examples.")
    return tmp.name

def cosine_with_warmup(opt, warmup, total):
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def evaluate(model, loader, device):
    model.eval()
    agg = {"loss":0.0,"perplexity":0.0,"token_acc":0.0,"control_f1":0.0}
    n=0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            logits = model(input_ids)
            m = compute_metrics(logits, labels)
            for k in agg: agg[k] += m[k]
            n += 1
    for k in agg: agg[k] /= max(n,1)
    model.train()
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--ds_config", default="configs/deepspeed_config.json")
    ap.add_argument("--stage", default="pretrain", choices=["pretrain","dag","policy"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    total_steps = cfg["total_steps"]
    out_dir = Path(args.out_dir or cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    stage_train = split_curriculum(cfg["train_path"], args.stage)
    ds_train = JsonlDataset(stage_train)
    n_val = max(1, len(ds_train)//20)
    ds_val = torch.utils.data.Subset(ds_train, range(n_val))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolymorphNetX().to(device)

    if deepspeed is not None and torch.cuda.is_available():
        with open(args.ds_config, "r") as f: ds_conf = json.load(f)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_conf)
        ema = EMA(engine.module, decay=args.ema_decay)
        train_loader = DataLoader(ds_train, batch_size=ds_conf["train_micro_batch_size_per_gpu"], shuffle=True, collate_fn=collate, pin_memory=True)
        val_loader   = DataLoader(ds_val,   batch_size=ds_conf["train_micro_batch_size_per_gpu"], shuffle=False, collate_fn=collate, pin_memory=True)
        global_step = 0
        while global_step < total_steps:
            for batch in train_loader:
                logits = engine(batch["input_ids"].to(engine.device))
                labels = batch["labels"].to(engine.device)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                engine.backward(loss); engine.step(); global_step += 1
                ema.update(engine.module)
                if global_step % cfg["log_every_steps"] == 0:
                    logger.info(f"step={global_step} stage={args.stage} loss={loss.item():.4f}")
                if global_step % args.eval_every == 0:
                    m = evaluate(engine.module, val_loader, engine.device)
                    logger.info("[eval] step=%d " % global_step + " ".join(f"{k}={v:.4f}" for k,v in m.items()))
                if global_step % args.save_every == 0:
                    ckpt_dir = out_dir / f"step_{global_step}"; engine.save_checkpoint(str(ckpt_dir))
                    sd_path = save_state_dict(engine.module, str(out_dir), global_step)
                    logger.info(f"Saved engine@{ckpt_dir} | state_dict={sd_path}")
                if global_step >= total_steps: break
        ema.apply_to(engine.module)
        sd_path = save_state_dict(engine.module, str(out_dir), global_step)
        logger.info(f"Saved EMA state_dict={sd_path}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
        sched = cosine_with_warmup(opt, cfg["warmup_steps"], cfg["total_steps"])
        ema = EMA(model, decay=args.ema_decay)
        train_loader = DataLoader(ds_train, batch_size=cfg["micro_batch_size_per_gpu"], shuffle=True, collate_fn=collate)
        val_loader   = DataLoader(ds_val,   batch_size=cfg["micro_batch_size_per_gpu"], shuffle=False, collate_fn=collate)
        global_step = 0
        while global_step < total_steps:
            for batch in train_loader:
                model.train()
                opt.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step(); sched.step(); global_step += 1
                ema.update(model)
                if global_step % cfg["log_every_steps"] == 0:
                    logger.info(f"step={global_step} stage={args.stage} loss={loss.item():.4f}")
                if global_step % args.eval_every == 0:
                    m = evaluate(model, val_loader, device)
                    logger.info("[eval] step=%d " % global_step + " ".join(f"{k}={v:.4f}" for k,v in m.items()))
                if global_step % args.save_every == 0:
                    sd_path = save_state_dict(model, str(out_dir), global_step)
                    logger.info(f"Saved state_dict={sd_path}")
                if global_step >= total_steps: break
        ema.apply_to(model)
        sd_path = save_state_dict(model, str(out_dir), global_step)
        logger.info(f"Saved EMA state_dict={sd_path}")

if __name__ == "__main__":
    main()
