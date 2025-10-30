
import torch
from pathlib import Path
def save_state_dict(model, out_dir: str, step: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"pnetx_state_step_{step}.pt"
    torch.save(model.state_dict(), path)
    return str(path)
def load_state_dict(model, ckpt_path: str, map_location=None):
    sd = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(sd, strict=True)
    return model
