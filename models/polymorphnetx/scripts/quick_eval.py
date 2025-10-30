
import torch, sys
from pathlib import Path
from model.model import PolymorphNetX
from train.checkpoint import load_state_dict
from model.config import TOKEN_IDS

@torch.no_grad()
def sample(m, prompt_ids, max_new=32):
    m.eval()
    device = next(m.parameters()).device
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new):
        logits = m(x)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    return x[0].tolist()

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    m = PolymorphNetX()
    if torch.cuda.is_available(): m.cuda()
    if ckpt and Path(ckpt).exists():
        load_state_dict(m, ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loaded: {ckpt}")
    toks = [TOKEN_IDS["BOS"], TOKEN_IDS["[AGENT]"], TOKEN_IDS["[PLUGIN]"]]
    print(sample(m, toks, max_new=16))

if __name__ == "__main__":
    main()
