
import torch, torch.nn as nn
from model.config import CONF
from model.dag_transformer import TransformerBlock
from model.modules import PolicyHead

def build_causal_mask(T: int, device):
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

class ModalityTypeEmbedding(nn.Module):
    """Adds a small learned bias per token type: 0=text, 1=image, 2=audio (video uses 1)."""
    def __init__(self, dim=CONF.d_model, num_types=3):
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)
    def forward(self, x, t):
        return x + self.emb(t)

class PolymorphNetX(nn.Module):
    def __init__(self, grad_checkpoint: bool = False):
        super().__init__()
        self.tok_emb = nn.Embedding(CONF.vocab_size, CONF.d_model)
        self.pos_drop = nn.Dropout(CONF.dropout)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(CONF.n_layers)])
        self.norm = nn.LayerNorm(CONF.d_model)
        self.lm_head = PolicyHead(CONF.d_model, CONF.vocab_size)
        self.modtype = ModalityTypeEmbedding()
        self.grad_checkpoint = grad_checkpoint

    def forward(self, idx, mask=None, types=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.pos_drop(x)
        if types is None:
            types = torch.zeros(B, T, dtype=torch.long, device=idx.device)
        x = self.modtype(x, types)
        if mask is None:
            mask = build_causal_mask(T, x.device)
        for blk in self.blocks:
            if self.grad_checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, mask)
            else:
                x = blk(x, mask)
        return self.lm_head(self.norm(x))
