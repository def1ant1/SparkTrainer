
import torch, torch.nn as nn
from model.modules import RotaryEmbedding, MoE
from model.config import CONF

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(CONF.d_model, CONF.d_model*3, bias=False)
        self.proj = nn.Linear(CONF.d_model, CONF.d_model)
        self.rope = RotaryEmbedding(CONF.d_model//CONF.n_heads)
        self.scale = (CONF.d_model//CONF.n_heads) ** -0.5

    def forward(self, x, mask=None):
        B,T,_ = x.size()
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q=q.view(B,T,CONF.n_heads,-1).transpose(1,2)
        k=k.view(B,T,CONF.n_heads,-1).transpose(1,2)
        v=v.view(B,T,CONF.n_heads,-1).transpose(1,2)
        q,k = self.rope(q,k)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        if mask is not None: attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).contiguous().view(B,T,CONF.d_model)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.norm1 = nn.LayerNorm(CONF.d_model)
        self.moe = MoE(CONF.d_model)
        self.norm2 = nn.LayerNorm(CONF.d_model)
        self.drop = nn.Dropout(CONF.dropout)
    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        moe_out, aux = self.moe(self.norm2(x))
        x = x + self.drop(moe_out)
        # attach aux onto x for upper layers to optionally read
        x._moe_aux = (getattr(x, "_moe_aux", 0.0) + aux) if isinstance(aux, torch.Tensor) else getattr(x, "_moe_aux", 0.0)
        return x
