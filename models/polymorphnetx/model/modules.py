
import torch, torch.nn as nn
import torch.nn.functional as F
from model.config import CONF

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=CONF.max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :])
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def forward(self, q, k):
        cos, sin = self.cos[:, :, : q.size(-2), :], self.sin[:, :, : q.size(-2), :]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class MoE(nn.Module):
    """
    Token-level Top-K routing with auxiliary load-balance loss.
    """
    def __init__(self, dim, n_experts=CONF.n_experts, top_k=CONF.top_k, capacity_factor: float = 1.25):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.experts = nn.ModuleList([FeedForward(dim, CONF.expert_ffn_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B*T, D)
        logits = self.gate(x_flat)                     # [BT, E]
        scores = F.softmax(logits, dim=-1)

        top_scores, top_idx = torch.topk(scores, k=self.top_k, dim=-1)  # [BT,K], [BT,K]

        # Aux load-balance loss
        me = scores.mean(0)                       # [E]
        # Proxy for assignment proportion
        assign = torch.zeros_like(me)
        for e in range(self.n_experts):
            assign[e] = (top_idx == e).float().any(dim=-1).float().mean()
        aux_loss = (me * assign * self.n_experts).sum() * 0.01

        cap = int(self.capacity_factor * (B*T) / self.n_experts) + 1

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask_e = (top_idx == e)
            if not mask_e.any():
                continue
            pos = mask_e.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            pos = pos[:cap]
            x_e = x_flat.index_select(0, pos)
            # take max gate weight per token for this expert
            w_e = top_scores[pos, mask_e[pos].float().argmax(dim=-1)]
            y_e = self.experts[e](x_e) * w_e.unsqueeze(-1)
            out.index_copy_(0, pos, out.index_select(0, pos) + y_e)

        return out.reshape(B, T, D), aux_loss

class PolicyHead(nn.Module):
    def __init__(self, dim, vocab=CONF.vocab_size):
        super().__init__()
        self.out = nn.Linear(dim, vocab, bias=False)
    def forward(self, x, mask=False):
        logits = self.out(x)
        if mask:
            logits[..., CONF.special_tokens["[POLICY]"]] += 10.0
        return logits
