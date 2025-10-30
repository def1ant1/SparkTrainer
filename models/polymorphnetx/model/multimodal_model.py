
import torch, torch.nn as nn
from typing import Dict, Optional
from model.model import PolymorphNetX, build_causal_mask
from model.config import CONF, TOKEN_IDS
from model.adapters import ImageAdapter, AudioAdapter

class PolymorphNetXMultimodal(nn.Module):
    """
    Accepts:
      - input_ids: [B,T]
      - images:    [B,3,H,W] (optional)
      - audio:     [B,T_audio] (optional)
      - video:     [B,F,3,H,W] (optional)  -- F sampled frames
    Concatenates text embeddings with adapters' token embeddings wrapped by control tokens.
    """
    def __init__(self):
        super().__init__()
        self.core = PolymorphNetX()
        self.img = ImageAdapter()
        self.aud = AudioAdapter()
        self.img_proj = nn.Identity()
        self.aud_proj = nn.Identity()

    def _wrap(self, B, start_id, end_id, mid_tokens):
        start = self.core.tok_emb.weight[start_id][None,None,:].expand(B,1,-1)
        end   = self.core.tok_emb.weight[end_id][None,None,:].expand(B,1,-1)
        return torch.cat([start, mid_tokens, end], dim=1)

    def _video_tokens(self, video: torch.Tensor) -> torch.Tensor:
        # video: [B,F,3,H,W] -> process each frame through ImageAdapter, concat on token dimension
        B, F, C, H, W = video.shape
        frames = video.view(B*F, C, H, W)
        tok = self.img_proj(self.img(frames))     # [B*F, Ntok, D]
        tok = tok.view(B, F, -1, tok.size(-1))    # [B,F,Ntok,D]
        tok = tok.flatten(1,2)                    # [B, F*Ntok, D]
        return tok

    def forward(self, batch: Dict[str, torch.Tensor]):
        input_ids: torch.Tensor = batch["input_ids"]
        B = input_ids.size(0)
        device = input_ids.device

        x_txt = self.core.tok_emb(input_ids)
        seqs = [x_txt]

        if "images" in batch and batch["images"] is not None:
            img_tokens = self.img_proj(self.img(batch["images"]))
            seqs.append(self._wrap(B, TOKEN_IDS["[IMG]"], TOKEN_IDS["[/IMG]"], img_tokens))

        if "video" in batch and batch["video"] is not None:
            vid_tokens = self._video_tokens(batch["video"])
            seqs.append(self._wrap(B, TOKEN_IDS["[IMG]"], TOKEN_IDS["[/IMG]"], vid_tokens))

        if "audio" in batch and batch["audio"] is not None:
            aud_tokens = self.aud_proj(self.aud(batch["audio"]))
            seqs.append(self._wrap(B, TOKEN_IDS["[AUD]"], TOKEN_IDS["[/AUD]"], aud_tokens))

        x = torch.cat(seqs, dim=1)
        # modality type ids aligned with seqs
        types = []
        for s in seqs:
            types.append(torch.zeros(s.size(0), s.size(1), dtype=torch.long, device=device))
        if len(seqs) > 1 and (batch.get('images') is not None or batch.get('video') is not None):
            types[1] = types[1].fill_(1)
        if len(seqs) > 2 and (batch.get('audio') is not None):
            types[2] = types[2].fill_(2)
        types_all = torch.cat(types, dim=1)
        T = x.size(1)
        mask = build_causal_mask(T, device)
        for blk in self.core.blocks:
            x = blk(x, mask)
        return self.core.lm_head(self.core.norm(x))
