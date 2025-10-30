
import torch, torch.nn as nn
from model.model import PolymorphNetX, build_causal_mask
from model.config import TOKEN_IDS, CONF
from model.adapters import ImageAdapter, AudioAdapter

class VideoAdapter(nn.Module):
    def __init__(self, dim, patch=16):
        super().__init__()
        self.img = ImageAdapter(dim=dim, patch=patch)
        self.temporal = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
    def forward(self, frames):  # [B,F,3,H,W]
        B,F,C,H,W = frames.shape
        frames = frames.reshape(B*F, C, H, W)
        tokens = self.img(frames)              # [B*F, N, D]
        N = tokens.shape[1]
        tokens = tokens.view(B, F, N, -1).transpose(1,2) # [B,N,F,D]
        x = tokens.transpose(2,3).transpose(1,2)         # [B,D,N,F]
        x = self.temporal(x).transpose(1,2).transpose(2,3)  # [B,N,F,D]
        fused = x.mean(dim=2)                   # [B,N,D]
        return fused

class PolymorphNetXMultimodalV2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.core = PolymorphNetX()
        self.img = ImageAdapter(dim=dim)
        self.aud = AudioAdapter(dim=dim)
        self.vid = VideoAdapter(dim=dim)
        self.drop = nn.Dropout(0.1)

    def _wrap_embed(self, B, start_id, end_id, mid_tokens):
        start = self.core.tok_emb.weight[start_id][None,None,:].expand(B,1,-1)
        end   = self.core.tok_emb.weight[end_id][None,None,:].expand(B,1,-1)
        return torch.cat([start, mid_tokens, end], dim=1)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        B = input_ids.size(0)

        x = self.core.tok_emb(input_ids)  # [B,T,D]
        chunks = [x]

        if batch.get("images") is not None:
            imgs = batch["images"]  # [B,K,3,H,W]
            B,K,_,H,W = imgs.shape
            imgs_f = imgs.view(B*K, 3, H, W)
            img_tokens = self.img(imgs_f)             # [B*K, N, D]
            N = img_tokens.shape[1]
            img_tokens = img_tokens.view(B, K*N, -1)
            chunks.append(self._wrap_embed(B, TOKEN_IDS["[IMG]"], TOKEN_IDS["[/IMG]"], img_tokens))

        if batch.get("audio") is not None:
            aud = batch["audio"]  # [B,K,T]
            B,K,T = aud.shape
            aud_f = aud.view(B*K, T)
            aud_tokens = self.aud(aud_f)              # [B*K, N, D]
            N = aud_tokens.shape[1]
            aud_tokens = aud_tokens.view(B, K*N, -1)
            chunks.append(self._wrap_embed(B, TOKEN_IDS["[AUD]"], TOKEN_IDS["[/AUD]"], aud_tokens))

        if batch.get("video") is not None:
            vid = batch["video"]  # [B,F,3,H,W]
            vid_tokens = self.vid(vid)                # [B, N, D]
            chunks.append(self._wrap_embed(B, TOKEN_IDS["[IMG]"], TOKEN_IDS["[/IMG]"], vid_tokens))

        X = torch.cat(chunks, dim=1)
        T = X.size(1)
        mask = build_causal_mask(T, X.device)
        for blk in self.core.blocks:
            X = blk(X, mask)
        return self.core.lm_head(self.core.norm(self.drop(X)))
