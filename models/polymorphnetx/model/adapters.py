
import torch, torch.nn as nn
from model.config import CONF

class ImageAdapter(nn.Module):
    def __init__(self, dim=CONF.d_model, patch=16, in_ch=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    def forward(self, x):  # [B,3,H,W] in [0,1]
        B = x.size(0)
        feats = self.proj(x)                               # [B,D,h,w]
        tokens = feats.flatten(2).transpose(1,2)           # [B,N,D]
        cls = self.cls.expand(B, -1, -1)
        return torch.cat([cls, tokens], dim=1)

class AudioAdapter(nn.Module):
    def __init__(self, dim=CONF.d_model, n_fft=512, hop=256):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.proj = nn.Conv1d(n_fft//2 + 1, dim, kernel_size=3, stride=2, padding=1)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    def _stft(self, wav):
        window = torch.hann_window(self.n_fft, device=wav.device)
        stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, window=window, return_complex=True)
        mag = stft.abs() + 1e-6
        logmag = mag.log()
        return logmag  # [B,F,Frames]
    def forward(self, wav):  # [B,T] in [-1,1]
        B = wav.size(0)
        spec = self._stft(wav)
        feats = self.proj(spec)                            # [B,D,Frames/2]
        tokens = feats.transpose(1,2)                      # [B,N,D]
        cls = self.cls.expand(B, -1, -1)
        return torch.cat([cls, tokens], dim=1)
