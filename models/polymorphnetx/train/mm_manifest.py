
"""Multimodal JSONL Manifest Dataset for PolymorphNet-X (text + images + audio + video)."""
import os, json, warnings
from typing import Dict, List, Optional
from dataclasses import dataclass
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from model.config import CONF, TOKEN_IDS
from train.tokenizer import encode as byte_encode

USE_SPM = False
try:
    from train.spm_tokenizer import SentencePiecePNX
    import sentencepiece as _spm_probe  # noqa
    USE_SPM = os.path.exists("configs/spm.model")
except Exception:
    USE_SPM = False

def _tokenize(text: str) -> List[int]:
    return SentencePiecePNX().encode(text) if USE_SPM else byte_encode(text)

def _load_image(path: str, size: int = 256) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        warnings.warn(f"[MM] missing image: {path}"); return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: warnings.warn(f"[MM] failed to read image: {path}"); return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    arr = (img.astype(np.float32) / 255.0).transpose(2,0,1)
    return torch.from_numpy(arr)

def _load_audio(path: str) -> Optional[torch.Tensor]:
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=-1)
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        return torch.from_numpy(wav)
    except Exception as e:
        warnings.warn(f"[MM] audio load failed: {path} ({e})"); return None

def _sample_video_frames(path: str, max_frames: int = 16, size: int = 256) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        warnings.warn(f"[MM] missing video: {path}"); return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        warnings.warn(f"[MM] failed to open video: {path}"); return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total-1, num=min(max_frames, total), dtype=np.int32)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        arr = (frame.astype(np.float32) / 255.0).transpose(2,0,1)
        frames.append(arr)
    cap.release()
    if not frames: warnings.warn(f"[MM] no frames decoded: {path}"); return None
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr)

@dataclass
class MMItem:
    input_ids: List[int]
    images: Optional[List[torch.Tensor]]
    audio: Optional[List[torch.Tensor]]
    video: Optional[torch.Tensor]

class MMJsonlDataset(Dataset):
    def __init__(self, path: str, max_seq_len: int = CONF.max_seq_len,
                 image_limit: int = 4, audio_limit: int = 2, video_max_frames: int = 16,
                 image_size: int = 256):
        self.samples: List[MMItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt","")
                ids = _tokenize(prompt)[:max_seq_len-2]
                ids = [TOKEN_IDS["BOS"]] + ids + [TOKEN_IDS["EOS"]]

                img_lim = int(obj.get("image_limit", image_limit))
                aud_lim = int(obj.get("audio_limit", audio_limit))
                vid_frames = int(obj.get("video_max_frames", video_max_frames))

                imgs = []
                for p in obj.get("images", [])[:img_lim]:
                    t = _load_image(p, size=image_size)
                    if t is not None: imgs.append(t)
                imgs = imgs if imgs else None

                auds = []
                for p in obj.get("audio", [])[:aud_lim]:
                    t = _load_audio(p)
                    if t is not None: auds.append(t)
                auds = auds if auds else None

                vid = None
                if obj.get("video"):
                    vid = _sample_video_frames(obj["video"], max_frames=vid_frames, size=image_size)

                self.samples.append(MMItem(ids, imgs, auds, vid))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        it = self.samples[idx]
        x = torch.tensor(it.input_ids, dtype=torch.long)
        return {"input_ids": x[:-1], "labels": x[1:], "images": it.images, "audio": it.audio, "video": it.video}

def mm_collate(batch):
    pad_id = TOKEN_IDS["PAD"]
    max_len = max(item["input_ids"].shape[0] for item in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,  dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L]    = item["labels"]

    max_K_img = max(len(item["images"]) if item["images"] else 0 for item in batch)
    images = None
    if max_K_img > 0:
        H=W=None
        for item in batch:
            if item["images"]:
                H = item["images"][0].shape[1]
                W = item["images"][0].shape[2]
                break
        H = H or 256; W = W or 256
        images = torch.zeros(B, max_K_img, 3, H, W, dtype=torch.float32)
        for i, item in enumerate(batch):
            if not item["images"]: continue
            K = min(len(item["images"]), max_K_img)
            for k in range(K):
                images[i, k] = item["images"][k]

    max_K_aud = max(len(item["audio"]) if item["audio"] else 0 for item in batch)
    audio = None
    if max_K_aud > 0:
        max_T = max((a.shape[0] for item in batch for a in (item["audio"] or [])), default=0)
        audio = torch.zeros(B, max_K_aud, max_T, dtype=torch.float32)
        for i, item in enumerate(batch):
            if not item["audio"]: continue
            K = min(len(item["audio"]), max_K_aud)
            for k in range(K):
                wav = item["audio"][k]
                audio[i, k, :wav.shape[0]] = wav

    max_F = max((item["video"].shape[0] if isinstance(item["video"], torch.Tensor) else 0) for item in batch)
    video = None
    if max_F > 0:
        H=W=None
        for item in batch:
            if isinstance(item["video"], torch.Tensor):
                H = item["video"].shape[2]; W = item["video"].shape[3]; break
        H = H or 256; W = W or 256
        video = torch.zeros(B, max_F, 3, H, W, dtype=torch.float32)
        for i, item in enumerate(batch):
            v = item["video"]
            if v is None: continue
            F = min(v.shape[0], max_F)
            video[i, :F] = v[:F]

    return {"input_ids": input_ids, "labels": labels, "images": images, "audio": audio, "video": video}
