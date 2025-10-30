
import json, random
from typing import Dict, List
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from model.config import CONF, TOKEN_IDS
from train.tokenizer import encode

@dataclass
class JsonlExample:
    input_ids: List[int]

class JsonlDataset(Dataset):
    def __init__(self, path: str, max_len: int = CONF.max_seq_len):
        self.samples: List[JsonlExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ids = encode(obj["prompt"])
                if len(ids) < 1: continue
                ids = [TOKEN_IDS["BOS"]] + ids[:max_len-2] + [TOKEN_IDS["EOS"]]
                self.samples.append(JsonlExample(ids))
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.samples[idx].input_ids
        x = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": x[:-1], "labels": x[1:]}

def collate(batch):
    pad_id = TOKEN_IDS["PAD"]
    max_len = max(item["input_ids"].shape[0] for item in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,  dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L]    = item["labels"]
    return {"input_ids": input_ids, "labels": labels}
