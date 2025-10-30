
from typing import List
from model.config import CONF, TOKEN_IDS

BYTE_OFFSET = max(TOKEN_IDS.values()) + 1

def encode(text: str) -> List[int]:
    ids = []
    for tok in text.split():
        if tok in TOKEN_IDS:
            ids.append(TOKEN_IDS[tok])
        else:
            for b in tok.encode("utf-8"):
                ids.append(BYTE_OFFSET + b)
    return ids[:CONF.max_seq_len]

def decode(ids: List[int]) -> str:
    out = bytearray()
    for i in ids:
        if i in TOKEN_IDS.values(): continue
        b = i - BYTE_OFFSET
        if 0 <= b < 256: out.append(b)
    return out.decode("utf-8", errors="ignore")
