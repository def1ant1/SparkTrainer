
from typing import List
import sentencepiece as spm
from model.config import CONF, TOKEN_IDS

SPM_PATH = "configs/spm.model"
BASE_OFFSET = max(TOKEN_IDS.values()) + 1

class SentencePiecePNX:
    def __init__(self, model_path: str = SPM_PATH):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab = self.sp.vocab_size()

    def encode(self, text: str) -> List[int]:
        ids = []
        pieces = self.sp.encode(text, out_type=str)
        for p in pieces:
            if p in TOKEN_IDS:
                ids.append(TOKEN_IDS[p])
            elif p == "<s>":
                ids.append(TOKEN_IDS["BOS"])
            elif p == "</s>":
                ids.append(TOKEN_IDS["EOS"])
            elif p == "<unk>":
                ids.append(TOKEN_IDS["UNK"])
            else:
                spm_id = self.sp.piece_to_id(p)
                remap = BASE_OFFSET + spm_id
                if remap >= CONF.vocab_size:
                    remap = CONF.vocab_size - 1
                ids.append(remap)
        return ids[:CONF.max_seq_len]

    def decode(self, ids: List[int]) -> str:
        pieces = []
        for i in ids:
            # Map back: specials -> literal tokens, others -> spm pieces
            if i in TOKEN_IDS.values():
                # use token string for clarity
                for k,v in TOKEN_IDS.items():
                    if v == i: pieces.append(k); break
            else:
                spm_id = i - BASE_OFFSET
                if 0 <= spm_id < self.vocab:
                    pieces.append(self.sp.id_to_piece(spm_id))
        return self.sp.decode(pieces)
