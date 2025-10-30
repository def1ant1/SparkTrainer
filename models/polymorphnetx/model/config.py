
from dataclasses import dataclass, field
from typing import Dict

TOKEN_IDS: Dict[str, int] = {
    "PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3,
    "[AGENT]": 10, "[MEM]": 11, "[PLUGIN]": 12, "[VERIFY]": 13,
    "[PRECISION]": 14, "[NODE]": 15, "[POLICY]": 16,
    "[IMG]": 17, "[/IMG]": 18, "[AUD]": 19, "[/AUD]": 20,
}

@dataclass
class ModelConfig:
    d_model:int=12288
    n_layers:int=48
    n_heads:int=64
    vocab_size:int=131072          # leaves headroom for SPM vocab + specials
    max_seq_len:int=8192
    dropout:float=0.1
    n_experts:int=64
    top_k:int=4
    expert_ffn_dim:int=65536
    mem_heads:int=16
    policy_heads:int=4
    enable_flash:bool=True
    sparsity_prob:float=0.2
    lr:float=3e-4
    warmup_steps:int=2000
    total_steps:int=2000
    weight_decay:float=0.1
    fp16:bool=True
    special_tokens:Dict[str,int]=field(default_factory=lambda:TOKEN_IDS)

CONF = ModelConfig()
