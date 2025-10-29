import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from spark_trainer.models.gating import MoELoRA


class TinySeqDataset(Dataset):
    """Loads a tiny sequence classification dataset for quick tests.

    Each row stores 8 features and a binary label. We reshape features to
    [seq_len=2, d_model=4] to feed the MoE layer.
    """

    def __init__(self, path: str):
        self.items = []
        for line in Path(path).read_text().splitlines():
            obj = json.loads(line)
            self.items.append((obj["features"], int(obj["label"])) )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feats, label = self.items[idx]
        x = torch.tensor(feats, dtype=torch.float32).view(2, 4)  # [S=2, D=4]
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class ToyMoEClassifier(nn.Module):
    """A minimal classifier using the new MoE gating (LoRA experts)."""

    def __init__(self, d_model=4, d_ff=16, num_experts=3, num_selected=2):
        super().__init__()
        self.moe = MoELoRA(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            num_selected=num_selected,
            lora_rank=2,
            lora_alpha=4,
            capacity_factor=1.25,
        )
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: [B, S, D]
        out, _metrics = self.moe(x)
        pooled = out.mean(dim=1)  # [B, D]
        return self.head(pooled)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_tiny_moe_training_run(tmp_path):
    torch.manual_seed(0)

    data_path = Path("tests/data/tiny_sequence_classification.jsonl")
    assert data_path.exists(), "Expected tiny sequence dataset to exist"

    ds = TinySeqDataset(str(data_path))
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    model = ToyMoEClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def get_loss():
        model.train()
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb  # [B, 2, 4]
            xb = xb.view(xb.size(0), 2, 4)  # Ensure [B,S,D]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
            # Keep it quick
            if n >= 16:
                break
        return total / max(1, n)

    # Run two mini-epochs and ensure loss is finite and parameters update
    with torch.no_grad():
        initial_params = [p.clone() for p in model.parameters()]

    loss1 = get_loss()
    loss2 = get_loss()

    assert math.isfinite(loss1) and math.isfinite(loss2)

    # Some parameter must have changed
    changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_params, model.parameters()))
    assert changed, "Model parameters did not update during training"
