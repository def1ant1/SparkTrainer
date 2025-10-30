
import math, torch
from model.config import TOKEN_IDS

@torch.no_grad()
def compute_metrics(logits, labels):
    B,T,V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="mean"
    )
    ppl = math.exp(loss.item()) if loss.isfinite() else float("inf")

    preds = logits.argmax(-1)
    mask = labels != -100
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item() + 1e-9
    acc = correct / total

    ctrl = ["[AGENT]","[MEM]","[PLUGIN]","[VERIFY]","[POLICY]","[IMG]","[/IMG]","[AUD]","[/AUD]"]
    ctrl_ids = [TOKEN_IDS[c] for c in ctrl if c in TOKEN_IDS]
    ctrl_ids = torch.tensor(ctrl_ids, device=preds.device)
    pred_ctrl = ((preds.unsqueeze(-1) == ctrl_ids).any(-1)) & mask
    true_ctrl = ((labels.unsqueeze(-1) == ctrl_ids).any(-1)) & mask

    tp = (pred_ctrl & true_ctrl).sum().item()
    fp = (pred_ctrl & ~true_ctrl).sum().item()
    fn = (~pred_ctrl & true_ctrl).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2*precision*recall / (precision + recall + 1e-9)

    return {"loss": loss.item(), "perplexity": ppl, "token_acc": acc, "control_f1": f1}
