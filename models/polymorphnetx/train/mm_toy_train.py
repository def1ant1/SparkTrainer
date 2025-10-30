
import torch
from model.multimodal_model import PolymorphNetXMultimodal
from model.config import TOKEN_IDS

def build_toy_batch(B=2, T=16):
    input_ids = torch.full((B,T), TOKEN_IDS["BOS"], dtype=torch.long)
    input_ids[:,1] = TOKEN_IDS["[AGENT]"]; input_ids[:,2] = TOKEN_IDS["[PLUGIN]"]
    input_ids[:,3:] = torch.randint(30, 100, (B,T-3))
    images = torch.rand(B,3,256,256)
    audio  = torch.randn(B, 32000)
    labels = torch.cat([input_ids[:,1:], input_ids[:, :1]], dim=1)  # dummy next-token
    return {"input_ids": input_ids, "images": images, "audio": audio, "labels": labels}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolymorphNetXMultimodal().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in build_toy_batch().items()}
    logits = model(batch)
    # build extended labels to logits length
    pad = logits.new_full((batch["labels"].shape[0], logits.shape[1]-batch["labels"].shape[1]), -100, dtype=torch.long)
    labels = torch.cat([batch["labels"].to(device), pad], dim=1)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    loss.backward(); opt.step()
    print("toy multimodal OK, loss=", float(loss))

if __name__ == "__main__":
    main()
