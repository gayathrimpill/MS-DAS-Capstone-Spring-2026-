"""
Sparse MoE ViT (CIFAR-10)
Bridges-2 H100 GPU-ready
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
dataset = load_dataset("cifar10")


def collate_fn(batch):
    images = [item["img"].convert("RGB") for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = labels
    return inputs


train_loader = DataLoader(
    dataset["train"],
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    dataset["test"],
    batch_size=128,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


class Router(nn.Module):
    def __init__(self, dim, num_experts=8):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gates = torch.softmax(self.gate(x), -1)  # [B*T, 8]
        topk_gates, topk_idx = torch.topk(gates, 2, dim=-1)
        return topk_gates, topk_idx


class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8):
        super().__init__()
        self.router = Router(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        gates, idx = self.router(x_flat)
        output = torch.zeros_like(x_flat)

        for k in range(2):
            expert_mask = idx[:, k]
            for e_idx, expert in enumerate(self.experts):
                mask = expert_mask == e_idx
                if mask.any():
                    expert_out = expert(x_flat[mask])
                    output[mask] += expert_out * gates[mask, k].unsqueeze(-1)

        return output.view(b, t, d)


print("\nSPARSE MoE ViT (25% active params)")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True,
).to(device)

# Optional: start from dense baseline
try:
    model.load_state_dict(torch.load("vit-baseline-h100.pth", map_location=device), strict=False)
    print("Loaded baseline ? converting to MoE")
except Exception:
    print("No baseline weights found, using pretrained ViT")

dim = model.config.hidden_size
for i, layer in enumerate(model.vit.encoder.layer):
    layer.intermediate = MoELayer(dim).to(device)
    layer.output.dense = nn.Identity()
    layer.output.dropout = nn.Dropout(0.0)
    print(f"  Layer {i+1}/12: FFN ? MoE(8 experts)")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# TRAIN
model.train()
for epoch in range(3):
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3 (moe)")
    for batch in pbar:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"  Loss: {total_loss/len(train_loader):.4f}")

# EVAL
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Eval"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(**batch)
        pred = outputs.logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

accuracy = 100.0 * correct / total
print(f"\nSPARSE MoE: {accuracy:.2f}% accuracy")

torch.save(model.state_dict(), "vit-moe-h100.pth")
print("Saved: vit-moe-h100.pth")

print("\nCAPSTONE RESULTS (example target):")
print("Dense Baseline: 98.48% (100% active params)")
print(f"Sparse MoE:     {accuracy:.2f}% (25% active params)")
