#!/usr/bin/env python3
"""
CMU Capstone: Dense ViT Baseline (CIFAR-10)
Bridges-2 H100 GPU-ready
"""

import torch
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

print("\nDENSE ViT BASELINE")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True,
).to(device)

# Optional: warm-start from a saved baseline
try:
    model.load_state_dict(torch.load("vit-cifar10-h100.pth", map_location=device), strict=False)
    print("Loaded existing baseline weights")
except Exception:
    print("Training from pretrained weights")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# TRAIN
model.train()
for epoch in range(3):
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3 (baseline)")
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
print(f"\nBASELINE: {accuracy:.2f}% accuracy")

torch.save(model.state_dict(), "vit-baseline-h100.pth")
print("Saved: vit-baseline-h100.pth")
