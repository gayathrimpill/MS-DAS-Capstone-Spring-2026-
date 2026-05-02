#!/usr/bin/env python3
"""
Fast-Track MoE Hyperparameter Experiments on CIFAR-100
Tests key configurations efficiently
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time

device = torch.device("cuda")
print(f"\nGPU: {torch.cuda.get_device_name(0)}\n")

# Load CIFAR-100
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
dataset = load_dataset("cifar100")

def collate_fn(batch):
    images = [item["img"].convert("RGB") for item in batch]
    labels = torch.tensor([item["fine_label"] for item in batch])
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = labels
    return inputs

train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size=64, shuffle=False, collate_fn=collate_fn)

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, router_temp=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temp = router_temp
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ) for _ in range(num_experts)
        ])
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        
        logits = self.gate(x_flat)
        gates = torch.softmax(logits / self.router_temp, -1)
        
        topk_gates, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)
        
        output = torch.zeros_like(x_flat)
        
        if self.training:
            for i in range(self.num_experts):
                self.expert_usage[i] += (topk_idx == i).sum().item()
        
        for k in range(self.top_k):
            for e_idx, expert in enumerate(self.experts):
                mask = topk_idx[:, k] == e_idx
                if mask.any():
                    output[mask] += expert(x_flat[mask]) * topk_gates[mask, k].unsqueeze(-1)
        
        return output.view(b, t, d)

def create_moe_model(num_experts=8, top_k=2, router_temp=1.0):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        num_labels=100,  # CIFAR-100 has 100 classes
        ignore_mismatched_sizes=True
    ).to(device)
    
    dim = model.config.hidden_size
    for i in range(12):
        layer = model.vit.encoder.layer[i]
        layer.intermediate = MoELayer(dim, num_experts, top_k, router_temp).to(device)
        layer.output.dense = nn.Identity()
    
    sparsity = 100 * (1 - top_k / num_experts)
    return model, sparsity

def train_and_eval(model, name, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    
    start_time = time.time()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred = model(**batch).logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    accuracy = 100.0 * correct / total
    elapsed_time = (time.time() - start_time) / 60
    
    # Get expert usage stats
    expert_usage = []
    total_usage = torch.zeros(model.vit.encoder.layer[0].intermediate.num_experts)
    
    for i, layer in enumerate(model.vit.encoder.layer):
        if isinstance(layer.intermediate, MoELayer):
            usage = layer.intermediate.expert_usage.cpu()
            total_usage += usage
    
    total_usage = total_usage / (total_usage.sum() + 1e-9)
    entropy = -(total_usage * torch.log(total_usage + 1e-9)).sum().item()
    max_usage = total_usage.max().item()
    
    return {
        'accuracy': accuracy,
        'losses': losses,
        'routing_entropy': entropy,
        'max_expert_share': max_usage * 100,
        'time_minutes': elapsed_time
    }

# EXPERIMENTS
print("="*70)
print("FAST-TRACK MOE EXPERIMENTS ON CIFAR-100")
print("="*70)

results = {}
start_total = time.time()

# EXPERIMENT 1: Baseline MoE (8 experts, top-2)
print("\n" + "="*70)
print("EXPERIMENT 1/4: Baseline MoE")
print("Config: 8 experts, top_k=2, temp=1.0")
print("Sparsity: 75.0%")
print("="*70 + "\n")

model, sparsity = create_moe_model(num_experts=8, top_k=2, router_temp=1.0)
results['baseline_8exp_k2_t1.0'] = train_and_eval(model, 'baseline_8exp_k2_t1.0')
res = results['baseline_8exp_k2_t1.0']
print(f"\n✓ baseline_8exp_k2_t1.0: {res['accuracy']:.2f}%")
print(f"  Routing entropy: {res['routing_entropy']:.3f}")
print(f"  Max expert share: {res['max_expert_share']:.1f}%")
print(f"  Time: {res['time_minutes']:.1f} min")

# Clear GPU memory
del model
torch.cuda.empty_cache()

elapsed = (time.time() - start_total) / 60
eta = elapsed * 3  # Estimate based on first experiment
print(f"\nProgress: 1/4 complete | Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

# EXPERIMENT 2: Ultra-sparse (8 experts, top-1)
print("\n" + "="*70)
print("EXPERIMENT 2/4: Ultra-sparse (12.5% active)")
print("Config: 8 experts, top_k=1, temp=1.0")
print("Sparsity: 87.5%")
print("="*70 + "\n")

model, sparsity = create_moe_model(num_experts=8, top_k=1, router_temp=1.0)
results['ultrasparse_8exp_k1_t1.0'] = train_and_eval(model, 'ultrasparse_8exp_k1_t1.0')
res = results['ultrasparse_8exp_k1_t1.0']
print(f"\n✓ ultrasparse_8exp_k1_t1.0: {res['accuracy']:.2f}%")
print(f"  Routing entropy: {res['routing_entropy']:.3f}")
print(f"  Max expert share: {res['max_expert_share']:.1f}%")
print(f"  Time: {res['time_minutes']:.1f} min")

del model
torch.cuda.empty_cache()

elapsed = (time.time() - start_total) / 60
eta = elapsed * 2  # 2 more to go
print(f"\nProgress: 2/4 complete | Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

# EXPERIMENT 3: Sharp routing (temp=0.5)
print("\n" + "="*70)
print("EXPERIMENT 3/4: Sharp routing")
print("Config: 8 experts, top_k=2, temp=0.5")
print("Sparsity: 75.0%")
print("="*70 + "\n")

model, sparsity = create_moe_model(num_experts=8, top_k=2, router_temp=0.5)
results['sharp_8exp_k2_t0.5'] = train_and_eval(model, 'sharp_8exp_k2_t0.5')
res = results['sharp_8exp_k2_t0.5']
print(f"\n✓ sharp_8exp_k2_t0.5: {res['accuracy']:.2f}%")
print(f"  Routing entropy: {res['routing_entropy']:.3f}")
print(f"  Max expert share: {res['max_expert_share']:.1f}%")
print(f"  Time: {res['time_minutes']:.1f} min")

del model
torch.cuda.empty_cache()

elapsed = (time.time() - start_total) / 60
eta = elapsed / 3  # 1 more to go
print(f"\nProgress: 3/4 complete | Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

# EXPERIMENT 4: Soft routing (temp=2.0)
print("\n" + "="*70)
print("EXPERIMENT 4/4: Soft routing")
print("Config: 8 experts, top_k=2, temp=2.0")
print("Sparsity: 75.0%")
print("="*70 + "\n")

model, sparsity = create_moe_model(num_experts=8, top_k=2, router_temp=2.0)
results['soft_8exp_k2_t2.0'] = train_and_eval(model, 'soft_8exp_k2_t2.0')
res = results['soft_8exp_k2_t2.0']
print(f"\n✓ soft_8exp_k2_t2.0: {res['accuracy']:.2f}%")
print(f"  Routing entropy: {res['routing_entropy']:.3f}")
print(f"  Max expert share: {res['max_expert_share']:.1f}%")
print(f"  Time: {res['time_minutes']:.1f} min")

del model
torch.cuda.empty_cache()

# SUMMARY
total_time = (time.time() - start_total) / 60
print("\n" + "="*70)
print("EXPERIMENTS COMPLETE!")
print("="*70)
print(f"Total time: {total_time:.1f} minutes")

print("\n" + "="*70)
print("SUMMARY - CIFAR-100 MoE HYPERPARAMETER STUDY")
print("="*70)
print(f"{'Config':<30} {'Accuracy':<12} {'Entropy':<12} {'Max Expert':<12}")
print("-"*70)
for name, res in results.items():
    print(f"{name:<30} {res['accuracy']:>10.2f}% {res['routing_entropy']:>10.3f} {res['max_expert_share']:>10.1f}%")

# Save results
with open('cifar100_fast_track_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: cifar100_fast_track_results.json")
print("\nNext steps:")
print("1. Compare with CIFAR-10 results")
print("2. Analyze task complexity effects")
print("3. Generate comparison plots")
