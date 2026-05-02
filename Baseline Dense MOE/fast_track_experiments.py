#!/usr/bin/env python3
"""
Fast-Track MoE Experiments (4 configs, ~2.5 hours)
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
print(f"GPU: {torch.cuda.get_device_name(0)}\n")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
dataset = load_dataset("cifar10")

def collate_fn(batch):
    images = [item["img"].convert("RGB") for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = labels
    return inputs

train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size=64, shuffle=False, collate_fn=collate_fn)

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, temp=1.0):
        super().__init__()
        self.num_experts, self.top_k, self.temp = num_experts, top_k, temp
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)) 
            for _ in range(num_experts)
        ])
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        logits = self.gate(x_flat)
        gates = torch.softmax(logits / self.temp, -1)
        topk_gates, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)
        
        if self.training:
            for i in range(self.num_experts):
                self.expert_usage[i] += (topk_idx == i).sum().item()
        
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e_idx, expert in enumerate(self.experts):
                mask = topk_idx[:, k] == e_idx
                if mask.any():
                    output[mask] += expert(x_flat[mask]) * topk_gates[mask, k].unsqueeze(-1)
        return output.view(b, t, d)

def run_experiment(num_experts, top_k, temp, epochs=3):
    start = time.time()
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", num_labels=10, ignore_mismatched_sizes=True
    ).to(device)
    
    dim = model.config.hidden_size
    for layer in model.vit.encoder.layer:
        layer.intermediate = MoELayer(dim, num_experts, top_k, temp).to(device)
        layer.output.dense = nn.Identity()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"E{epoch+1}/{epochs}"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            correct += (model(**batch).logits.argmax(-1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    accuracy = 100.0 * correct / total
    
    # Get expert usage
    total_usage = torch.zeros(num_experts)
    for layer in model.vit.encoder.layer:
        if isinstance(layer.intermediate, MoELayer):
            total_usage += layer.intermediate.expert_usage.cpu()
    
    total_usage = total_usage / (total_usage.sum() + 1e-9)
    entropy = -(total_usage * torch.log(total_usage + 1e-9)).sum().item()
    max_share = total_usage.max().item()
    
    elapsed = time.time() - start
    
    return {
        'accuracy': accuracy,
        'losses': losses,
        'routing_entropy': entropy,
        'max_expert_share': max_share,
        'expert_usage': total_usage.tolist(),
        'time_seconds': elapsed
    }

# EXPERIMENTS
print("="*70)
print("FAST-TRACK MOE EXPERIMENTS")
print("="*70)

experiments = [
    {'name': 'baseline_8exp_k2_t1.0', 'experts': 8, 'k': 2, 'temp': 1.0, 'desc': 'Baseline MoE'},
    {'name': 'ultrasparse_8exp_k1_t1.0', 'experts': 8, 'k': 1, 'temp': 1.0, 'desc': 'Ultra-sparse (12.5% active)'},
    {'name': 'sharp_8exp_k2_t0.5', 'experts': 8, 'k': 2, 'temp': 0.5, 'desc': 'Sharp routing'},
    {'name': 'soft_8exp_k2_t2.0', 'experts': 8, 'k': 2, 'temp': 2.0, 'desc': 'Soft routing'},
]

results = {}
total_start = time.time()

for i, exp in enumerate(experiments, 1):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {i}/4: {exp['desc']}")
    print(f"Config: {exp['experts']} experts, top_k={exp['k']}, temp={exp['temp']}")
    print(f"Sparsity: {100*(1 - exp['k']/exp['experts']):.1f}%")
    print(f"{'='*70}\n")
    
    results[exp['name']] = run_experiment(exp['experts'], exp['k'], exp['temp'])
    results[exp['name']]['config'] = {
        'num_experts': exp['experts'],
        'top_k': exp['k'],
        'router_temp': exp['temp'],
        'sparsity': 100*(1 - exp['k']/exp['experts'])
    }
    
    print(f"\n✓ {exp['name']}: {results[exp['name']]['accuracy']:.2f}%")
    print(f"  Routing entropy: {results[exp['name']]['routing_entropy']:.3f}")
    print(f"  Max expert share: {results[exp['name']]['max_expert_share']:.1%}")
    print(f"  Time: {results[exp['name']]['time_seconds']/60:.1f} min")
    
    # Save incremental results
    with open('moe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    elapsed_total = (time.time() - total_start) / 60
    remaining = len(experiments) - i
    eta = elapsed_total / i * remaining if i > 0 else 0
    print(f"\nProgress: {i}/{len(experiments)} complete | Elapsed: {elapsed_total:.1f}min | ETA: {eta:.1f}min\n")

total_time = (time.time() - total_start) / 60

# Final summary
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"{'Config':<30} {'Accuracy':<12} {'Entropy':<10} {'Max Share':<12} {'Active %':<10}")
print("-"*70)
for name, res in results.items():
    cfg = res['config']
    print(f"{name:<30} {res['accuracy']:>10.2f}% {res['routing_entropy']:>8.3f} "
          f"{res['max_expert_share']:>10.1%} {100*cfg['top_k']/cfg['num_experts']:>8.1f}%")

print(f"\n✓ Total time: {total_time:.1f} minutes")
print(f"✓ Results saved to: moe_results.json")
print("\nKey findings:")
print("1. Compare baseline vs ultra-sparse: does k=1 work?")
print("2. Compare temp=0.5 vs temp=1.0 vs temp=2.0: routing collapse?")
print("3. Check entropy and max_expert_share: is routing balanced?")
