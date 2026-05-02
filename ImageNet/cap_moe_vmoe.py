"""
cap_moe_vmoe.py
---------------
Training script for V-MoE-style ViT-B/16 on ImageNet (ILSVRC-2012).

V-MoE changes applied vs. the original cap_moe.py
===================================================
1. Dataset        -> ImageNet-1k (1 000 classes) via HuggingFace datasets,
                     cached at HF_CACHE_DIR (default: cis260039p allocation).
                     Uses the official train/validation splits.
2. MoE placement  -> Experts only on ODD-indexed transformer layers (0-based:
                     layers 1, 3, 5, 7, 9, 11), matching the V-MoE-B/16 config.
                     Even layers keep the original dense MLP.
3. Router module  -> VMoEFeedForward with buffer-token routing (see vit_moe_vmoe.py).
4. Aux-loss coef  -> 0.01  (paper value; original was 0.1, too strong).
5. Overflow log   -> wandb logs overflow_fraction per step.
6. RandAugment    -> Added to training transforms (V-MoE uses it for ImageNet).
7. Higher LR      -> Default 3e-4; cosine decay with 5% warmup.
8. AMP            -> bfloat16 autocast.

Extra wandb metrics
===================
Per step : grad_norm, top1_train_acc, loss/task, loss/aux, loss/total,
           expert_load/layer{i}/aux_loss
Per epoch: val_accuracy, val_top5_acc, val_loss,
           macro_f1, macro_auroc, macro_auprc, epoch_overflow
Final    : final_accuracy, final_top5_acc, final_macro_f1,
           final_auroc, final_auprc
"""

import os
import gc
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from transformers import (
    ViTForImageClassification,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from vit_moe_vmoe import VMoEFeedForward

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# HuggingFace cache dir -- where imagenet-1k parquet files are stored
# ---------------------------------------------------------------------------
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR",
    "/ocean/projects/cis250163p/tnair/Imagenet_1k",
)

# Model cache (ViT weights)
MODEL_CACHE_DIR = os.environ.get(
    "HF_HOME",
    "/ocean/projects/cis250163p/tnair/hf_cache",
)

# AUROC/AUPRC computed over a random subset of classes (one-vs-rest).
# 1000 sampled classes is representative and keeps runtime tractable.
AUROC_CLASS_SAMPLE = 1000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int):
    """
    ImageNet-1k dataloaders using HuggingFace datasets (parquet format).
    Uses the official train and validation splits.
    """
    print(f"[Data] Loading imagenet-1k from cache: {HF_CACHE_DIR}")
    dataset = load_dataset("imagenet-1k", cache_dir=HF_CACHE_DIR, token = True)

    # V-MoE training augmentation
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def train_collate(batch):
        pixel_values = torch.stack(
            [train_transform(item["image"].convert("RGB")) for item in batch]
        )
        labels = torch.tensor([item["label"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    def val_collate(batch):
        pixel_values = torch.stack(
            [val_transform(item["image"].convert("RGB")) for item in batch]
        )
        labels = torch.tensor([item["label"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    print(f"[Data] Train: {len(dataset['train']):,} images | "
          f"Val: {len(dataset['validation']):,} images")

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class ViTOutputPassthrough(nn.Module):
    """Drop-in replacement for ViTOutput -- adds residual only."""
    def forward(self, hidden_states, input_tensor):
        return hidden_states + input_tensor


def build_vmoe_vit(
    num_experts: int = 8,
    top_k: int = 1,
    capacity_factor: float = 1.25,
    num_buffer_tokens: int = 8,
    cache_dir: Optional[str] = None,
):
    cache_dir = cache_dir or MODEL_CACHE_DIR

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=1000,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )

    hidden       = model.config.hidden_size        # 768
    intermediate = model.config.intermediate_size  # 3072

    num_layers = len(model.vit.encoder.layer)
    moe_layers = [i for i in range(num_layers) if i % 2 == 1]  # 1,3,5,7,9,11
    print(f"[V-MoE] Placing experts on layers: {moe_layers}")

    for i, layer in enumerate(model.vit.encoder.layer):
        if i in moe_layers:
            layer.intermediate = VMoEFeedForward(
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                num_buffer_tokens=num_buffer_tokens,
            )
            layer.output = ViTOutputPassthrough()

    return model


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, val_loader, criterion, num_classes: int = 1000):
    """
    Run a full validation pass and return a dict of metrics:
        val_loss, val_top1_acc, val_top5_acc,
        macro_f1, macro_auroc, macro_auprc

    AUROC and AUPRC are macro-averaged over AUROC_CLASS_SAMPLE randomly
    sampled classes (one-vs-rest) to keep memory and runtime tractable.
    """
    model.eval()

    total_loss = 0.0
    correct1   = 0
    correct5   = 0
    total      = 0
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            batch  = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in batch.items()}
            labels = batch["labels"]

            outputs = model(**batch)
            logits  = outputs.logits                        # (B, C)
            loss    = criterion(logits, labels)
            total_loss += loss.item()

            # Top-1
            preds1    = logits.argmax(-1)
            correct1 += (preds1 == labels).sum().item()

            # Top-5
            top5      = logits.topk(5, dim=-1).indices
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

            probs = F.softmax(logits.float(), dim=-1).cpu()
            all_probs.append(probs)
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(val_loader)
    top1_acc = 100.0 * correct1 / total
    top5_acc = 100.0 * correct5 / total

    all_labels_np = torch.cat(all_labels).numpy()
    all_probs_np  = torch.cat(all_probs,  dim=0).numpy()
    all_preds_np  = all_probs_np.argmax(axis=1)

    macro_f1 = f1_score(
        all_labels_np, all_preds_np, average="macro", zero_division=0
    )

    
    present_classes = np.unique(all_labels_np).tolist()
    # All classes, no sampling
    auroc_scores = []
    auprc_scores = []
    for c in present_classes:
        y_true  = (all_labels_np == c).astype(int)
        y_score = all_probs_np[:, c]
        if y_true.sum() == 0:
            continue
        try:
            auroc_scores.append(roc_auc_score(y_true, y_score))
            auprc_scores.append(average_precision_score(y_true, y_score))
        except ValueError:
            pass
            
    macro_auroc = float(np.mean(auroc_scores)) if auroc_scores else 0.0
    macro_auprc = float(np.mean(auprc_scores)) if auprc_scores else 0.0

    return {
        "val_loss":     avg_loss,
        "val_top1_acc": top1_acc,
        "val_top5_acc": top5_acc,
        "macro_f1":     macro_f1,
        "macro_auroc":  macro_auroc,
        "macro_auprc":  macro_auprc,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_moe(
    lr: float = 3e-4,
    epochs: int = 30,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    num_experts: int = 8,
    top_k: int = 1,
    capacity_factor: float = 1.25,
    num_buffer_tokens: int = 8,
    aux_loss_coef: float = 0.01,
    trial_number: int = 0,
):
    print(f"Device: {device} | experts={num_experts} | top_k={top_k} | "
          f"capacity_factor={capacity_factor} | buffer_tokens={num_buffer_tokens}")

    train_loader, val_loader = get_dataloaders(batch_size)

    model = build_vmoe_vit(
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        num_buffer_tokens=num_buffer_tokens,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    ckpt_dir  = os.environ.get("CKPT_DIR", os.environ.get("SCRATCH", "."))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"vmoe_ImageNet_trial{trial_number}-experts{num_experts}-topk{top_k}.pt"
    )

    # ---- Resume ----
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1

    # Pre-collect MoE layer refs
    moe_layer_refs = [
        (i, layer)
        for i, layer in enumerate(model.vit.encoder.layer)
        if isinstance(layer.intermediate, VMoEFeedForward)
    ]

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss     = 0.0
        epoch_overflow = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            batch  = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in batch.items()}
            labels = batch["labels"]

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs   = model(**batch)
                task_loss = criterion(outputs.logits, labels)

            # Aux losses + overflow
            load_balance     = torch.tensor(0.0, device=device)
            overflow_fracs   = []
            expert_load_logs = {}

            for layer_idx, layer in moe_layer_refs:
                moe = layer.intermediate
                load_balance = load_balance + moe.load_balance_loss
                overflow_fracs.append(moe.overflow_fraction)
                expert_load_logs[f"expert_load/layer{layer_idx}/aux_loss"] = \
                    moe.load_balance_loss.item()

            loss = task_loss + aux_loss_coef * load_balance
            loss.backward()

            # Grad norm (before clip)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += task_loss.item()

            avg_overflow = (sum(overflow_fracs) / len(overflow_fracs)
                            if overflow_fracs else 0.0)
            epoch_overflow.append(avg_overflow)

            with torch.no_grad():
                preds_train = outputs.logits.argmax(-1)
                train_acc   = (preds_train == labels).float().mean().item()

            wandb.log({
                "loss/task":         task_loss.item(),
                "loss/aux":          (aux_loss_coef * load_balance).item(),
                "loss/total":        loss.item(),
                "train_loss":        task_loss.item(),
                "load_balance_loss": load_balance.item(),
                "overflow_fraction": avg_overflow,
                "grad_norm":         total_norm,
                "top1_train_acc":    train_acc * 100.0,
                "lr":                scheduler.get_last_lr()[0],
                **expert_load_logs,
            })

            pbar.set_postfix(
                loss=f"{task_loss.item():.4f}",
                acc=f"{train_acc*100:.1f}%",
                overflow=f"{avg_overflow:.3f}",
                gnorm=f"{total_norm:.2f}",
            )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        val_metrics = evaluate(model, val_loader, criterion)
        print(
            f"  val top-1: {val_metrics['val_top1_acc']:.2f}%  "
            f"top-5: {val_metrics['val_top5_acc']:.2f}%  "
            f"F1: {val_metrics['macro_f1']:.4f}  "
            f"AUROC: {val_metrics['macro_auroc']:.4f}  "
            f"AUPRC: {val_metrics['macro_auprc']:.4f}"
        )

        wandb.log({
            "epoch":          epoch + 1,
            "epoch_loss":     avg_loss,
            "epoch_overflow": sum(epoch_overflow) / len(epoch_overflow),
            "val_loss":       val_metrics["val_loss"],
            "val_accuracy":   val_metrics["val_top1_acc"],
            "val_top5_acc":   val_metrics["val_top5_acc"],
            "macro_f1":       val_metrics["macro_f1"],
            "macro_auroc":    val_metrics["macro_auroc"],
            "macro_auprc":    val_metrics["macro_auprc"],
        })

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, ckpt_path)

        gc.collect()
        torch.cuda.empty_cache()
        model.train()

    # ---- Final evaluation ----
    val_metrics = evaluate(model, val_loader, criterion)
    accuracy    = val_metrics["val_top1_acc"]
    print(f"V-MoE Final Top-1 Accuracy: {accuracy:.2f}%")

    wandb.log({
        "final_accuracy":  accuracy,
        "final_top5_acc":  val_metrics["val_top5_acc"],
        "final_macro_f1":  val_metrics["macro_f1"],
        "final_auroc":     val_metrics["macro_auroc"],
        "final_auprc":     val_metrics["macro_auprc"],
    })

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy