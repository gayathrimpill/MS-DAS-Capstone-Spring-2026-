import os
import torch
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTImageProcessor, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(batch_size):

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    dataset = load_dataset("cifar10")

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip()
    ])

    test_transform = T.Resize((224, 224))

    def train_collate(batch):

        images = [
            train_transform(item["img"].convert("RGB"))
            for item in batch
        ]

        labels = torch.tensor([item["label"] for item in batch])

        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs


    def test_collate(batch):

        images = [
            test_transform(item["img"].convert("RGB"))
            for item in batch
        ]

        labels = torch.tensor([item["label"] for item in batch])

        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs


    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=test_collate,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    return train_loader, test_loader



def run_baseline(
    lr=5e-5,
    epochs=15,
    batch_size=128,
    weight_decay=1e-4,
    trial_number=0
):

    print("Device:", device)

    train_loader, test_loader = get_dataloaders(batch_size)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    # checkpoint location
    ckpt_dir = os.environ.get("SCRATCH", ".")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(
        ckpt_dir,
        f"baseline_trial_{trial_number}.pt"
    )


    # -------- Resume Logic --------
    start_epoch = 0

    if os.path.exists(ckpt_path):

        print("Resuming from checkpoint:", ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1

        print("Resuming at epoch", start_epoch)

    else:
        print("Starting fresh training")


    # -------- Training --------
    model.train()

    for epoch in range(start_epoch, epochs):

        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            wandb.log({"train_loss": loss.item()})

            pbar.set_postfix(loss=loss.item())


        avg_loss = total_loss / len(train_loader)
        print("Epoch loss:", avg_loss)

        wandb.log({"epoch_loss": avg_loss})


        # save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, ckpt_path)

        print("Checkpoint saved:", ckpt_path)


    # -------- Evaluation --------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in tqdm(test_loader, desc="Eval"):

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            outputs = model(**batch)

            preds = outputs.logits.argmax(-1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)


    accuracy = 100.0 * correct / total

    print(f"Accuracy: {accuracy:.2f}%")

    wandb.log({"final_accuracy": accuracy})

    return accuracy