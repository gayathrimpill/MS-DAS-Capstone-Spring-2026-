"""
cap_baseline_moe_vmoe.py
------------------------
Optuna hyperparameter sweep for V-MoE on ImageNet.

Sweep space aligned with V-MoE paper recommendations:
  - num_experts: 8 or 16  (paper's main configs)
  - top_k:       1         (paper's inference config; 2 during fine-tune)
  - capacity_factor: 1.0–2.0
  - num_buffer_tokens: 0, 4, 8  (0 = hard-drop baseline)
  - lr, epochs as usual
"""

import os
import time
import gc

import optuna
import torch
import wandb

from cap_moe_vmoe import run_moe


def objective(trial: optuna.Trial) -> float:
    try:
        # --- Hyperparameter space ---
        #lr               = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        lr = trial.suggest_float("lr", 5e-5, 2e-4, log=True)
        epochs           = trial.suggest_int("epochs", 30, 60)
        num_experts      = trial.suggest_categorical("experts", [8, 16])
        top_k            = trial.suggest_categorical("top_k", [1, 2,3,4,5,6,7,8])
        capacity_factor  = trial.suggest_float("capacity_factor", 1.0, 2.0)
        num_buffer_tokens = trial.suggest_categorical("buffer_tokens", [0])
        aux_loss_coef    = trial.suggest_float("aux_loss_coef", 0.005, 0.05, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

        wandb.init(
            project="cmu-capstone-vmoe-imagenet-1k",
            name=(f"vmoe-trial{trial.number}-E{num_experts}-k{top_k}"
                  f"-cf{capacity_factor:.2f}-buf{num_buffer_tokens}"),
            config={
                "lr": lr, "epochs": epochs,
                "experts": num_experts, "top_k": top_k,
                "capacity_factor": capacity_factor,
                "buffer_tokens": num_buffer_tokens,
                "aux_loss_coef": aux_loss_coef,
                "weight_decay": weight_decay,
            },
            reinit=True,
        )

        accuracy = run_moe(
            lr=lr,
            epochs=epochs,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            num_buffer_tokens=num_buffer_tokens,
            aux_loss_coef=aux_loss_coef,
            weight_decay=weight_decay,
            trial_number=trial.number,
        )

        wandb.log({"test_accuracy": accuracy})
        wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()
        return accuracy

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            print(f"OOM on trial {trial.number} — skipping")
            return 0.0
        raise


if __name__ == "__main__":

    study_name  = "vmoe_imagenet_1k_v1"
    storage_url = "sqlite:///vmoe_imagenet_1k.db"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
        )
    else:
        study = None
        while study is None:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_url)
            except Exception:
                time.sleep(2)

    study.optimize(objective, n_trials=5)

    if local_rank == 0:
        print(f"\nBest accuracy : {study.best_value:.2f}%")
        print(f"Best params   : {study.best_params}")
