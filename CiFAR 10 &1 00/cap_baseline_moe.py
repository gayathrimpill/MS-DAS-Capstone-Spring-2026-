import optuna
import wandb
import torch
from cap_moe import run_moe


def objective(trial):

    try:

        lr = trial.suggest_float("lr",1e-6,1e-4,log=True)
        epochs = trial.suggest_int("epochs",3,8)
        num_experts = trial.suggest_int("experts",4,8)
        top_k = trial.suggest_int("top_k",1,num_experts)

        wandb.init(
            project="cmu-capstone-vit",
            name=f"moe-trial-{trial.number}-experts-{num_experts}-topk-{top_k}",
            config={
                "lr": lr,
                "epochs": epochs,
                "experts": num_experts,
                "top_k": top_k
            },
            reinit=True
        )

        accuracy = run_moe(
            lr=lr,
            epochs=epochs,
            num_experts=num_experts,
            top_k=top_k,
            trial_number=trial.number
        )

        wandb.log({"test_accuracy": accuracy})
        wandb.finish()

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return accuracy

    except RuntimeError as e:

        if "out of memory" in str(e):

            print("OOM trial skipped")

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            return 0.0

        else:
            raise e


if __name__ == "__main__":

    # study = optuna.create_study(
    #     study_name="vit_moe_cifar10",
    #     direction="maximize",
    #     storage="sqlite:///vit_moe.db",
    #     load_if_exists=True
    # )
    study = optuna.create_study(
        study_name="vit_moe_cifar10_v2",  # new name
        direction="maximize",
        storage="sqlite:////ocean/projects/cis250163p/tnair/vit_moe_v2.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20, catch=(RuntimeError,))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)