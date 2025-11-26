import optuna
import torch
from maskaugfish.augmentation import Augmentation
from maskaugfish.dataloader import make_dataloaders
from maskaugfish.training import train_model

Z = 2.132 # 90% confidence interval for t-distribution with df=4


def build_cfg(selected_augs, trial):
    cfg = {"augmentations": {}, "pipeline": []}
    for aug in selected_augs:
        if aug in cfg["pipeline"]:
            continue
        else:
            cfg["pipeline"].append(aug)
        if aug == "channel_switch":
            cfg["augmentations"][aug] = {"prob": trial.suggest_float("channel_switch_prob", 0.0, 1.0)}
        elif aug == "addition":
            cfg["augmentations"][aug] = {
                "prob": trial.suggest_float("addition_prob", 0.0, 1.0),
                "value": trial.suggest_int("addition_value", 0, 100),
                "range_val": trial.suggest_int("addition_range_val", 1, 100)
            }
        elif aug == "guassian_noise":
            cfg["augmentations"][aug] = {
                "prob": trial.suggest_float("gaussian_noise_prob", 0.0, 1.0),
                "mean": trial.suggest_float("gaussian_noise_mean", 0.0, 1.0),
                "std": trial.suggest_float("gaussian_noise_std", 0.0, 0.5)
            }
        elif aug == "dropout":
            cfg["augmentations"][aug] = {
                "prob": trial.suggest_float("dropout_prob", 0.0, 1.0),
                "dropout_prob": trial.suggest_float("dropout_dropout_prob", 0.0, 1.0)
            }
        elif aug == "gaussian_blur":
            cfg["augmentations"][aug] = {
                "prob": trial.suggest_float("gaussian_blur_prob", 0.0, 1.0),
                "kernel_size": trial.suggest_int("gaussian_blur_kernel_size", 1, 15, step=2),
                "sigma": trial.suggest_float("gaussian_blur_sigma", 0.1, 5.0)
            }
        elif aug == "solarize":
            cfg["augmentations"][aug] = {
                "prob": trial.suggest_float("solarize_prob", 0.0, 1.0),
                "threshold": trial.suggest_int("solarize_threshold", 0, 255)
            }
        elif aug == "equalize":
            cfg["augmentations"][aug] = {
                "prob": trial.suggest_float("equalize_prob", 0.0, 1.0)
            }
    return cfg


def eval_with_added_aug(additional_aug, prev_augs,
                        samples, # For generating dataloaders
                        model, # Original model with pre-trained weights
                        input_size, # Input size for the model
                        ):
    T = 20
    def objective(trial):
        selected_augs = prev_augs + [additional_aug]
        cfg = build_cfg(selected_augs, trial)
        pipeline = Augmentation(cfg)
        other_metrics = dict()
        acc = torch.zeros(5)
        for i in range(5):
            train_loader, val_loader, _ = make_dataloaders(
                samples=samples,
                batch_size=512,
                num_workers=0,
                img_size=input_size,
                weighted_train=True,
                seed=42,
                test_ratio=0.2,
                n_splits=5,
                fold_id=i,
                augment_pipeline=pipeline
            )
            _, best_info=  train_model(
                model,
                train_loader,
                val_loader,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                num_classes=len(train_loader.dataset.y.unique()),
                epochs=400,
                opt_name="adamw",
                lr=1e-3,
                patience=10,
                save_model=False,
            )
            acc[i] = best_info["val_macroAcc"]
            for key, value in best_info.items():
                if key != "val_macroAcc":
                    if key not in other_metrics:
                        other_metrics[key] = []
                    other_metrics[key].append(value)
        return acc.mean().item() - Z * (acc.std().item())
    pipeline_length = len(prev_augs) + 1
    study = optuna.create_study(direction="maximize",study_name=f"eval_{pipeline_length}")
    study.optimize(objective, n_trials=T * pipeline_length, n_jobs=1)
    return study

def save_model(trial, selected_augs,
               samples,
               model,
               input_size,
               fname_prefix,
               model_save_path):
    cfg = build_cfg(selected_augs, trial)
    pipeline = Augmentation(cfg)
    for i in range(5):
        train_loader, val_loader, _ = make_dataloaders(
            samples=samples,
            batch_size=512,
            num_workers=0,
            img_size=input_size,
            weighted_train=True,
            seed=42,
            test_ratio=0.2,
            n_splits=5,
            fold_id=i,
            augment_pipeline=pipeline
        )
        best_state, best_info=  train_model(
            model,
            train_loader,
            val_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_classes=len(train_loader.dataset.y.unique()),
            epochs=400,
            opt_name="adamw",
            lr=1e-3,
            patience=10,
            save_model=True)
        torch.save({"weight":best_state,
                    "info":best_info}, f"{model_save_path}/{fname_prefix}_fold{i}.pth")

