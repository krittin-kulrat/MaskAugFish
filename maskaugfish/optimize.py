import optuna
import torch
from maskaugfish.augmentation import Augmentation
from maskaugfish.dataloader import make_dataloaders
from maskaugfish.training import train_model
import os

Z = {1: 63.657,
     2: 9.925,
     3: 5.841,
     4: 4.604,
     5: 4.032,
     6: 3.707,
     7: 3.499,
     8: 3.355,
     9: 3.250,
     10: 3.169,
     11: 3.106,
     12: 3.055,
     13: 3.012,
     14: 2.977,
     15: 2.947,
     16: 2.921,
     17: 2.898,
     18: 2.878,
     19: 2.861,
     20: 2.845}


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
                        regime='whole-image'
                        ):
    T = 5
    def objective(trial):
        selected_augs = prev_augs + [additional_aug]
        cfg = build_cfg(selected_augs, trial)
        pipeline = Augmentation(cfg, regime=regime)
        other_metrics = dict()
        acc = torch.zeros(5)
        for i in range(5):
            train_loader, val_loader, _ = make_dataloaders(
                samples=samples,
                batch_size=1024,
                num_workers=4,
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
                epochs=100,
                opt_name="adamw",
                lr=1e-3,
                patience=3,
                save_model=False,
            )
            acc[i] = best_info["val_macroAcc"]
            for key, value in best_info.items():
                if key != "val_macroAcc":
                    if key not in other_metrics:
                        other_metrics[key] = []
                    other_metrics[key].append(value)
        return acc.mean().item() - Z[pipeline.dof] * (acc.std().item()) / (5 ** 0.5)
    pipeline_length = len(prev_augs) + 1
    study = optuna.create_study(direction="maximize",study_name=f"eval_{pipeline_length}")
    study.optimize(objective, n_trials=T * pipeline_length, n_jobs=1)
    return study

def save_model(trial, selected_augs,
               samples,
               model,
               input_size,
               fname_prefix,
               model_save_path,
               regime='whole-image'):
    cfg = build_cfg(selected_augs, trial)
    pipeline = Augmentation(cfg, regime=regime)
    for i in range(5):
        if os.path.exists(f"{model_save_path}/{fname_prefix}_fold{i}.pth"):
            print(f"Model for fold {i} already exists. Skipping...")
            continue
        train_loader, val_loader, _ = make_dataloaders(
            samples=samples,
            batch_size=1024,
            num_workers=4,
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
            epochs=100,
            opt_name="adamw",
            lr=1e-3,
            patience=3,
            save_model=True)
        torch.save({"weight":best_state,
                    "info":best_info}, f"{model_save_path}/{fname_prefix}_fold{i}.pth")

