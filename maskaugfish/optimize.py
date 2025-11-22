import optuna

Z = 1.96  # 95% confidence interval
n = 5  # folds


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


def eval_with_added_aug(additional_aug, prev_augs):
    T = 20

    def objective(trial):
        selected_augs = prev_augs + [additional_aug]
        # cfg = build_cfg(selected_augs, trial)
        # pipeline = Augmentation(cfg)
        # Here you would integrate with your training and evaluation pipeline.
        # For demonstration purposes, we'll return a mock accuracy value.
        x = len(selected_augs)
        mock_accuracy = trial.suggest_float("mock_accuracy", x, x+1)
        mock_std = trial.suggest_float("mock_std", 0.0, 1.0)
        return mock_accuracy - Z * (mock_std / (n ** 0.5))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=T * (len(prev_augs) + 1))
    return study.best_params, study.best_value
