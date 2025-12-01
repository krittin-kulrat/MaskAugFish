from maskaugfish.optimize import eval_with_added_aug, save_model
from maskaugfish.dataloader import build_index
from maskaugfish.training import load_model
import torch
import numpy as np
import random

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    aug_list = [
        "channel_switch",
        "addition",
        "guassian_noise",
        "dropout",
        "gaussian_blur",
        "solarize",
        "equalize"
    ]
    regime = 'fish-only'  # Options: 'whole-image', 'fish-only', 'background-only'
    data_path = "./data"
    samples, class_to_id = build_index(data_path)
    model_name = 'shufflenet_v2_x0_5'
    model, input_size = load_model(model_name, len(class_to_id),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    best_study = None
    current_augs = []
    for step in range(1):
        best_step_study = None
        for i in range(len(aug_list)):
            if aug_list[i] in current_augs:
                continue
            print(f"Evaluating addition of augmentation: {aug_list[i]}")
            study = eval_with_added_aug(aug_list[i], current_augs,
                                        samples,
                                        model,
                                        input_size,
                                        regime=regime
                                        )
            if best_step_study is None:
                best_step_study = study
                best_added_aug = aug_list[i]
            if study.best_value > best_step_study.best_value:
                best_step_study = study
                best_added_aug = aug_list[i]
            if step == 0:
                save_model(
                    study.best_trial,
                    [aug_list[i]],
                    samples,
                    model,
                    input_size,
                    fname_prefix=f"best_model_{aug_list[i]}",
                    model_save_path="models_fish",
                    regime=regime
                )

        save_model(
            best_step_study.best_trial,
            current_augs + [best_added_aug],
            samples,
            model,
            input_size,
            fname_prefix=f"best_model_step_{step+1}",
            model_save_path="models_fish"
        )
        if best_study is None:
            best_study = best_step_study
            current_augs.append(best_added_aug)
        elif best_step_study.best_value > best_study.best_value:
            best_study = best_step_study
            current_augs.append(best_added_aug)
            print(f"Added augmentation: {best_added_aug} with score: {best_step_study.best_value}")
            print(f"Current augmentations: {current_augs}")
            print(f"Best parameters so far: {best_study.best_params}")
        else:
            print("No further improvement, stopping augmentation addition.")
            break

    print("Optimization completed.")
    print(f"Final augmentations: {current_augs}")
    print(f"Best parameters: {best_study.best_params}")
    print(f"Best score: {best_study.best_value}")
    save_model(
        best_study.best_trial,
        current_augs,
        samples,
        model,
        input_size,
        fname_prefix="best_model_final",
        model_save_path="models"
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
