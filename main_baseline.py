from maskaugfish.optimize import save_model
from maskaugfish.training import load_model
from maskaugfish.dataloader import build_index
import torch
import optuna
import numpy as np
import random

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    data_path = "./data"
    samples, class_to_id = build_index(data_path)
    model_name = 'shufflenet_v2_x0_5'
    model, input_size = load_model(model_name, len(class_to_id),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    aug_list = []
    study = optuna.create_study(direction="maximize",study_name="baseline")
    # Dummy study for baseline
    save_model(
        study,
        aug_list,
        samples,
        model,
        input_size,
        fname_prefix="best_model_baseline",
        model_save_path="models"
    )

if __name__ == "__main__":
    main()
