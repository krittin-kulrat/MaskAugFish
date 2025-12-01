from maskaugfish.evaluating import evaluate
from maskaugfish.dataloader import build_index
import torch
import numpy as np
import random
import glob

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    aug_list = []
    data_path = "./data"
    samples, class_to_id = build_index(data_path)
    checkpoints = glob.glob("models/best_model_*.pth")
    for i in range(len(checkpoints)//5):
        if len(checkpoints[5*i].split("_")) == 5:
            regime, augmentation = checkpoints[5*i].split("_")[1].replace("\\best", ""), checkpoints[5*i].split("_")[-2]
        elif len(checkpoints[5*i].split("_")) == 6:
            regime, augmentation = checkpoints[5*i].split("_")[1].replace("\\best", ""), "_".join(checkpoints[5*i].split("_")[-3:-1])
        else:
            regime, augmentation = "None", "None"
        aug_list.append((regime, augmentation))
        ckpt_glob = checkpoints[5*i].replace("fold0.pth", "*.pth")
        print(f"Evaluating Regime: {regime}, Augmentation: {augmentation}")
        evaluate(
            ckpt_glob,
            samples,
            class_to_id,
            batch_size=1024,
            img_size=224,
            num_workers=4,
            regime=regime,
            augmentation=augmentation
        )

if __name__ == "__main__":
    main()
