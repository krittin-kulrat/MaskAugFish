import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)


def load_model(backbone, num_classes, device):
    if backbone.lower not in models.list_models():
        raise ValueError(f"""Backbone '{backbone}' is not supported by torch vision,
                         see https://pytorch.org/vision/stable/models.html.""")
    model = models.__dict__[backbone](weights=models.__dict__[f"{backbone}_Weights"].DEFAULT)
    if 'resnet' in backbone.lower():
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        input_size = 224
    elif 'efficientnet_b2' in backbone.lower():
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        input_size = 288
    # Add more backbones as needed
    else:
        raise ValueError(f"Backbone '{backbone}' is not supported for modification.")
    model.to(device)
    return model, input_size



class EarlyStoppingLoss:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = torch.inf
        self.bad_epochs = 0

    def step(self, current: float) -> bool:
        if 100*(self.best - current)/abs(self.best+1e-8) < self.min_delta:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience
        else:
            self.best = current
            self.bad_epochs = 0
            return False


def compute_metrics_torch(
    preds: torch.Tensor, gts: torch.Tensor, num_classes: int
):
    """
    Compute confusion matrix, macro accuracy, and macro F1 using torchmetrics.
    Args:
        preds: [N] tensor of predicted class indices
        gts: [N] tensor of ground-truth class indices
        num_classes: number of classes
    """

    preds = preds.to(torch.int64)
    gts = gts.to(torch.int64)

    acc_metric = MulticlassAccuracy(
        num_classes=num_classes,
        average="macro",
    )
    f1_metric = MulticlassF1Score(
        num_classes=num_classes,
        average="macro",
    )
    cm_metric = MulticlassConfusionMatrix(
        num_classes=num_classes,
    )

    # Compute metrics
    macro_acc = acc_metric(preds, gts)   # scalar tensor
    macro_f1 = f1_metric(preds, gts)     # scalar tensor
    cm = cm_metric(preds, gts)           # [C, C] tensor

    return float(macro_acc.item()), float(macro_f1.item()), cm



# ------------------------------------------------------------
# training loop
# ------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    num_classes: int,
    epochs: int,
    criterion: nn.Module,
    opt_name: str = "sgd",
    lr: float = 1e-3,
    patience: int = 5,
    save_model: bool = False,
):

    # optimizer + scheduler
    if opt_name == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopper = EarlyStoppingLoss(patience=patience, min_delta=0.0)

    best_val_loss = float("inf")
    best_state = None
    best_info = None
    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= max(1, len(train_loader.dataset))
        scheduler.step()

        # -------- validation --------
        model.eval()
        val_loss = 0.0
        val_acc = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
        val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                preds = torch.argmax(logits, dim=1)
                val_acc.update(preds, y)
                val_f1.update(preds, y)

        val_loss /= max(1, len(val_loader.dataset))

        val_macroAcc = val_acc.compute().item()
        val_macroF1 = val_f1.compute().item()


        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macroAcc={val_macroAcc:.3f} val_macroF1={val_macroF1:.3f}"
        )

        # track best by lowest val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_info = {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "val_macroAcc": float(val_macroAcc),
                "val_macroF1": float(val_macroF1),
            }

        # early stopping
        if early_stopper.step(val_loss):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best val_loss={early_stopper.best:.4f})"
            )
            break
    return best_state, best_info # If not save model, best_state is None


# ------------------------------------------------------------
# One fold of cross-validation
# ------------------------------------------------------------
def train_one_fold(args,
                   train_loader,
                   val_loader,
                   backbone,
                   model, # Original model with pre-trained weights
                   fold_id: int = 0,
                   save_model: bool = False, # Only save model after optimization
                   ):
    """
    Train + evaluate a single fold (fold_id) and return summary metrics
    for cross-validation aggregation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ----- class weights (all in Torch) -----
    train_labels = train_loader.dataset.y
    num_classes = len(train_labels.unique)


    criterion = nn.CrossEntropyLoss()

    # ----- train using generic loop -----
    best_state, best_info = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_classes=num_classes,
        epochs=args.epochs,
        criterion=criterion,
        opt_name=args.opt,
        lr=args.lr,
        patience=args.patience,
        save_model=True,
    )

    return best_info, best_state



# main: run all folds and summarize CV
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)  # folder that contains data/fish_images
    ap.add_argument("--phase", choices=["feature", "finetune"], default="feature")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--opt", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without val loss improvement)",
    )
    args = ap.parse_args()

    fold_summaries = []
    for fid in range(args.n_splits):
        print(f"\n========== Training fold {fid} / {args.n_splits - 1} ==========\n")
        fold_summary = train_one_fold(args, fold_id=fid)
        fold_summaries.append(fold_summary)

    # cross-validation summary (pure Python, no NumPy)
    valid_folds = [fs for fs in fold_summaries if fs["test_macroAcc"] is not None]
    if valid_folds:
        mean_macroAcc = sum(fs["test_macroAcc"] for fs in valid_folds) / len(
            valid_folds
        )
        mean_macroF1 = sum(fs["test_macroF1"] for fs in valid_folds) / len(
            valid_folds
        )
    else:
        mean_macroAcc = None
        mean_macroF1 = None

    cv_summary = {
        "n_splits": args.n_splits,
        "fold_summaries": fold_summaries,
        "mean_test_macroAcc": mean_macroAcc,
        "mean_test_macroF1": mean_macroF1,
    }
    cv_out = Path("runs") / "cv_summary.json"
    cv_out.write_text(json.dumps(cv_summary, indent=2))

    print("\n===== Cross-validation summary =====")
    print(json.dumps(cv_summary, indent=2))


if __name__ == "__main__":
    main()
