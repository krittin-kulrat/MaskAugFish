import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
)
import torchvision.models as models
import csv

from dataloader import build_index, make_dataloaders


def freeze_all_but_fc(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze all layers except the final fully-connected layer."""
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in model.fc.parameters():
        p.requires_grad = True


class EarlyStoppingLoss:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, current: float) -> bool:
        """Return True if training should stop."""
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


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
):

    # ----- optimizer + scheduler -----
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

    # ----- early stopping on validation loss -----
    early_stopper = EarlyStoppingLoss(patience=patience, min_delta=0.0)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_info = None

    for epoch in range(1, epochs + 1):
        # ---- train ----
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

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        all_preds, all_gts = [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_gts.append(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_gts = np.concatenate(all_gts)

        # macro accuracy + macro F1 (no micro metrics)
        cm_val = confusion_matrix(
            all_gts, all_preds, labels=list(range(num_classes))
        )
        per_class_acc = []
        for i in range(num_classes):
            support = cm_val[i].sum()
            correct = cm_val[i, i]
            acc_i = correct / support if support > 0 else 0.0
            per_class_acc.append(acc_i)
        val_macroAcc = float(np.mean(per_class_acc))

        val_macroF1 = f1_score(
            all_gts, all_preds, average="macro", zero_division=0.0
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_macroAcc": float(val_macroAcc),
                "val_macroF1": float(val_macroF1),
            }
        )

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macroAcc={val_macroAcc:.3f} val_macroF1={val_macroF1:.3f}"
        )

        # track best model by *lowest val_loss*
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # copy state_dict to CPU to decouple from further training
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_info = {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "val_macroAcc": float(val_macroAcc),
                "val_macroF1": float(val_macroF1),
            }

        # early stopping check
        if early_stopper.step(val_loss):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best val_loss={early_stopper.best:.4f})"
            )
            break

    return best_state, history, best_info


def train_one_fold(args, fold_id: int = 0):
    """
    Train + evaluate a single fold (fold_id) and return summary metrics
    for cross-validation aggregation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path("runs") / f"{timestamp}_fold{fold_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- dataset & loaders 
    samples, class_to_id = build_index(args.data_root)
    num_classes = len(class_to_id)

    train_loader, val_loader, test_loader = make_dataloaders(
        samples=samples,
        batch_size=args.batch,
        num_workers=args.workers,
        img_size=args.img_size,
        augment_pipeline=None,          # baseline: no aug 
        weighted_train=True,            # uses WeightedRandomSampler
        seed=args.seed,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        fold_id=fold_id,
    )

    # ----- compute class weights for weighted CE  -----
    if hasattr(train_loader.dataset, "y"):
        train_labels = np.array(train_loader.dataset.y)
    else:
        all_labels = []
        for batch in train_loader:
            all_labels.append(batch["label"].numpy())
        train_labels = np.concatenate(all_labels)

    class_counts = np.bincount(train_labels, minlength=num_classes)
    inv = 1.0 / np.clip(class_counts.astype(np.float32), 1, None)
    weights = inv / inv.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # ----- model: ResNet18 + ImageNet weights -----
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # feature-extract or fine-tune (for your experiments, you'll use finetune only)
    if args.phase == "feature":
        freeze_all_but_fc(model, freeze=True)
    else:
        freeze_all_but_fc(model, freeze=False)

    # weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ----- call generic training loop -----
    best_state, history, best_info = train_model(
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
    )

    # ----- save metrics.csv -----
    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # ----- save best checkpoint info -----
    torch.save(
        {"state_dict": best_state, "class_to_id": class_to_id},
        outdir / "best.pt",
    )
    (outdir / "best.txt").write_text(json.dumps(best_info, indent=2))

    # ----- final test evaluation using best_state -----
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            gts.append(y.cpu().numpy())

    fold_summary = {
        "fold_id": fold_id,
        "best_epoch": best_info["epoch"] if best_info is not None else None,
        "best_val_loss": best_info["val_loss"] if best_info is not None else None,
        "best_val_macroAcc": best_info["val_macroAcc"] if best_info is not None else None,
        "best_val_macroF1": best_info["val_macroF1"] if best_info is not None else None,
        "test_macroAcc": None,
        "test_macroF1": None,
    }

    if len(preds) > 0:
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        cm = confusion_matrix(gts, preds, labels=list(range(num_classes)))

        # macro accuracy from confusion matrix
        per_class_acc = []
        for i in range(num_classes):
            support = cm[i].sum()
            correct = cm[i, i]
            acc_i = correct / support if support > 0 else 0.0
            per_class_acc.append(acc_i)
        test_macroAcc = float(np.mean(per_class_acc))
        test_macroF1 = f1_score(
            gts, preds, average="macro", zero_division=0.0
        )

        fold_summary["test_macroAcc"] = test_macroAcc
        fold_summary["test_macroF1"] = test_macroF1

        np.savetxt(
            outdir / "confusion_matrix.csv", cm, fmt="%d", delimiter=","
        )
        (outdir / "test.txt").write_text(
            json.dumps(
                {"macroAcc": test_macroAcc, "macroF1": test_macroF1},
                indent=2,
            )
        )

        # ---- per-class / per-species metrics ----
        class_to_id = torch.load(outdir / "best.pt", map_location="cpu")["class_to_id"]
        id_to_class = {v: k for k, v in class_to_id.items()}
        class_names = [id_to_class[i] for i in range(num_classes)]

        report_text = classification_report(
            gts, preds, target_names=class_names, zero_division=0.0
        )
        (outdir / "classification_report.txt").write_text(report_text)

        report_dict = classification_report(
            gts,
            preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0.0,
        )
        (outdir / "classification_report.json").write_text(
            json.dumps(report_dict, indent=2)
        )

        per_class_rows = []
        for i, name in enumerate(class_names):
            support = int(cm[i].sum())
            correct = int(cm[i, i])
            acc_i = correct / support if support > 0 else 0.0
            metrics_i = report_dict.get(name, {})
            per_class_rows.append(
                {
                    "class_index": i,
                    "class_name": name,
                    "support": support,
                    "macroAcc": float(acc_i),
                    "precision": float(metrics_i.get("precision", 0.0)),
                    "recall": float(metrics_i.get("recall", 0.0)),
                    "f1": float(metrics_i.get("f1-score", 0.0)),
                }
            )

        with open(outdir / "per_class_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_index",
                    "class_name",
                    "support",
                    "macroAcc",
                    "precision",
                    "recall",
                    "f1",
                ],
            )
            writer.writeheader()
            writer.writerows(per_class_rows)

        summary = {
            "fold_id": fold_id,
            "test_macroAcc": test_macroAcc,
            "test_macroF1": test_macroF1,
            "num_classes": num_classes,
            "num_test_samples": int(len(gts)),
        }
        (outdir / "test_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        print(
            f"[fold {fold_id}] TEST macroAcc={test_macroAcc:.3f} "
            f"macroF1={test_macroF1:.3f} (results in {outdir})"
        )
    else:
        print(f"[fold {fold_id}] No test samples (test_loader empty).")

    return fold_summary


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

    # ----- cross-validation summary -----
    cv_out = Path("runs") / "cv_summary.json"
    valid_folds = [
        fs for fs in fold_summaries if fs["test_macroAcc"] is not None
    ]
    if valid_folds:
        mean_macroAcc = float(
            np.mean([fs["test_macroAcc"] for fs in valid_folds])
        )
        mean_macroF1 = float(
            np.mean([fs["test_macroF1"] for fs in valid_folds])
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
    cv_out.write_text(json.dumps(cv_summary, indent=2))

    print("\n===== Cross-validation summary =====")
    print(json.dumps(cv_summary, indent=2))


if __name__ == "__main__":
    main()
