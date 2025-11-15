import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, confusion_matrix
import torchvision.models as models

from dataloader import build_index, make_dataloaders


def freeze_all_but_fc(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze all layers except the final fully-connected layer."""
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in model.fc.parameters():
        p.requires_grad = True


def train_one_fold(args, fold_id: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path("runs") / time.strftime(f"%Y%m%d-%H%M%S_fold{fold_id}")
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- dataset & loaders (Krittin-style) -----
    samples, class_to_id = build_index(args.data_root)
    num_classes = len(class_to_id)

    train_loader, val_loader, test_loader = make_dataloaders(
        samples=samples,
        batch_size=args.batch,
        num_workers=args.workers,
        img_size=args.img_size,
        augment_pipeline=None,          # keep augmentation OFF for now (simpler)
        weighted_train=True,            # uses WeightedRandomSampler
        seed=args.seed,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        fold_id=fold_id,
    )

    # ----- model: ResNet18 + ImageNet weights -----
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # feature-extract or fine-tune
    if args.phase == "feature":
        freeze_all_but_fc(model, freeze=True)
    else:
        freeze_all_but_fc(model, freeze=False)

    # plain CrossEntropyLoss (WeightedRandomSampler already helps imbalance)
    criterion = nn.CrossEntropyLoss()

    # optimizer + scheduler
    if args.opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ----- train loop -----
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
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

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        preds, gts = [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        acc = (preds == gts).mean()
        f1 = f1_score(gts, preds, average="macro", zero_division=0.0)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(acc),
                "val_macroF1": float(f1),
            }
        )

        print(
            f"[fold {fold_id}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={acc:.3f} macroF1={f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {"state_dict": model.state_dict(), "class_to_id": class_to_id},
                outdir / "best.pt",
            )
            (outdir / "best.txt").write_text(
                json.dumps(
                    {"epoch": epoch, "val_acc": float(acc), "val_macroF1": float(f1)},
                    indent=2,
                )
            )

    # ----- save metrics.csv -----
    import csv

    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # ----- final test evaluation -----
    ckpt = torch.load(outdir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            gts.append(y.cpu().numpy())

    if len(preds) > 0:
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        test_acc = (preds == gts).mean()
        test_f1 = f1_score(gts, preds, average="macro", zero_division=0.0)
        cm = confusion_matrix(gts, preds, labels=list(range(num_classes)))

        np.savetxt(outdir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        (outdir / "test.txt").write_text(
            json.dumps(
                {"acc": float(test_acc), "macroF1": float(test_f1)}, indent=2
            )
        )

        print(
            f"[fold {fold_id}] TEST acc={test_acc:.3f} macroF1={test_f1:.3f}"
        )
    else:
        print("[fold {fold_id}] No test samples (test_loader empty).")


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
    args = ap.parse_args()

    train_one_fold(args, fold_id=0)


if __name__ == "__main__":
    main()
