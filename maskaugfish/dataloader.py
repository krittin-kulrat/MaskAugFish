# dataloader.py
# Fish4Knowledge dataloader with:
# - torchvision.transforms.v2
# - Stratified 5-fold CV
# - Optional WeightedRandomSampler
# - Optional test loader
# - Basic sanity checks (class freq, fold determinism)

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2
from sklearn.model_selection import StratifiedKFold


# -----------------------------
# Helpers
# -----------------------------
def _find_subdir(root: Path, prefer: str, alt: str) -> Path:
    p1, p2 = root / prefer, root / alt
    return p1 if p1.exists() else p2


def list_images(dirpath: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in dirpath.rglob("*") if p.suffix.lower() in exts]


# -----------------------------
# Index building
# -----------------------------
def build_index(
    root: str | Path,
    image_sub: str = "fish_images",
    mask_sub: str = "fish_masks",
    alt_image_sub: str = "fish_image",
    alt_mask_sub: str = "mask_image",
) -> Tuple[List[Tuple[str, Optional[str], int]], Dict[str, int]]:
    root = Path(root)
    img_root = _find_subdir(root, image_sub, alt_image_sub)
    mask_root = _find_subdir(root, mask_sub, alt_mask_sub)

    classes = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    class_to_id = {c: i for i, c in enumerate(classes)}

    mask_lookup: Dict[str, Path] = {}
    if mask_root.exists():
        for mp in list_images(mask_root):
            mask_lookup[mp.stem] = mp

    samples: List[Tuple[str, Optional[str], int]] = []
    for c in classes:
        cls_id = class_to_id[c]
        cls_dir = img_root / c
        for ip in list_images(cls_dir):
            mp = mask_lookup.get(ip.stem)
            samples.append((str(ip), str(mp) if mp is not None else None, cls_id))

    return samples, class_to_id


# -----------------------------
# Dataset
# -----------------------------
def make_img_tf(img_size: int, train: bool, augment: bool) -> v2.Compose:
    tfs = [v2.ToImage(), v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)]
    if train and augment:
        tfs += [v2.RandomHorizontalFlip(p=0.5)]
    tfs += [v2.ToDtype(torch.float32, scale=True)]
    return v2.Compose(tfs)


def make_mask_tf(img_size: int) -> v2.Compose:
    return v2.Compose(
        [v2.ToImage(), v2.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)]
    )


class FishDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, Optional[str], int]],
        img_size: int = 224,
        train: bool = True,
        augment: bool = False,
    ):
        self.samples = samples
        self.tf_img = make_img_tf(img_size, train, augment)
        self.tf_mask = make_mask_tf(img_size)
        self.y = np.array([lbl for _, _, lbl in samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.tf_img(img)

        out = {
            "image": img,
            "label": torch.as_tensor(label, dtype=torch.long),
            "path": img_path,
        }

        if mask_path is not None and Path(mask_path).exists():
            mask = Image.open(mask_path)
            mask = self.tf_mask(mask)
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
            out["mask"] = mask

        return out


# -----------------------------
# Sampler / folds
# -----------------------------
def make_sample_weights(labels: np.ndarray, clip_max: float = 50.0) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))
    w = np.array([1.0 / freq[int(y)] for y in labels], dtype=np.float32)
    w = w / w.mean()
    if clip_max is not None:
        w = np.clip(w, a_min=None, a_max=clip_max)
    return torch.from_numpy(w)


def make_fold_loaders(
    samples: List[Tuple[str, Optional[str], int]],
    batch_size: int = 32,
    num_workers: int = 4,
    n_splits: int = 5,
    fold_id: int = 0,
    seed: int = 42,
    img_size: int = 224,
    augment: bool = False,
    weighted_train: bool = True,
    test_samples: Optional[List[Tuple[str, Optional[str], int]]] = None,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y_all = np.array([lbl for _, _, lbl in samples], dtype=np.int64)
    splits = list(skf.split(np.zeros(len(y_all)), y_all))
    assert 0 <= fold_id < n_splits, f"fold_id must be in [0,{n_splits-1}]"

    tr_idx, va_idx = splits[fold_id]
    tr_samples = [samples[i] for i in tr_idx]
    va_samples = [samples[i] for i in va_idx]

    train_ds = FishDataset(tr_samples, img_size=img_size, train=True, augment=augment)
    val_ds = FishDataset(va_samples, img_size=img_size, train=False, augment=False)

    if weighted_train:
        weights = make_sample_weights(train_ds.y)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = None
    if test_samples is not None and len(test_samples) > 0:
        test_ds = FishDataset(test_samples, img_size=img_size, train=False, augment=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader, test_loader


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="train/val dataset root")
    ap.add_argument("--test_root", type=str, default="", help="optional separate test root")
    ap.add_argument("--fold", type=int, default=0, help="which fold [0..4]")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="enable train-time augmentation")
    ap.add_argument("--no_weighted", action="store_true", help="disable weighted sampler")
    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[INFO] Scanning dataset at: {args.root}")
    samples, class_to_id = build_index(args.root)
    print(f"[INFO] Found {len(class_to_id)} classes, {len(samples)} samples")

    # Class frequency check
    ys = [s[2] for s in samples]
    freq = dict(sorted(Counter(ys).items()))
    print(f"[CHECK] Class frequencies: {freq}")

    # Determinism check for folds
    skf_a = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    skf_b = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    idxs = np.arange(len(ys))
    a = list(skf_a.split(np.zeros_like(idxs), np.array(ys)))[args.fold][0]
    b = list(skf_b.split(np.zeros_like(idxs), np.array(ys)))[args.fold][0]
    print(f"[CHECK] same indices across runs with seed={args.seed}: {np.array_equal(a, b)}")

    # Optional test set
    test_samples = None
    if args.test_root:
        test_samples, _ = build_index(args.test_root)

    train_loader, val_loader, test_loader = make_fold_loaders(
        samples,
        batch_size=args.bs,
        num_workers=args.num_workers,
        n_splits=args.n_splits,
        fold_id=args.fold,
        seed=args.seed,
        img_size=args.img_size,
        augment=args.augment,
        weighted_train=not args.no_weighted,
        test_samples=test_samples,
    )

    # Quick sanity batches
    batch = next(iter(train_loader))
    x, y = batch["image"], batch["label"]
    print(f"[OK] Fold {args.fold} | train batch shape: {tuple(x.shape)} | labels sample: {y[:8].tolist()}")

    batch_val = next(iter(val_loader))
    xv, yv = batch_val["image"], batch_val["label"]
    print(f"[OK] val batch shape: {tuple(xv.shape)} | labels sample: {yv[:8].tolist()}")

    if test_loader is not None:
        batch_test = next(iter(test_loader))
        xt, yt = batch_test["image"], batch_test["label"]
        print(f"[OK] test batch shape: {tuple(xt.shape)} | labels sample: {yt[:8].tolist()}")
