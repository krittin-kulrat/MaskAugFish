# dataloader.py
# Fish4Knowledge dataloader with WeightedRandomSampler + Stratified 5-fold CV

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedKFold


# -----------------------------
# Helpers
# -----------------------------
def _find_subdir(root: Path, prefer: str, alt: str) -> Path:
    """Return root/prefer if exists; otherwise root/alt."""
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
    """Build a (image_path, mask_path_or_none, class_id) list and class mapping."""
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
class FishDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, Optional[str], int]],
        img_size: int = 224,
        train: bool = True,
    ):
        self.samples = samples

        if train:
            self.tf_img = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.tf_img = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )

        self.tf_mask = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.NEAREST
                ),
            ]
        )

        self.y = np.array([lbl for _, _, lbl in samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.tf_img(img)

        out: Dict[str, torch.Tensor | str] = {
            "image": img,
            "label": torch.as_tensor(label, dtype=torch.long),
            "path": img_path,
        }

        if mask_path is not None and Path(mask_path).exists():
            mask_img = Image.open(mask_path)
            mask_img = self.tf_mask(mask_img)
            mask = torch.from_numpy(np.array(mask_img, dtype=np.int64))
            out["mask"] = mask

        return out


# -----------------------------
# Sampler / folds
# -----------------------------
def make_sample_weights(labels: np.ndarray, clip_max: float = 50.0) -> torch.Tensor:
    """Compute per-sample weights: w_i = 1 / count(class_of_i)."""
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
    weighted_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for a given fold with optional weighted sampling."""
    ds_all = FishDataset(samples, img_size=img_size, train=True)
    y = ds_all.y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    assert 0 <= fold_id < n_splits, f"fold_id must be in [0, {n_splits - 1}]"
    tr_idx, va_idx = splits[fold_id]

    tr_samples = [samples[i] for i in tr_idx]
    va_samples = [samples[i] for i in va_idx]

    train_ds = FishDataset(tr_samples, img_size=img_size, train=True)
    val_ds = FishDataset(va_samples, img_size=img_size, train=False)

    if weighted_train:
        weights = make_sample_weights(y[tr_idx])
        sampler = WeightedRandomSampler(
            weights, num_samples=len(tr_idx), replacement=True
        )
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

    return train_loader, val_loader


# -----------------------------
# CLI test
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/Fish4Knowledge", help="dataset root")
    ap.add_argument("--fold", type=int, default=0, help="which fold [0..4]")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    print(f"[INFO] Scanning dataset at: {args.root}")
    samples, class_to_id = build_index(args.root)
    print(f"[INFO] Found {len(class_to_id)} classes, {len(samples)} samples")

    train_loader, _ = make_fold_loaders(
        samples,
        batch_size=args.bs,
        n_splits=5,
        fold_id=args.fold,
        img_size=args.img_size,
    )

    batch = next(iter(train_loader))
    x = batch["image"]
    y = batch["label"]
    print(f"[OK] Fold {args.fold} | train batch shape: {x.shape} | labels: {y[:8].tolist()}")
