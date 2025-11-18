# dataloader.py
# Fish4Knowledge dataloader with:
# - torchvision.io.read_file + decode_image (tensor I/O)
# - torchvision.transforms.v2-style pipelines
# - StratifiedShuffleSplit (hold-out test)
# - StratifiedKFold (train/val CV on remaining pool)
# - Optional WeightedRandomSampler for class balance
# - Basic sanity prints (class freq, reproducibility)
#
# Usage from training code:
#
#   from maskaugfish.dataloader import build_index, make_dataloaders
#
#   samples, class_to_id = build_index("data/Fish4Knowledge")
#
#   aug_train = v2.Compose([
#       v2.RandomHorizontalFlip(p=0.5),
#       # add more if needed
#   ])
#
#   train_loader, val_loader, test_loader = make_dataloaders(
#       samples,
#       batch_size=8,
#       num_workers=0,
#       img_size=224,
#       augment_pipeline=aug_train,
#       weighted_train=True,
#       seed=42,
#       test_ratio=0.2,
#       n_splits=5,
#       fold_id=0,
#   )
#
#   batch = next(iter(train_loader))
#   x, y = batch["image"], batch["label"]
#   print("train:", x.shape, y[:8].tolist())


from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import torchvision.io as tvio
from torchvision.io.image import ImageReadMode  # for decode_image mode

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


# -----------------------------
# Helpers
# -----------------------------

def _find_subdir(root: Path, prefer: str, alt: str) -> Path:
    """
    Returns prefer if it exists, else alt.
    """
    p1, p2 = root / prefer, root / alt
    return p1 if p1.exists() else p2


def list_images(dirpath: Path) -> List[Path]:
    """
    Recursively list image files under dirpath with common extensions.
    """
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
    """
    Scan dataset folders and build a list of samples.

    We assume directory structure like:
        root/
          fish_images/ (or fish_image/)
            class_A/
               img1.png
               img2.png
            class_B/
               ...
          fish_masks/ (or mask_image/)
            class_A/
               img1.png   (mask has same stem as img)
               ...
            class_B/
               ...

    Returns:
      samples: list of (img_path:str, mask_path:str|None, class_id:int)
      class_to_id: {class_name: class_id}
    """
    root = Path(root)
    img_root = _find_subdir(root, image_sub, alt_image_sub)
    mask_root = _find_subdir(root, mask_sub, alt_mask_sub)

    classes = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    class_to_id = {c: i for i, c in enumerate(classes)}

    # build lookup: stem -> mask_path
    mask_lookup: Dict[str, Path] = {}
    if mask_root.exists():
        for mp in list_images(mask_root):
            mask_lookup[mp.stem] = mp

    samples: List[Tuple[str, Optional[str], int]] = []
    for c in classes:
        cls_id = class_to_id[c]
        cls_dir = img_root / c
        for ip in list_images(cls_dir):
            mp = mask_lookup.get(ip.stem.replace('fish', 'mask'))
            samples.append((str(ip), str(mp) if mp is not None else None, cls_id))

    return samples, class_to_id


# -----------------------------
# Basic transforms
# -----------------------------

def make_img_base_tf(img_size: int) -> v2.Compose:
    """
    Resize + convert dtype to float32 in [0,1].
    Assumes input is already torch.Tensor [C,H,W] uint8 in [0..255].
    We do NOT bake in augmentation here.
    """
    return v2.Compose([
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
    ])


def make_mask_tf(img_size: int) -> v2.Compose:
    """
    For segmentation mask / instance mask.
    NEAREST to avoid interpolating class IDs.
    We'll later cast to long.
    """
    return v2.Compose([
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
    ])


# -----------------------------
# Dataset
# -----------------------------

class FishDataset(Dataset):
    """
    A dataset that returns a dict:
      {
        "image": FloatTensor [3,H,W] in [0,1],
        "label": LongTensor scalar class id,
        "path":  str path to the image,
        "mask":  LongTensor [H,W] of class indices (optional, only if mask exists)
      }

    Notes:
    - We accept an optional `augment` pipeline (v2.Compose or callable).
      If a mask exists, augmentation is applied jointly as (img, mask) -> (img, mask).
      Otherwise, we call it as augment(img).
      This runs BEFORE base resizing/dtype.
    """

    def __init__(
        self,
        samples: List[Tuple[str, Optional[str], int]],
        img_size: int = 224,
        augment: Optional[v2.Compose] = None,
    ):
        self.samples = samples
        self.augment = augment
        self.base_tf_img = make_img_base_tf(img_size)
        self.tf_mask = make_mask_tf(img_size)

        # cache labels for sampler / sanity checks
        self.y = np.array([lbl for _, _, lbl in samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_uint8(path: str, gray: bool = False) -> torch.Tensor:
        """
        Load encoded bytes then decode to uint8 tensor [C,H,W].
        Replaces obsolete tvio.read_image.
        """
        data = tvio.read_file(path)
        mode = ImageReadMode.GRAY if gray else ImageReadMode.RGB
        return tvio.decode_image(data, mode=mode)  # uint8 [C,H,W]

    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]

        # --- load RGB image as tensor ---
        img = self._read_uint8(img_path, gray=False)  # (3,H,W) uint8

        # --- optionally load mask (assume path is valid when provided) ---
        mask_raw = None
        if mask_path is not None:  # NOTE: removed Path(mask_path).exists() per reviewer
            mask_raw = self._read_uint8(mask_path, gray=True)  # [1,H,W] uint8

        # apply optional augment first (on uint8)
        if self.augment is not None:
            img = self.augment(img, mask_raw)   # joint augment

        # then resize + to float32 [0,1]
        img = self.base_tf_img(img)

        out = {
            "image": img,
            "label": torch.as_tensor(label, dtype=torch.long),
            "path": img_path,
        }

        # resize + cast mask if present
        if mask_raw is not None:
            mask_resized = self.tf_mask(mask_raw)           # [1,H,W] uint8
            mask_tensor = mask_resized.squeeze(0).to(torch.long)  # [H,W] long
            out["mask"] = mask_tensor

        return out


# -----------------------------
# Class balancing weights
# -----------------------------

def make_sample_weights(labels: np.ndarray, clip_max: float = 50.0) -> torch.Tensor:
    """
    Build per-sample weights ~ 1/freq(class).
    clip_max prevents insanely large weights for ultra-rare classes.
    """
    classes, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))

    w = np.array([1.0 / freq[int(y)] for y in labels], dtype=np.float32)
    w = w / w.mean()
    if clip_max is not None:
        w = np.clip(w, a_min=None, a_max=clip_max)
    return torch.from_numpy(w)


# -----------------------------
# DataLoader builder
# -----------------------------

def make_dataloaders(
    samples: List[Tuple[str, Optional[str], int]],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    augment_pipeline: Optional[v2.Compose] = None,
    weighted_train: bool = True,
    seed: int = 42,
    test_ratio: float = 0.2,
    n_splits: int = 5,
    fold_id: int = 0,
):
    """
    Full pipeline:

    1. StratifiedShuffleSplit -> get a fixed hold-out test set
       (size = test_ratio), and the remaining pool for training/validation.

    2. StratifiedKFold on that remaining pool -> pick one fold_id
       as validation and the rest as training.

    Returns:
      train_loader, val_loader, test_loader
    """

    # -----------------
    # Step 0: labels
    # -----------------
    y_all = np.array([lbl for _, _, lbl in samples], dtype=np.int64)

    # -----------------
    # Step 1: Train/Test split
    # -----------------
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed,
    )

    train_pool_idx, test_idx = next(sss.split(np.zeros(len(y_all)), y_all))

    train_pool_samples = [samples[i] for i in train_pool_idx]
    test_samples = [samples[i] for i in test_idx]

    y_pool = np.array([lbl for _, _, lbl in train_pool_samples], dtype=np.int64)

    # -----------------
    # Step 2: K-fold on the train_pool for train/val
    # -----------------
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(skf.split(np.zeros(len(y_pool)), y_pool))
    assert 0 <= fold_id < n_splits, f"fold_id must be in [0,{n_splits-1}]"
    tr_idx, va_idx = splits[fold_id]

    tr_samples = [train_pool_samples[i] for i in tr_idx]
    va_samples = [train_pool_samples[i] for i in va_idx]

    # -----------------
    # Step 3: Datasets
    # -----------------
    train_ds = FishDataset(
        tr_samples,
        img_size=img_size,
        augment=augment_pipeline,   # augment only on train (joint if mask exists)
    )
    val_ds = FishDataset(
        va_samples,
        img_size=img_size,
        augment=None,               # no augment on val
    )
    test_ds = FishDataset(
        test_samples,
        img_size=img_size,
        augment=None,               # no augment on test
    )

    # -----------------
    # Step 4: Samplers / Loaders
    # -----------------
    if weighted_train:
        weights = make_sample_weights(train_ds.y)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_ds),
            replacement=True,
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
# CLI / sanity check runner
# -----------------------------

if __name__ == "__main__":
    import argparse
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="dataset root containing fish_images/ and fish_masks/")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2,
                    help="fraction of data held out as final test set")
    ap.add_argument("--n_splits", type=int, default=5,
                    help="number of CV folds on the train pool")
    ap.add_argument("--fold", type=int, default=0,
                    help="which CV fold to use as validation [0..n_splits-1]")
    ap.add_argument("--no_weighted", action="store_true",
                    help="disable weighted sampler for training loader")
    args = ap.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[INFO] Scanning dataset at: {args.root}")
    samples, class_to_id = build_index(args.root)
    print(f"[INFO] Found {len(class_to_id)} classes, {len(samples)} total samples")

    # Class frequency check (overall)
    ys_all = [s[2] for s in samples]
    freq_all = dict(sorted(Counter(ys_all).items()))
    print(f"[CHECK] Overall class frequencies: {freq_all}")

    train_loader, val_loader, test_loader = make_dataloaders(
        samples=samples,
        batch_size=args.bs,
        num_workers=args.num_workers,
        img_size=args.img_size,
        augment_pipeline=None,
        weighted_train=not args.no_weighted,
        seed=args.seed,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        fold_id=args.fold,
    )

    # Quick sanity batches
    train_batch = next(iter(train_loader))
    xtr, ytr = train_batch["image"], train_batch["label"]
    print(f"[OK] train batch: {tuple(xtr.shape)} | labels sample: {ytr[:8].tolist()}")

    val_batch = next(iter(val_loader))
    xva, yva = val_batch["image"], val_batch["label"]
    print(f"[OK]   val batch: {tuple(xva.shape)} | labels sample: {yva[:8].tolist()}")

    test_batch = next(iter(test_loader))
    xte, yte = test_batch["image"], test_batch["label"]
    print(f"[OK]  test batch: {tuple(xte.shape)} | labels sample: {yte[:8].tolist()}")

    # Repro check:
    print("[CHECK] Reproducibility check:")
    y_all_np = np.array(ys_all, dtype=np.int64)

    sss_a = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio,
                                   random_state=args.seed)
    train_pool_idx_a, test_idx_a = next(sss_a.split(np.zeros(len(y_all_np)), y_all_np))

    sss_b = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio,
                                   random_state=args.seed)
    train_pool_idx_b, test_idx_b = next(sss_b.split(np.zeros(len(y_all_np)), y_all_np))

    same_test = np.array_equal(test_idx_a, test_idx_b)
    same_pool = np.array_equal(train_pool_idx_a, train_pool_idx_b)

    print(f"  same train_pool/test split across runs? "
          f"train_pool={same_pool}, test={same_test}")

    pool_labels = y_all_np[train_pool_idx_a]
    skf_a = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    skf_b = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits_a = list(skf_a.split(np.zeros(len(pool_labels)), pool_labels))
    splits_b = list(skf_b.split(np.zeros(len(pool_labels)), pool_labels))

    tr_a, va_a = splits_a[args.fold]
    tr_b, va_b = splits_b[args.fold]

    print(f"  same train/val fold indices within train_pool? "
          f"train={np.array_equal(tr_a, tr_b)}, val={np.array_equal(va_a, va_b)}")
