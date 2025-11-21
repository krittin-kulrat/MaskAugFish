## Quick context for AI coding agents

- Purpose: research codebase for mask-aware data augmentation for fish species recognition (Fish4Knowledge dataset).
- Key scripts: `train_vscode.py` (self-contained training harness), `training.py` (alternative training loop expecting a teammate `dataloader`), and package `maskaugfish/` for modular utilities.

## Architecture & data flow (high level)

- Data: expected layout contains image folders and optional mask folders. Example paths used in code:
  - `data/f4k/images/` and `data/f4k/masks/`
  - top-level helper variants: `FishImages/` and `mask_image/` exist in the repo.
- Dataset: `FishMaskDataset` (in `train_vscode.py`) builds train/val/test splits and returns tensors. It implements mask-aware photometric augmentation with `scope` parameter: `whole`, `mask`, `background`.
- Dataloader helpers: `make_loaders` returns (train, val, test) and optionally uses a `WeightedRandomSampler` to handle class imbalance.
- Models: ResNet backbones (resnet18/resnet34) and EfficientNet-B0 are supported; classifier head replaced to match the dataset classes.
- Training loops exist in two places:
  - `train_vscode.py`: standalone, includes dataset, loaders, model selection, losses (CE, weighted_ce, focal), and plotting of confusion matrix.
  - `training.py`: teammate-style training that imports `dataloader.build_index` and `make_dataloaders` — expect slightly different loader API (samples, labels arrays).

## Project-specific conventions and patterns

- Mask file naming: mask images expected as PNGs with the same stem as input image (e.g., `img001.jpg` -> `img001.png` mask). Masks are single-channel (L) and thresholded (`>127`) to binary.
- Augmentation scope: photometric augmentation is applied via `scope` in dataset. Implementation pattern in `train_vscode.py` uses numpy compositing of augmented and original images depending on `mask` array.
- Imbalance handling: class-weighted losses and WeightedRandomSampler are commonly used. Look for flags `--no_sampler` and loss choices `ce|weighted_ce|focal`.
- Weight initialization: when `--pretrained imagenet` is selected model weights come from torchvision `Weights.DEFAULT` constants.

## How to run & debug (concrete examples)

Use PowerShell (project root is workspace):

```powershell
# Quick training (mask-aware background augmentation)
python .\train_vscode.py --img_dir data\f4k\images --mask_dir data\f4k\masks --scope background --epochs 12 --batch 32

# Run the other training loop (teammate loader expected in repository root)
python .\training.py --data_root . --phase feature --aug --epochs 12 --batch 32 --img_size 224
```

Notes:
- `train_vscode.py` is self-contained: it implements dataset, loaders, training, and evaluation. Prefer it for fast iteration.
- `training.py` expects helper functions from a `dataloader` module with a different API (`build_index`, `make_dataloaders`). Use when integrating with that teammate API.

## Files to inspect for changes or extensions

- `train_vscode.py` — dataset implementation and quick experiments (recommended starting point).
- `maskaugfish/augmentation.py` — intended to centralize transforms/pipelines (currently minimal but canonical place to extend augmentation strategies).
- `maskaugfish/dataloader.py` — team-style dataloader utilities; inspect when switching between the two training loops.
- `maskaugfish/training.py` — alternate training utilities (keeps training/eval logic modular).

## Common edits an agent may be asked to perform (examples)

- Add a new augmentation: implement it in `maskaugfish/augmentation.py` and wire it into `train_vscode.py` by replacing `self.photometric` or adding a `scope`-aware pipeline.
- Add a new backbone: extend `get_model()` in `train_vscode.py` and map weights/classifier adaptation similarly to existing branches.
- Fix dataset path assumptions: ensure masks exist and fallback behavior (when a mask is missing, `mask_path` is None and augmentation falls back to whole-image behavior).

## Debugging and pitfalls to watch for

- VSCode launch config: `.vscode/launch.json` contains an entry with `"type": "debuggy"` — likely a typo; change to `"python"` for the Python debugger to work.
- Watch image/mask alignment: `train_vscode.py` resizes masks to image size using nearest neighbor — be careful when adding augmentations that change geometry.
- Pay attention to different loader APIs: `train_vscode.py` returns (x,label) tuples; `training.py` assumes DataLoader yields dicts with keys like `image` and `label`. Merge carefully.

## Minimal contract for changes

- Input: small changes should preserve dataset split seeds (`--seed`) and not silently change default `img_size` or `pretrained` flags.
- Output: saved experiment artifacts go into `runs/YYYYMMDD-HHMMSS*` (check both `train_vscode.py` and `training.py`). Keep that behavior unless feature requires new artifact layout.

---

If you want, I can also:
- merge this into an existing `.github/copilot-instructions.md` if one exists, or
- add short examples (unit tests) that validate dataset/mask alignment.

Please tell me if you'd like more detail about any part (loader APIs, exact flag combinations to reproduce published results, or to fix the `launch.json` typo).
