"""
augmentation.py
================
Purpose:
    Define transform pipelines for three regimes:
        1) fish-only (mask-only)
        2) background-only
        3) whole-image
    plus "no-aug" / eval transforms.
"""

from torchvision.transforms import v2
from torch.nn.functional import dropout as f_dropout
import torch
import json
import itertools
from functools import partial


def identity_transform(x):
    return x


class ChannelPermute(torch.nn.Module):
    """Permute RGB channels according to a given permutation."""
    def __init__(self, perm):
        super().__init__()
        self.perm = list(perm)  # e.g. (0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, H, W]
        return x[self.perm, ...]


def channel_switch_transform(prob: float) -> v2.RandomChoice:
    """
    Randomly permute the RGB channels with probability `prob`.

    - With probability (1 - prob): keep channels as-is (identity).
    - With probability prob: apply one of the non-identity permutations
      uniformly at random.

    Args
    ----
    prob : float
        Probability of applying *any* non-identity channel permutation.

    Returns
    -------
    v2.RandomChoice
        A transform that randomly chooses between identity and
        channel-permutation transforms according to the given probabilities.
    """
    # ----- define all permutations of (0,1,2) -----
    all_perms = list(itertools.permutations([0, 1, 2]))

    # identity is (0,1,2); the rest are "real" permutations
    identity_perm = (0, 1, 2)
    non_identity_perms = [p for p in all_perms if p != identity_perm]

    # ----- build choices -----
    choices = []
    p_choices = []

    # 1) identity transform
    choices.append(ChannelPermute(identity_perm))
    p_choices.append(1.0 - prob)

    # 2) non-identity permutations, each sharing prob / N
    if non_identity_perms and prob > 0:
        per_perm_prob = prob / len(non_identity_perms)
        for perm in non_identity_perms:
            choices.append(ChannelPermute(perm))
            p_choices.append(per_perm_prob)

    # In edge cases (prob=0 or prob=1), this is still valid:
    # - prob=0  -> only identity is used
    # - prob=1  -> identity has prob 0, all perms share prob=1/5

    return v2.RandomChoice(choices, p_choices)


# --------- ADDITION ---------

def _addition_impl(x: torch.Tensor, value: int, range_val: int) -> torch.Tensor:
    """Add a random integer offset around `value` and clamp to [0,255]."""
    # keep randomness on same device as x
    offset = torch.randint(
        value - range_val,
        value + range_val + 1,
        (1,),
        device=x.device,
    ).item()
    x16 = x.to(torch.int16) + offset
    return x16.clamp(0, 255).to(torch.uint8)


def addition_transform(prob: float,
                       value: int,
                       range_val: int = 25) -> v2.RandomChoice:
    """Create an addition transformation."""
    identity = v2.Lambda(identity_transform)
    augmented = v2.Lambda(partial(_addition_impl, value=value, range_val=range_val))
    return v2.RandomChoice([identity, augmented], [1 - prob, prob])


# --------- GAUSSIAN NOISE ---------

def gaussian_noise_transform(prob: float,
                             mean: float = 0.0,
                             std: float = 1.0) -> v2.RandomChoice:
    """Create a Gaussian noise transformation."""
    identity = v2.Lambda(identity_transform)
    noise_tf = v2.GaussianNoise(mean=mean, sigma=std)
    # noise_tf is already a transform module, no lambda wrapper needed
    return v2.RandomChoice([identity, noise_tf], [1 - prob, prob])


# --------- GAUSSIAN BLUR ---------

def gaussian_blur_transform(prob: float,
                            kernel_size: int = 3,
                            sigma: float = 1.0) -> v2.RandomChoice:
    """Create a Gaussian blur transformation."""
    identity = v2.Lambda(identity_transform)
    blur_tf = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return v2.RandomChoice([identity, blur_tf], [1 - prob, prob])


# --------- DROPOUT ---------

def _dropout_impl(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Dropout that preserves uint8 dtype if input is uint8.
    """
    if x.dtype == torch.uint8:
        x_float = x.float() / 255.0
        x_drop = f_dropout(x_float, p=p, training=True)
        return (x_drop * 255.0).round().clamp(0, 255).to(torch.uint8)
    else:
        return f_dropout(x, p=p, training=True)


def dropout_transform(prob: float,
                      dropout_prob: float = 0.1) -> v2.RandomChoice:
    """Create a dropout transformation that preserves uint8 dtype."""
    identity = v2.Lambda(identity_transform)
    augmented = v2.Lambda(partial(_dropout_impl, p=dropout_prob))
    return v2.RandomChoice([identity, augmented], [1 - prob, prob])


# --------- AUGMENTATION MODULE ---------

class Augmentation(torch.nn.Module):
    """
    Configurable augmentation module that builds a torchvision v2 transform pipeline from a JSON
    configuration and applies it to either the whole image, the foreground ("fish") region, the
    background, or not at all, using a binary mask to select regions.
    This module composes a sequence of transforms based on the "pipeline" list and corresponding
    parameters under the "augmentation" mapping in the provided config file. Transforms with a
    probability of 0.0 are skipped. If all transforms are skipped, an identity transform is used.
    Parameters:
        config_file (str): Path to a JSON file describing the augmentation pipeline.
        regime (str): Region-selection mode indicating where to apply the composed transforms.
            One of:
            - "whole-image": apply to the entire image.
            - "fish-only": apply only where mask == True.
            - "background-only": apply only where mask == False.
            - "none": disable augmentation and return the input image unchanged.
    Expected JSON schema:
        {
          "pipeline": [
            "channel_switch",
            "addition",
            "gaussian_noise",
            "dropout",
            "gaussian_blur",
            "solarize",
            "equalize"
          ],
          "augmentation": {
            "channel_switch": {
              "prob": float
            },
            "addition": {
              "prob": float,
              "value": int,
              "range_val": int  # optional, default: 25
            },
            "gaussian_noise": {
              "prob": float,
              "mean": float,    # optional, default: 0.0
              "std": float      # optional, default: 1.0
            },
            "dropout": {
              "prob": float,
              "dropout_prob": float  # optional, default: 0.1
            },
            "gaussian_blur": {
              "prob": float,
              "kernel_size": int,    # optional, default: 3
              "sigma": float         # optional, default: 1.0
            },
            "solarize": {
              "prob": float,
              "threshold": int
            },
            "equalize": {
              "prob": float
            }
          }
        }
    Supported transforms:
        - channel_switch: constructed via channel_switch_transform(prob).
        - addition: constructed via addition_transform(prob, value, range_val).
        - gaussian_noise: constructed via gaussian_noise_transform(prob, mean, std).
        - dropout: constructed via dropout_transform(prob, dropout_prob).
        - gaussian_blur: constructed via gaussian_blur_transform(prob, kernel_size, sigma).
        - solarize: torchvision.transforms.v2.RandomSolarize(p, threshold).
        - equalize: torchvision.transforms.v2.RandomEqualize(p).
        If none are active (prob = 0.0 for all), an identity transform is applied.
    Inputs to forward():
        image (torch.Tensor): Image tensor of shape (C, H, W). Dtype can be uint8 in [0, 255]
            or float in [0.0, 1.0], as supported by torchvision v2 transforms.
        mask (torch.Tensor): Binary/boolean mask indicating the foreground ("fish") region.
            Shape must be broadcastable to image, e.g. (H, W) or (1, H, W). Values are converted
            to bool internally (True = foreground).
    Returns:
        torch.Tensor: Augmented image tensor with the same shape and dtype as the input image.
    Behavior:
        - regime == "whole-image": the composed transform is applied to the entire image.
        - regime == "fish-only": only pixels where mask is True are replaced by the augmented image.
        - regime == "background-only": only pixels where mask is False are replaced by the augmented image.
        - regime == "none": the input image is returned unchanged.
    Raises:
        ValueError: If an unknown regime is provided.
    Notes:
        - The functions channel_switch_transform, addition_transform, gaussian_noise_transform,
          dropout_transform, gaussian_blur_transform, and identity_transform must be available
          in the import path.
        - For reproducibility, set PyTorch and torchvision RNG seeds prior to calling forward().
        - This module operates on single images (C, H, W). If working with batches, apply it per-sample.
    Example:
        JSON (config_file):
            {
              "pipeline": ["gaussian_noise", "gaussian_blur", "equalize"],
              "augmentation": {
                "gaussian_noise": {"prob": 0.5, "mean": 0.0, "std": 0.1},
                "gaussian_blur": {"prob": 0.3, "kernel_size": 5, "sigma": 1.2},
                "equalize": {"prob": 0.2}
              }
            }
        Usage:
            aug = Augmentation(config_file="path/to/config.json", regime="fish-only")
            out = aug(image, mask)
    """
    def __init__(self, config_file: str,
                 regime: str = "whole-image") -> None:
        super().__init__()
        if regime not in ["fish-only", "background-only",
                          "whole-image", "none"]:
            raise ValueError(f"Unknown regime: {regime}")
        self.regime = regime

        with open(config_file, 'r') as f:
            cfg = json.load(f)
        aug_def = cfg["augmentation"]

        pipeline = []
        for name in cfg["pipeline"]:
            if name == "channel_switch":
                cs_params = aug_def['channel_switch']
                if cs_params['prob'] == 0.0:
                    continue
                pipeline.append(channel_switch_transform(cs_params['prob']))

            elif name == "addition":
                add_params = aug_def['addition']
                if add_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    addition_transform(
                        prob=add_params['prob'],
                        value=add_params['value'],
                        range_val=add_params.get('range_val', 25),
                    )
                )

            elif name == "gaussian_noise":
                gn_params = aug_def['gaussian_noise']
                if gn_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    gaussian_noise_transform(
                        prob=gn_params['prob'],
                        mean=gn_params.get('mean', 0.0),
                        std=gn_params.get('std', 1.0),
                    )
                )

            elif name == "dropout":
                do_params = aug_def['dropout']
                if do_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    dropout_transform(
                        prob=do_params['prob'],
                        dropout_prob=do_params.get('dropout_prob', 0.1),
                    )
                )

            elif name == "gaussian_blur":
                gb_params = aug_def['gaussian_blur']
                if gb_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    gaussian_blur_transform(
                        prob=gb_params['prob'],
                        kernel_size=gb_params.get('kernel_size', 3),
                        sigma=gb_params.get('sigma', 1.0),
                    )
                )

            elif name == "solarize":
                solarize_params = aug_def['solarize']
                if solarize_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    v2.RandomSolarize(
                        p=solarize_params['prob'],
                        threshold=solarize_params['threshold'],
                    )
                )

            elif name == "equalize":
                eq_params = aug_def['equalize']
                if eq_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    v2.RandomEqualize(p=eq_params['prob'])
                )

        if len(pipeline) == 0:
            pipeline.append(v2.Lambda(identity_transform))

        self.transforms = v2.Compose(pipeline)

    def region_transform(self, original_image: torch.Tensor,
                         augmented_image: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        if self.regime == "whole-image":
            return augmented_image
        elif self.regime == "background-only":
            return torch.where(mask.bool(), original_image, augmented_image)
        elif self.regime == "fish-only":
            return torch.where(mask.bool(), augmented_image, original_image)
        else:
            raise ValueError(f"Unknown regime: {self.regime}")

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.regime == "none":
            return image
        original_image = image.clone()
        image = self.transforms(image)
        return self.region_transform(original_image, image, mask)
