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
from torch.nn.functional import dropout
import torch
import json
import itertools


def channel_switch_transform(prob: float
                            ) -> v2.RandomChoice:
    """Create a channel switch transformation

    Args:
        prob (float): Probability of applying the transformation

    Returns:
        v2.RandomChoice: Channel switch transformation
    """
    choices = []
    p_choices = []
    perms = list(itertools.permutations([0, 1, 2]))
    for i, perm in enumerate(perms):
        choices.append(v2.Lambda(lambda x, p=perm: x[list(p), ...]))
        if i == 0:
            p_choices.append(1 - prob)
        else:
            p_choices.append(prob / (len(perms) - 1))
    return v2.RandomChoice(choices, p_choices)


def addition_transform(prob: float,
                       value: int,
                       range_val: int = 25) -> v2.RandomChoice:
    """Create an addition transformation

    Args:
        prob (float): Probability of applying the transformation
        value (int): Base value to add
        range (int, optional): Range around the base value. Defaults to 25.

    Returns:
        v2.RandomChoice: Addition transformation
    """
    choices = [v2.Lambda(lambda x: x)]
    augmented = v2.Lambda(
        lambda x: (
            x.to(torch.int16) + torch.randint(value - range_val,
                                              value + range_val + 1, (1,)
                                              ).item()
                                              ).clamp(0, 255).to(torch.uint8))
    choices.append(augmented)
    return v2.RandomChoice(choices, [1 - prob, prob])


def gaussian_noise_transform(prob: float,
                             mean: float = 0.0,
                             std: float = 1.0) -> v2.RandomChoice:
    """Create a Gaussian noise transformation

    Args:
        prob (float): Probability of applying the transformation
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.0.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to 1.0.

    Returns:
        v2.RandomChoice: Gaussian noise transformation
    """
    choices = [v2.Lambda(lambda x: x)]
    noise_ = v2.GaussianNoise(mean=mean, sigma=std)
    augmented = v2.Lambda(lambda x: noise_(x))
    choices.append(augmented)
    return v2.RandomChoice(choices, [1 - prob, prob])


def gaussian_blur_transform(prob: float,
                            kernel_size: int = 3,
                            sigma: float = 1.0) -> v2.RandomChoice:
    """Create a Gaussian blur transformation

    Args:
        prob (float): Probability of applying the transformation
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 3.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.

    Returns:
        v2.RandomChoice: Gaussian blur transformation
    """
    choices = [v2.Lambda(lambda x: x)]
    blur_ = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    augmented = v2.Lambda(lambda x: blur_(x))
    choices.append(augmented)
    return v2.RandomChoice(choices, [1 - prob, prob])


def dropout_transform(prob: float,
                      dropout_prob: float = 0.1) -> v2.RandomChoice:
    """Create a dropout transformation that preserves uint8 dtype.

    If input is uint8: convert to float (0-1), apply dropout, rescale to uint8.
    If input is float: apply dropout directly, preserve dtype.

    Args:
        prob (float): Probability of applying the transformation.
        dropout_prob (float): Dropout probability.

    Returns:
        v2.RandomChoice
    """
    def _apply(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x_float = x.float() / 255.0
            x_drop = dropout(x_float, p=dropout_prob, training=True)
            x_out = (x_drop * 255.0).round().clamp(0, 255).to(torch.uint8)
            return x_out
        else:
            return dropout(x, p=dropout_prob, training=True)

    choices = [v2.Lambda(lambda x: x)]
    augmented = v2.Lambda(_apply)
    return v2.RandomChoice(choices + [augmented], [1 - prob, prob])


class Augmentation(torch.nn.Module):
    """
    A configurable augmentation module that constructs a transformation pipeline
    from a JSON configuration file. The class supports region-aware augmentation,
    allowing augmentations to be applied selectively to the fish-only region,
    background-only region, or the entire image based on a provided mask.

    The augmentation pipeline is dynamically built from a list of operations
    specified under `cfg["pipeline"]` in the configuration file. Each operation
    references a corresponding augmentation definition in `cfg["augmentation"]`.
    Augmentations with probability=0.0 are automatically skipped.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file defining available augmentations
        and the sequence of operations.
    regime : str, optional
        Controls how the augmented image is blended with the original image.
        Must be one of:
        - "whole-image": apply augmentations to the entire image
        - "fish-only": apply augmentations to foreground (mask=True)
        - "background-only": apply augmentations to background (mask=False)
        - "none": disable augmentation entirely
        Default is "whole-image".

    Notes
    -----
    - If all augmentations are disabled or removed (prob=0), the module falls
      back to an identity transform.
    - This design enables flexible experimentation with augmentation ordering
      by editing only the JSON file.
    """
    def __init__(self, config_file: str,
                 regime: str = "whole-image",
                 ) -> None:
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
                if aug_def['channel_switch']['prob'] == 0.0:
                    continue
                pipeline.append(
                    channel_switch_transform(
                        aug_def['channel_switch']['prob'])
                )
            elif name == "addition":
                addition_params = aug_def['addition']
                if addition_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    addition_transform(
                        prob=addition_params['prob'],
                        value=addition_params['value'],
                        range_val=25 if 'range_val' not in addition_params
                        else addition_params['range_val']
                    )
                )
            elif name == "gaussian_noise":
                gn_params = aug_def['gaussian_noise']
                if gn_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    gaussian_noise_transform(
                        prob=gn_params['prob'],
                        mean=0.0 if 'mean' not in gn_params
                        else gn_params['mean'],
                        std=1.0 if 'std' not in gn_params
                        else gn_params['std']
                    )
                )
            elif name == "dropout":
                do_params = aug_def['dropout']
                if do_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    dropout_transform(
                        prob=do_params['prob'],
                        dropout_prob=0.1 if 'dropout_prob' not in do_params
                        else do_params['dropout_prob']
                    )
                )
            elif name == "gaussian_blur":
                gb_params = aug_def['gaussian_blur']
                if gb_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    gaussian_blur_transform(
                        prob=gb_params['prob'],
                        kernel_size=3 if 'kernel_size' not in gb_params
                        else gb_params['kernel_size'],
                        sigma=1.0 if 'sigma' not in gb_params
                        else gb_params['sigma']
                    )
                )
            elif name == "solarize":
                solarize_params = aug_def['solarize']
                if solarize_params['prob'] == 0.0:
                    continue
                pipeline.append(
                    v2.RandomSolarize(
                        p=solarize_params['prob'],
                        threshold=solarize_params['threshold']
                    )
                )
            elif name == "equalize":
                if aug_def['equalize']['prob'] == 0.0:
                    continue
                pipeline.append(
                    v2.RandomEqualize(
                        p=aug_def['equalize']['prob']
                    )
                )
        if len(pipeline) == 0:
            pipeline.append(v2.Lambda(lambda x: x))
        self.transforms = v2.Compose(pipeline)

    def region_transform(self, original_image: torch.Tensor,
                         augmented_image: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """
        Blend an augmented image with the original according to the configured regime and mask.
        Parameters
        - original_image: torch.Tensor
            The unmodified input image. Shape (..., C, H, W) or (C, H, W).
        - augmented_image: torch.Tensor
            The image after applying the augmentation pipeline. Same shape as original_image.
        - mask: torch.Tensor
            Foreground/background mask. Must be boolean or convertible to boolean and
            broadcastable to the image shape. True indicates foreground (fish).
        Returns
        - torch.Tensor
            The composited image where pixels are selected from original or augmented
            according to the regime:
            - "whole-image": augmented_image
            - "background-only": original where mask is True, augmented otherwise
            - "fish-only": augmented where mask is True, original otherwise
        """
        if self.regime == "whole-image":
            return augmented_image
        elif self.regime == "background-only":
            return torch.where(mask.bool(), original_image, augmented_image)
        elif self.regime == "fish-only":
            return torch.where(mask.bool(), augmented_image, original_image)
        else:
            raise ValueError(f"Unknown regime: {self.regime}")

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation pipeline to the input image and optionally blend
        it with the original based on the region-aware augmentation regime.

        Parameters
        ----------
        image : torch.Tensor
            Input tensor representing an image. Expected shape is (C, H, W).
        mask : torch.Tensor
            Foreground mask used for region-specific augmentation.

        Returns
        -------
        torch.Tensor
            Augmented (or partially augmented) image. If `regime="none"`,
            the function returns the original image unchanged.

        Notes
        -----
        - The augmentation pipeline is built using torchvision v2 transforms.
        - A clone of the input image is preserved for blending purposes.
        """
        if self.regime == "none":
            return image
        original_image = image.clone()
        image = self.transforms(image)
        return self.region_transform(original_image, image, mask)
