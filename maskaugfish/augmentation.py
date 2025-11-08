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
                       range: int = 25) -> v2.RandomChoice:
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
            x.to(torch.int16) + torch.randint(value - range,
                                              value + range + 1, (1,)
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

    def __init__(self, config_file: str,
                 regime: str = "fish-only",
                 ) -> None:
        super().__init__()
        if regime not in ["fish-only", "background-only",
                          "whole-image", "none"]:
            raise ValueError(f"Unknown regime: {regime}")
        self.regime = regime
        with open(config_file, 'r') as f:
            self.augmentation_params = json.load(f)
        self.channel_switch = channel_switch_transform(
                                self.augmentation_params['channel_switch'])
        self.addition = addition_transform(
            prob=self.augmentation_params['addition'][0],
            value=self.augmentation_params['addition'][1],
            range=25 if len(self.augmentation_params['addition']) < 3 else
            self.augmentation_params['addition'][2]
        )
        self.gaussian_noise = gaussian_noise_transform(
            prob=self.augmentation_params['gaussian_noise'][0],
            mean=0.0 if len(self.augmentation_params['gaussian_noise']) < 2
            else self.augmentation_params['gaussian_noise'][1],
            std=1.0 if len(self.augmentation_params['gaussian_noise']) < 3
            else self.augmentation_params['gaussian_noise'][2]
        )
        self.dropout = dropout_transform(
            prob=self.augmentation_params['dropout'][0],
            dropout_prob=0.1 if len(self.augmentation_params['dropout']) < 2
            else self.augmentation_params['dropout'][1]
        )
        self.gaussian_blur = gaussian_blur_transform(
            prob=self.augmentation_params['gaussian_blur'][0],
            kernel_size=3 if len(self.augmentation_params['gaussian_blur']) < 2
            else self.augmentation_params['gaussian_blur'][1],
            sigma=1.0 if len(self.augmentation_params['gaussian_blur']) < 3
            else self.augmentation_params['gaussian_blur'][2]
        )
        self.solarize = v2.RandomSolarize(
            threshold=self.augmentation_params['solarize'][0],
            p=self.augmentation_params['solarize'][1]
        )
        self.equalize = v2.RandomEqualize(
            p=self.augmentation_params['equalize']
        )

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
            raise ValueError("region_transform is only for " +
                             "'fish-only' or 'background-only' regimes.")

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.regime == "none":
            return image
        original_image = image.clone()
        image = self.solarize(image)
        image = self.channel_switch(image)
        image = self.addition(image)
        image = self.gaussian_noise(image)
        image = self.equalize(image)
        image = self.gaussian_blur(image)
        image = self.dropout(image)
        return self.region_transform(original_image, image, mask)
