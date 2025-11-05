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


class Augmentation(torch.nn.Module):

    def __init__(self, config_file: str,
                 regime: str = "fish-only",
                 seed: int = 42
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
        # TODO: add other augmentations here

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
        image = self.channel_switch(image)
        image = self.addition(image)
        # TODO: add other augmentations here
        return self.region_transform(original_image, image, mask)
