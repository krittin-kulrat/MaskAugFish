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
from typing import List, Tuple


def generate_channel_switch(prob: float
                            ) -> Tuple[List[v2.Lambda], List[float]]:
    """Generate input for channel switch augmentation
    based on probability

    Args:
        prob (float): Probability of applying the transformation

    Returns:
        choices (List[v2.Lambda]): Channel switch
        p_choices (List[float]): Probabilities for each channel switch
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
    return choices, p_choices


def addition_transform(value: int,
                       range: int = 25) -> v2.Lambda:
    """Create an addition transformation

    Args:
        value (int): Mean value that will be added to the Tensor
        range (int, optional): Range of the random variation. Defaults to 25.

    Returns:
        v2.Lambda: Addition transformation
    """
    variation = torch.randint(value - range, value + range + 1, (1,)).item()
    return v2.Lambda(
        lambda x: (
            x.to(torch.int16) + variation).clamp(0, 255).to(torch.uint8))


def region_transform(image: torch.Tensor,
                     mask: torch.Tensor,
                     transform: v2.Compose,) -> torch.Tensor:
    original_image = image.clone()
    transformed_image = transform(image)
    return torch.where(mask.bool(), transformed_image, original_image)


class Augmentation(torch.nn.Module):

    def __init__(self, config_file: str,
                 regime: str = "fish-only",
                 seed: int = 42
                 ) -> None:
        super().__init__()
        self.regime = regime
        with open(config_file, 'r') as f:
            self.augmentation_params = json.load(f)
        self.channel_switch = v2.RandomChoice(*generate_channel_switch(
                                self.augmentation_params['channel_switch']))

        if self.augmentation_params['addition'] is not False:
            self.addition = addition_transform(
                self.augmentation_params['addition'], range=25)
        else:
            self.addition = v2.Lambda(lambda x: x)

    def forward(self, image):
        image = self.channel_switch(image)
        if self.addition is not None:
            image = self.addition(image)
        return image
