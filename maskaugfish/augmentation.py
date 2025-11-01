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


def addition_transform(prob: float,
                       value: int,
                       range: int = 25) -> v2.RandomChoice:
    """Create an addition transformation

    Args:
        value (int): Mean value that will be added to the Tensor
        range (int, optional): Range of the random variation. Defaults to 25.

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
        self.channel_switch = v2.RandomChoice(*generate_channel_switch(
                                self.augmentation_params['channel_switch']))
        self.addition = addition_transform(
            prob=self.augmentation_params['addition'][0],
            value=self.augmentation_params['addition'][1],
            range=25 if len(self.augmentation_params['addition']) < 3 else
            self.augmentation_params['addition'][2]
        )

    def switch_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return ((mask + 1) * 255)

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.regime == "none":
            return image
        original_image = image.clone()
        image = self.channel_switch(image)
        image = self.addition(image)
        # TODO: add other augmentations here
        if self.regime == "whole-image":
            return image
        elif self.regime == "background-only":
            mask = self.switch_mask(mask)
            return torch.where(mask.bool(), original_image, image)
        elif self.regime == "fish-only":
            return torch.where(mask.bool(), image, original_image)
        else:
            raise ValueError(f"Unknown regime: {self.regime}")
