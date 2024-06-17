from dataclasses import dataclass

import torch
from audiomentations import Compose

from src.utils.recursive_repr import recursive_repr


@dataclass
class AudiomentationsCompose:
    compose: Compose = None
    sr: int = 32000

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            augmented_x[i] = torch.from_numpy(self.compose(x[i].squeeze().numpy(), self.sr))
        return augmented_x

    def __repr__(self) -> str:
        out = ""
        for field in self.compose.__dict__["transforms"]:
            out += recursive_repr(field)
        return out
