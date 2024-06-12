import glob
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from src.utils.logger import logger


@dataclass
class AddUnlabeledBackground:
    """Randomly add background noise based on medians of unlabeled soundscapes.
    Includes option to remove the average background noise of training data.
    Works with 2D spectrogram's, shape determined by the specified noise dataset.
    Adds the noise in linear, then moves back to log scale.
    """

    p: float = 0.5
    background_path: str = "./data/raw/2024/background"
    subtract_old_background: bool = False

    enabled: bool = field(default=True, init=False, repr=False)

    def __post_init__(self):
        self.files = glob.glob(f"{self.background_path}/*.npy")

        # check if path exists
        if not os.path.exists(self.background_path) or len(self.files) == 0:
            self.enabled = False
            logger.warning(f"Background path {self.background_path} does not exists, or is empty. Will crash on call.")
            return

        if self.subtract_old_background:
            self.source_background = np.exp(np.load(f"{self.background_path}/source.npy"))

        self.files = glob.glob(f"{self.background_path}/target_*.npy")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input tensor.

        :param x: torch.Tensor (B,C,H,W) spectrogram in log scale
        :return torch.Tensor (B,C,H,W) spectrogram in log scale with background noise.
        """
        if not self.enabled:
            raise ValueError("Background path does not exist.")

        x_test = x.clone()

        # to linear scale
        x = torch.exp(x)

        random_apply = torch.rand(x.shape[0]) < self.p
        random_apply = random_apply.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # subtract
        if self.subtract_old_background:
            x = x - torch.from_numpy(self.source_background).unsqueeze(0).unsqueeze(0) * random_apply

        # randomly sample files, only load if needed
        files = np.random.choice(self.files, x.shape[0])
        noise = []
        for i in range(x.shape[0]):
            if random_apply[i]:
                noise.append(torch.from_numpy(np.exp(np.load(files[i]))).unsqueeze(0))
            else:
                noise.append(torch.zeros_like(x[i]))
        x = x + torch.stack(noise) * random_apply
        x = torch.log(x)

        # min max scale the output
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        x = (x - min_) / (max_ - min_ + 1e-8)

        return x
