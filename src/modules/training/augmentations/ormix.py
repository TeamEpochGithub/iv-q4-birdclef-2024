from dataclasses import dataclass, field
from typing import Any

import torch


def get_kornia_mix() -> Any:  # noqa: ANN401
    """Return kornia mix."""
    try:
        import kornia

    except ImportError:
        raise ImportError(
            "If you want to use this augmentation you must install kornia",
        ) from None

    else:
        return kornia.augmentation._2d.mix  # noqa: SLF001


@dataclass
class OrMix:
    """2D MixUp implementation for spectrogram data augmentation.

    :param lambda_val: The range of the mixup coefficient
    :param same_on_batch: Apply the same transformation across the batch
    :param p: The probability of applying the filter
    """

    lambda_val: tuple[float, float] = field(default=(0.0, 1.0))
    same_on_batch: bool = False
    p: float = 0.5

    def __post_init__(self) -> None:
        """Check if the filter type is valid."""
        self.mixup = get_kornia_mix().mixup.RandomMixUpV2(
            p=self.p,
            lambda_val=self.lambda_val,
            same_on_batch=self.same_on_batch,
            data_keys=["input", "class"],
        )

    def __call__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly patch the input with another sample."""
        dummy_labels = torch.arange(x.size(0))
        augmented_x, augmentation_info = self.mixup(x, dummy_labels)

        y = y.float()
        y_result = y.clone()
        for i in range(augmentation_info.shape[0]):
            y_result[i] = torch.clip(y[i] + y[int(augmentation_info[i, 1])], 0, 1)

        return augmented_x, y_result
