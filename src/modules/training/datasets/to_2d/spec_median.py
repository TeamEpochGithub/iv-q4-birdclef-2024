"""Wrapper class to create spectrograms from the data."""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torchaudio.transforms import MelSpectrogram


@dataclass
class Spec:
    """Wrapper class for spectrogram functions from torchaudio.

    :param spec: The spectrogram function to use.
    :param output_shape: The desired output shape.
    :param sequence_length: The desired sequence length.
    :param scale: The scaling function to use.
    :param sample_rate: The sample rate of the input data.
    :param f_min: The minimum frequency.
    """

    spec: functools.partial[torch.nn.Module]
    output_shape: tuple[int, int] = (224, 224)  # H, W
    sequence_length: int = 160000  # 5 seconds x 32kHz
    scale: Callable[[torch.Tensor], torch.Tensor] | None = None
    sample_rate: int = 32000
    f_min: int = 0
    f_max: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Calculate the params for the desired input shape and instantiate the spec class."""
        self.n_fft = self.output_shape[0] * 2 - 1
        self.hop_length = self.sequence_length // self.output_shape[1] + 1
        # Re-instantiate spec class with params for deisred output shape
        if self.spec.func is MelSpectrogram:
            self.instantiated_spec = self.spec(
                n_fft=self.n_fft * 4,
                hop_length=self.hop_length,
                n_mels=self.output_shape[0],
                sample_rate=self.sample_rate,
                f_min=self.f_min,
                f_max=self.f_max,
            )
        else:
            self.instantiated_spec = self.spec(n_fft=self.n_fft, hop_length=self.hop_length)

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """Create spectrograms from the input.

        :param input_data: The input data.
        :return: The spectrogram.
        """
        spec_out = self.instantiated_spec(input_data)
        if len(spec_out.shape) == 3:
            spec_out = spec_out.unsqueeze(1)
        if self.scale is None:
            self.scale = torch.log10
        spec_out = torch.nan_to_num(self.scale(spec_out + 10**-10), neginf=-10, posinf=1, nan=0)
        spec_out_median = spec_out - torch.quantile(spec_out, 0.25, dim=2, keepdim=True)
        spec_out_median -= torch.quantile(spec_out_median, 0.25, dim=3, keepdim=True)
        min_ = spec_out.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = spec_out.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        spec_out = (spec_out - min_) / (max_ - min_ + 10**-10)

        return torch.cat([spec_out, spec_out_median], dim=1)
