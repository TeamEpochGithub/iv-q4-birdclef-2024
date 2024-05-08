"""Wrapper class to create spectrograms from the data."""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torchaudio.transforms import MelSpectrogram


@dataclass
class Spec:
    """Wrapper class for spectrogram functions from torchaudio."""

    spec: functools.partial[torch.nn.Module]
    output_shape: tuple[int, int] = (224, 224)  # H, W
    seqeunce_length: int = 160000  # 5 seconds x 32kHz
    scale: Callable[[torch.Tensor], torch.Tensor] = torch.log10
    sample_rate: int = 32000

    def __post_init__(self) -> None:
        """Calculate the params for the desired input shape and instantiate the spec class."""
        self.n_fft = self.output_shape[0] * 2 - 1
        self.hop_length = self.seqeunce_length // self.output_shape[1] + 1
        # Re-instantiate spec class with params for deisred output shape
        if self.spec.func is MelSpectrogram:
            self.instantiated_spec = self.spec(
                n_fft=self.n_fft * 12,
                hop_length=self.hop_length,
                n_mels=self.output_shape[0]*3,
                sample_rate=self.sample_rate,
            )
        else:
            self.instantiated_spec = self.spec(n_fft=self.n_fft, hop_length=self.hop_length)

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """Create spectrograms from the input."""
        # Create spectrograms from the input

        spec_out = self.instantiated_spec(input_data)

        if len(spec_out.shape) == 3:
            spec_out = spec_out.unsqueeze(1)

        rgb_spec = torch.empty([spec_out.shape[0], spec_out.shape[1]*3, spec_out.shape[2]//3, spec_out.shape[3]], device=spec_out.device)
        rgb_spec[:,0,:,:] = spec_out[:,0,0:self.output_shape[0], :]
        rgb_spec[:,1,:,:] = spec_out[:,0,self.output_shape[0]:2*self.output_shape[0], :]
        rgb_spec[:,2,:,:] = spec_out[:,0,2*self.output_shape[0]:3*self.output_shape[0], :]
        # log10 returns some -inf's replace them with low amplitude
        return torch.nan_to_num(self.scale(rgb_spec + 10**-10), neginf=-10, posinf=1, nan=0)
