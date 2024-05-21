"""Time series augmentations for PyTorch tensors."""

from dataclasses import dataclass

import torch


@dataclass
class Scale(torch.nn.Module):
    """Exponential scale 1d augmentation.

    Randomly scales the input signal by a factor sampled from a log-uniform distribution.
    """

    p: float = 0.5
    lower: float = 1e-5
    higher: float = 1e2

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input features and labels.

        :param x: Input features. (N,C,L)|(N,L)
        :param y: Input labels. (N,C)
        :return: The augmented features and labels
        """
        # Randomly sample between 10e-5 and 10e2

        if torch.rand(1) < self.p:
            scale = torch.exp(torch.rand(1) * (torch.log(torch.tensor(self.higher)) - torch.log(torch.tensor(self.lower))) + torch.log(torch.tensor(self.lower)))
            x = x * scale
        return x, y

@dataclass
class EnergyCutmix(torch.nn.Module):
    """Instead of taking the rightmost side from the donor sample, take the highests energy sample.
        Modified implementation of cutmix1d from epochalyst."""
    p: float = 0.5
    low: float = 0.25
    high: float = 0.75

    # def find_minmax_energy_windows(self, audio_tensor: torch.Tensor, window_size: int, stride: int):
    #     # Compute the squared values of the audio tensor
    #     squared_audio = audio_tensor ** 2     
        
    #     # Apply convolution to compute the sliding window sums
    #     energy_windows = torch.nn.functional.avg_pool1d(squared_audio, window_size, stride=stride, padding=0)
        
    #     # Remove the unnecessary dimensions
    #     energy_windows = energy_windows.squeeze()
        
    #     # The length of the energy_windows will be len(audio_tensor) - window_size + 1
    #     # Compute the start and end indices for the valid range
    #     valid_length = energy_windows.size(0)
        
    #     # Find the maximum energy window and its index
    #     max_energy, max_index = torch.max(energy_windows, dim=0)
    #     min_energy, min_index = torch.min(energy_windows, dim=0)

    #     # Compute the start and end indices for the max energy window
    #     start_index_max = max_index.item() * stride
    #     end_index_max = start_index_max + window_size
    #     # Now for the min window
    #     start_index_min = min_index.item() * stride
    #     end_index_min = start_index_min + window_size
        
    #     return max_energy.item(), (start_index_max, end_index_max) , min_energy.item(), (start_index_min, end_index_min)

    def find_window(self, donor, receiver, window_size, stride):
        """Extract the strongest window from donor and the weakest from th receiver. Return the indices for both windows."""
        donor_power = donor ** 2
        receiver_power = receiver ** 2
        
        # Extract the energies per window
        donor_energies = torch.nn.functional.avg_pool1d(donor_power, window_size, stride=stride, padding=0)
        receiver_energies = torch.nn.functional.avg_pool1d(receiver_power, window_size, stride=stride, padding=0)

        # Get the indices for the max and min
        max_donor, donor_index = torch.max(donor_energies, dim=-1)
        min_receiver, receiver_index = torch.min(receiver_energies, dim=-1)
        # Get windows
        donor_start = donor_index.item() * stride
        donor_end = donor_start + window_size
        receiver_start = receiver_index.item() * stride
        receiver_end = receiver_start + window_size

        return donor_start, donor_end, receiver_start, receiver_end

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appply CutMix to the batch of 1D signal.

        :param x: Input features. (N,C,L)
        :param y: Input labels. (N,C)
        :return: The augmented features and labels
        """
        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        shuffled_indices = torch.randperm(indices.shape[0])

        # Generate random floats between self.low and self.high for each sample in x
        cutoff_rates = torch.rand(x.shape[0], device=x.device) * (self.high - self.low) + self.low
        # Cutoff rates is how much of a donor sample will end up in the receiver sample        

        augmented_x = x.clone()
        augmented_y = y.clone().float()


        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                donor = x[shuffled_indices[i]]
                receiver = x[i]
                donor_start, donor_end, receiver_start, receiver_end = self.find_window(donor, receiver, (cutoff_rates[i] * x.shape[-1]).int().item(), stride=300)
                augmented_x[i,:,receiver_start:receiver_end] = donor[:,donor_start:donor_end]
                augmented_y[i] = torch.clip(y[i] + y[shuffled_indices[i]], 0, 1)
        return augmented_x, augmented_y

"""import matplotlib.pyplot as plt
import sounddevice
import numpy as np
i=9
receiver = x[i,0]
donor = x[shuffled_indices[i]][0]

augmented = augmented_x[i,0]
plt.figure()
plt.plot(receiver.numpy())
plt.title('Receiver')

plt.figure()
plt.plot(donor.numpy())
plt.title('Donor')

plt.figure()
plt.plot(augmented.numpy())
plt.title('Augmented')
plt.show()"""