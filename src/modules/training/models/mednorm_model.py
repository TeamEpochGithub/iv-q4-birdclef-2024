"""Module that subtracts the median over the time axis, and if specified also the frequency axis."""

import torch


class MedNormModel(torch.nn.Module):
    """Module that subtracts the median over the time axis, and if specified also the frequency axis."""

    def __init__(self, model: torch.nn.Module, use_vertical: bool = True, quantile: float = 0.25) -> None:
        """Initialize the MedNormModel.

        :param model: The model to be wrapped
        :param use_vertical: Whether to subtract the median over the frequency axis
        :param quantile: The quantile to be used for the median
        """
        super().__init__()
        self.model = model
        self.use_vertical = use_vertical
        self.quantile = quantile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract the median over the time axis, and if specified also the frequency axis.

        :param x: The input tensor of shape (B, C, H, W)
        :return: The output tensor of shape (B, C, H, W)
        """
        norm = torch.quantile(x, self.quantile, dim=2, keepdim=True)[0]
        x = x - norm
        if self.use_vertical:
            norm = torch.quantile(x, self.quantile, dim=3, keepdim=True)[0]
            x = x - norm
        return self.model(x)

    def __repr__(self) -> str:
        """Return the string representation of the model.

        :return: The string representation of the model
        """
        return f"MedNormModel(model={self.model}, use_vertical={self.use_vertical}, quantile={self.quantile})"
