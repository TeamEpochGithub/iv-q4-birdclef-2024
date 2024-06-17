"""Computes loss for two outputs from a model."""

import torch


class DoubleLoss(torch.nn.Module):
    """Applies a task loss to task preds and call loss to the call preds separately and adds the losses up"""

    def __init__(self, task_loss: torch.nn.Module, call_loss: torch.nn.Module) -> None:
        """Initialize the DoubleLoss class."""
        super().__init__()
        self.task_loss = task_loss
        self.call_loss = call_loss

    def forward(self, task_inputs: torch.Tensor, task_targets: torch.Tensor, call_inputs: torch.Tensor, call_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_loss = self.task_loss(task_inputs, task_targets)
        call_loss = self.call_loss(call_inputs, call_targets)
        return task_loss, call_loss
