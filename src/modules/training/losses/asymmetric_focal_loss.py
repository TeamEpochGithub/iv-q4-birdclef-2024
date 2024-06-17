import torch


class AsymmetricalFocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 0, zeta: float = 0) -> None:
        super().__init__()
        self.gamma = gamma  # balancing between classes
        self.zeta = zeta  # balancing between active/inactive frames

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = -(
            ((1 - pred) ** self.gamma) * target * torch.clamp_min(torch.log(pred), -100) + (pred**self.zeta) * (1 - target) * torch.clamp_min(torch.log(1 - pred), -100)
        )
        return torch.mean(losses)
