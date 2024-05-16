"""Wrapper class for a model that is meant to predict to things."""
from torch import nn
import torch


class DoubleModel(nn.Module):

    def __init__(self, model):
        """Initialize the model"""
        super.__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        """Call forward ofthe original model"""
        # Assume the last output of the model is the call nocall pred
        preds = self.model(x)

        # Separate the last pred in the head
        return preds[:,:-1], preds[:,-1]