"""
Custom Loss Functions
Implements weighted BCE loss and other custom losses for skin condition detection.
"""

import torch
import torch.nn as nn
from typing import Optional


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss for multi-label classification."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted BCE loss.

        Args:
            class_weights: Weights for each class (num_classes,)
        """
        super().__init__()
        self.class_weights = class_weights
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size, num_classes)

        Returns:
            Loss value
        """
        loss = self.bce(predictions, targets)

        if self.class_weights is not None:
            # Apply class weights
            weights = self.class_weights.to(predictions.device)
            loss = loss * weights.unsqueeze(0)

        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size, num_classes)

        Returns:
            Loss value
        """
        bce_loss = nn.functional.binary_cross_entropy(
            predictions, targets, reduction="none"
        )

        # Compute focal weight
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - pt) ** self.gamma

        # Compute alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Combine weights
        loss = alpha_weight * focal_weight * bce_loss

        return loss.mean()


def create_loss_function(loss_type: str, class_weights: Optional[list] = None):
    """
    Create loss function based on type.

    Args:
        loss_type: Type of loss ('bce', 'weighted_bce', 'focal')
        class_weights: Optional class weights

    Returns:
        Loss function
    """
    if loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "weighted_bce":
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weights = None
        return WeightedBCELoss(class_weights=weights)
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
