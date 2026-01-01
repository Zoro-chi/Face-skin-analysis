"""
Base Model Class
Abstract base class for all skin condition detection models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseSkinModel(nn.Module, ABC):
    """Abstract base class for skin condition detection models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.num_classes = config["model"]["num_classes"]
        self.dropout_rate = config["model"].get("dropout_rate", 0.3)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representations.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        pass

    def enable_dropout(self):
        """Enable dropout for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.

        Args:
            x: Input tensor
            n_samples: Number of MC Dropout samples

        Returns:
            Dictionary with predictions and uncertainty
        """
        self.eval()
        self.enable_dropout()

        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        # Calculate mean and std
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return {
            "prediction": mean_pred,
            "uncertainty": std_pred,
            "all_predictions": predictions,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "architecture": self.__class__.__name__,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters(),
            "dropout_rate": self.dropout_rate,
        }
