"""
EfficientNet-B3 Model
EfficientNet-based model for skin condition detection.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any

from .base_model import BaseSkinModel


class EfficientNetModel(BaseSkinModel):
    """EfficientNet-B3 model for multi-label skin condition classification."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EfficientNet model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=config["model"].get("pretrained", True),
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, self.num_classes),
        )

        # Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)

        # Classification
        logits = self.classifier(features)

        # Apply sigmoid for probabilities
        probs = self.sigmoid(logits)

        return probs

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representations.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.flatten(1)
        return features

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps before global pooling (for Grad-CAM).

        Args:
            x: Input tensor

        Returns:
            Feature maps
        """
        return self.backbone(x)


def create_efficientnet_model(config: Dict[str, Any]) -> EfficientNetModel:
    """
    Create EfficientNet model.

    Args:
        config: Configuration dictionary

    Returns:
        EfficientNet model
    """
    return EfficientNetModel(config)
