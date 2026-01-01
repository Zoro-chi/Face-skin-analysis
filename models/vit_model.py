"""
Vision Transformer (ViT) Model
ViT-based model for skin condition detection.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any

from .base_model import BaseSkinModel


class ViTModel(BaseSkinModel):
    """Vision Transformer model for multi-label skin condition classification."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ViT model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Load pretrained ViT
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=config["model"].get("pretrained", True),
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
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
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from transformer blocks.

        Args:
            x: Input tensor

        Returns:
            Attention maps
        """
        # Get attention weights from the last transformer block
        attn_weights = []

        def hook_fn(module, input, output):
            attn_weights.append(output)

        # Register hook on last attention layer
        hook = self.backbone.blocks[-1].attn.register_forward_hook(hook_fn)

        # Forward pass
        _ = self.backbone(x)

        # Remove hook
        hook.remove()

        return attn_weights[0] if attn_weights else None


def create_vit_model(config: Dict[str, Any]) -> ViTModel:
    """
    Create ViT model.

    Args:
        config: Configuration dictionary

    Returns:
        ViT model
    """
    return ViTModel(config)
