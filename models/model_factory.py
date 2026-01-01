"""
Model Factory
Factory function to create models based on configuration.
"""

import torch
from typing import Dict, Any

from .efficientnet import create_efficientnet_model
from .vit_model import create_vit_model


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Model instance
    """
    architecture = config["model"]["architecture"].lower()

    if "efficientnet" in architecture:
        return create_efficientnet_model(config)
    elif "vit" in architecture:
        return create_vit_model(config)
    elif "resnet" in architecture:
        # TODO: Implement ResNet model
        raise NotImplementedError("ResNet model not yet implemented")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: str = "cuda"
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
):
    """
    Save model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, save_path)
