"""
Helper Utilities
Miscellaneous helper functions for the project.
"""

import torch
import numpy as np
import random
import os
from typing import Optional
import logging

try:
    from .environment import get_environment, is_colab
except ImportError:
    from environment import get_environment, is_colab

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get device for PyTorch with environment awareness.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto)

    Returns:
        PyTorch device
    """
    env = get_environment()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if torch.cuda.is_available() and device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} in {env} environment")
    else:
        logger.info(f"Using CPU in {env} environment")

    return device

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_predictions(
    predictions: np.ndarray, targets: np.ndarray, image_paths: list, output_path: str
):
    """
    Save predictions to CSV file.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        image_paths: List of image paths
        output_path: Path to save CSV
    """
    import pandas as pd

    data = {"image_path": image_paths}

    # Add predictions
    for i in range(predictions.shape[1]):
        data[f"pred_{i}"] = predictions[:, i]
        data[f"target_{i}"] = targets[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved predictions to {output_path}")


def create_directory_structure(base_dir: str):
    """
    Create project directory structure.

    Args:
        base_dir: Base directory path
    """
    from pathlib import Path

    directories = [
        "data/raw",
        "data/processed",
        "data/augmented",
        "outputs/checkpoints",
        "outputs/explainability",
        "outputs/logs",
        "outputs/plots",
        "mlruns",
    ]

    base_path = Path(base_dir)

    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created directory structure in {base_dir}")
