"""
Data Loader Module
Creates PyTorch DataLoaders for training, validation, and testing.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

import sys

sys.path.append("..")
from preprocessing.augmentation import get_augmentation_pipeline

logger = logging.getLogger(__name__)


class SkinConditionDataset(Dataset):
    """Dataset for skin condition images."""

    def __init__(self, data_csv: str, config: Dict[str, Any], is_training: bool = True):
        """
        Initialize dataset.

        Args:
            data_csv: Path to CSV file with image paths and labels
            config: Configuration dictionary
            is_training: Whether this is training data
        """
        self.data = pd.read_csv(data_csv)
        self.config = config
        self.is_training = is_training

        # Get augmentation pipeline
        self.transform = get_augmentation_pipeline(config, is_training=is_training)

        # Condition names
        self.conditions = config["model"]["conditions"]

        logger.info(f"Loaded {len(self.data)} images from {data_csv}")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)

    def get_skin_tone(self, idx: int) -> str:
        """Get skin tone label for sample."""
        if "skin_tone" in self.data.columns:
            return str(self.data.iloc[idx]["skin_tone"])
        return "unknown"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (image, labels)
        """
        # Get image path
        img_path = self.data.iloc[idx]["image_path"]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        image = self.transform(image)

        # Get labels (multi-label)
        labels = torch.zeros(len(self.conditions), dtype=torch.float32)

        # Parse labels from CSV
        # Assuming columns like 'has_acne', 'has_pigmentation', 'has_wrinkles'
        for i, condition in enumerate(self.conditions):
            col_name = f"has_{condition}"
            if col_name in self.data.columns:
                labels[i] = float(self.data.iloc[idx][col_name])

        return image, labels


def create_skin_tone_sampler(
    dataset: SkinConditionDataset, config: Dict[str, Any]
) -> Optional[WeightedRandomSampler]:
    """
    Create a weighted sampler to balance skin tone distribution.

    Args:
        dataset: Training dataset
        config: Configuration dictionary

    Returns:
        WeightedRandomSampler or None if balancing is disabled
    """
    if not config.get("training", {}).get("balance_skin_tones", False):
        return None

    # Map Fitzpatrick values to groups
    tone_map = config["data"].get("skin_tones", {})
    light = set(map(str, tone_map.get("light", [])))
    medium = set(map(str, tone_map.get("medium", [])))
    dark = set(map(str, tone_map.get("dark", [])))

    def to_group(x: str) -> str:
        x_clean = str(x).strip()
        if x_clean in light:
            return "light"
        if x_clean in medium:
            return "medium"
        if x_clean in dark:
            return "dark"
        return "unknown"

    # Get skin tone groups for each sample
    skin_tone_groups = np.array(
        [to_group(dataset.get_skin_tone(i)) for i in range(len(dataset))]
    )

    # Count samples per group
    unique, counts = np.unique(skin_tone_groups, return_counts=True)
    group_counts = dict(zip(unique, counts))

    # Remove unknown if present
    if "unknown" in group_counts:
        del group_counts["unknown"]

    if len(group_counts) == 0:
        logger.warning("No known skin tone labels found; disabling balanced sampling")
        return None

    logger.info(f"Skin tone distribution: {group_counts}")

    # Compute sample weights (inverse frequency)
    total_known = sum(group_counts.values())
    group_weights = {
        g: total_known / (len(group_counts) * c) for g, c in group_counts.items()
    }

    # Assign weight to each sample (unknown gets average weight)
    avg_weight = (
        sum(group_weights.values()) / len(group_weights) if group_weights else 1.0
    )
    sample_weights = np.array(
        [group_weights.get(g, avg_weight) for g in skin_tone_groups]
    )

    logger.info(f"Skin tone group weights: {group_weights}")

    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def create_data_loaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of data loaders
    """
    processed_dir = Path(config["data"]["processed_dir"])
    batch_size = config["training"]["batch_size"]

    # Create datasets
    train_dataset = SkinConditionDataset(
        data_csv=str(processed_dir / "train.csv"), config=config, is_training=True
    )

    val_dataset = SkinConditionDataset(
        data_csv=str(processed_dir / "val.csv"), config=config, is_training=False
    )

    test_dataset = SkinConditionDataset(
        data_csv=str(processed_dir / "test.csv"), config=config, is_training=False
    )

    # Create balanced sampler for training (fairness)
    train_sampler = create_skin_tone_sampler(train_dataset, config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
