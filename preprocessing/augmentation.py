"""
Data Augmentation Module
Implements tone-aware augmentation strategies for skin condition analysis.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SkinAwareAugmentation:
    """Augmentation pipeline with skin-tone awareness."""

    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """
        Initialize augmentation pipeline.

        Args:
            config: Configuration dictionary
            is_training: Whether to apply training augmentations
        """
        self.config = config
        self.is_training = is_training

        # Get augmentation parameters
        aug_config = config.get("preprocessing", {}).get("augmentation", {})

        if is_training:
            self.transform = A.Compose(
                [
                    # Geometric transformations
                    A.HorizontalFlip(p=aug_config.get("horizontal_flip", 0.5)),
                    A.Rotate(
                        limit=aug_config.get("rotation", 15),
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),
                    # Prefer Affine over ShiftScaleRotate per Albumentations guidance
                    A.Affine(
                        scale=(1 - 0.1, 1 + 0.1),
                        translate_percent=(0.0625, 0.0625),
                        rotate=0,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.3,
                    ),
                    # Color augmentations
                    A.OneOf(
                        [
                            A.ColorJitter(
                                brightness=aug_config.get("brightness", 0.2),
                                contrast=aug_config.get("contrast", 0.2),
                                saturation=aug_config.get("saturation", 0.2),
                                hue=aug_config.get("hue", 0.1),
                                p=1.0,
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0,
                            ),
                        ],
                        p=0.5,
                    ),
                    # Gentle brightness for skin tone preservation
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.3,
                    ),
                    # Noise and blur
                    # Noise and blur (avoid GaussNoise args incompatibility)
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                            A.ISONoise(
                                color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0
                            ),
                        ],
                        p=0.3,
                    ),
                    # Normalization
                    A.Normalize(
                        mean=config["preprocessing"]["mean"],
                        std=config["preprocessing"]["std"],
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            # Validation/Test augmentation (only normalization)
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=config["preprocessing"]["mean"],
                        std=config["preprocessing"]["std"],
                    ),
                    ToTensorV2(),
                ]
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to image.

        Args:
            image: Input image (RGB format)

        Returns:
            Augmented image tensor
        """
        augmented = self.transform(image=image)
        return augmented["image"]


class CLAHEProcessor:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) processor."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """
        Initialize CLAHE processor.

        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to image.

        Args:
            image: Input image (RGB format)

        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced


def get_augmentation_pipeline(config: Dict[str, Any], is_training: bool = True):
    """
    Get augmentation pipeline based on config.

    Args:
        config: Configuration dictionary
        is_training: Whether to get training or validation pipeline

    Returns:
        Augmentation pipeline
    """
    return SkinAwareAugmentation(config, is_training=is_training)


def apply_clahe(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Apply CLAHE if enabled in config.

    Args:
        image: Input image
        config: Configuration dictionary

    Returns:
        Processed image
    """
    clahe_config = config.get("preprocessing", {}).get("clahe", {})

    if clahe_config.get("enabled", False):
        processor = CLAHEProcessor(
            clip_limit=clahe_config.get("clip_limit", 2.0),
            tile_grid_size=tuple(clahe_config.get("tile_grid_size", [8, 8])),
        )
        return processor.apply(image)

    return image
