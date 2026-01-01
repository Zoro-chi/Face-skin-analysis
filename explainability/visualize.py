"""
Visualization Utilities
Helper functions for creating visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def plot_predictions(
    image: np.ndarray,
    predictions: np.ndarray,
    condition_names: List[str],
    uncertainties: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Plot image with predictions.

    Args:
        image: Input image
        predictions: Prediction probabilities
        condition_names: Names of conditions
        uncertainties: Uncertainty estimates
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Display predictions
    y_pos = np.arange(len(condition_names))

    if uncertainties is not None:
        ax2.barh(y_pos, predictions, xerr=uncertainties, capsize=5, alpha=0.7)
    else:
        ax2.barh(y_pos, predictions, alpha=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(condition_names)
    ax2.set_xlabel("Probability")
    ax2.set_xlim([0, 1])
    ax2.set_title("Predictions")
    ax2.grid(axis="x", alpha=0.3)

    # Add threshold line
    ax2.axvline(x=0.5, color="r", linestyle="--", alpha=0.5, label="Threshold")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved prediction plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_gradcam(
    image: np.ndarray,
    cams: List[np.ndarray],
    condition_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Plot Grad-CAM for multiple conditions.

    Args:
        image: Input image
        cams: List of CAMs for each condition
        condition_names: Names of conditions
        save_path: Path to save plot
    """
    n_conditions = len(condition_names)
    fig, axes = plt.subplots(1, n_conditions + 1, figsize=(5 * (n_conditions + 1), 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM for each condition
    for i, (cam, name) in enumerate(zip(cams, condition_names)):
        import cv2

        # Resize CAM
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Convert to heatmap
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        axes[i + 1].imshow(overlayed)
        axes[i + 1].set_title(f"{name}")
        axes[i + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved multi-condition Grad-CAM to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curves(
    tpr_dict: dict,
    fpr_dict: dict,
    auc_dict: dict,
    condition_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Plot ROC curves for all conditions.

    Args:
        tpr_dict: Dictionary of TPR values
        fpr_dict: Dictionary of FPR values
        auc_dict: Dictionary of AUC values
        condition_names: Names of conditions
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    for condition in condition_names:
        plt.plot(
            fpr_dict[condition],
            tpr_dict[condition],
            label=f"{condition} (AUC = {auc_dict[condition]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curves to {save_path}")
    else:
        plt.show()

    plt.close()
