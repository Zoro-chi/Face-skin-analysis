"""
Explainability Module
Grad-CAM and visualization tools for model interpretability.
"""

from .gradcam import GradCAM, get_target_layer, visualize_gradcam
from .visualize import (
    plot_predictions,
    plot_multi_gradcam,
    plot_confusion_matrix,
    plot_roc_curves,
)

__all__ = [
    "GradCAM",
    "get_target_layer",
    "visualize_gradcam",
    "plot_predictions",
    "plot_multi_gradcam",
    "plot_confusion_matrix",
    "plot_roc_curves",
]
