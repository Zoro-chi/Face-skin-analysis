"""
Grad-CAM Implementation
Gradient-weighted Class Activation Mapping for model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for visual explanations of CNN predictions."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: Model to explain
            target_layer: Target layer for activation extraction
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Class Activation Map.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index

        Returns:
            CAM as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def generate_heatmap(
        self,
        cam: np.ndarray,
        input_size: Tuple[int, int] = (224, 224),
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Generate heatmap from CAM.

        Args:
            cam: Class Activation Map
            input_size: Size to resize heatmap to
            colormap: OpenCV colormap

        Returns:
            Heatmap as numpy array
        """
        # Resize CAM to input size
        cam_resized = cv2.resize(cam, input_size)

        # Convert to uint8
        heatmap = np.uint8(255 * cam_resized)

        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, colormap)

        return heatmap

    def overlay_heatmap(
        self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on image.

        Args:
            image: Original image (H, W, C) in RGB
            heatmap: Heatmap (H, W, C) in BGR
            alpha: Transparency of heatmap

        Returns:
            Overlayed image
        """
        # Convert heatmap from BGR to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)

        return overlayed


def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """
    Get target layer for Grad-CAM based on architecture.

    Args:
        model: Model
        architecture: Model architecture name

    Returns:
        Target layer
    """
    architecture = architecture.lower()

    if "efficientnet" in architecture:
        # Last convolutional layer of EfficientNet
        return model.backbone.blocks[-1]
    elif "vit" in architecture:
        # For ViT, use the last transformer block
        return model.backbone.blocks[-1].norm1
    elif "resnet" in architecture:
        # Last layer of ResNet
        return model.backbone.layer4[-1]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def visualize_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    condition_name: str,
    save_path: Optional[str] = None,
):
    """
    Visualize Grad-CAM results.

    Args:
        image: Original image
        cam: Class Activation Map
        condition_name: Name of condition
        save_path: Path to save visualization
    """
    gradcam = GradCAM(None, None)

    # Generate heatmap
    heatmap = gradcam.generate_heatmap(cam, input_size=(image.shape[1], image.shape[0]))

    # Overlay
    overlayed = gradcam.overlay_heatmap(image, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title(f"Grad-CAM: {condition_name}")
    axes[1].axis("off")

    axes[2].imshow(overlayed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Grad-CAM visualization to {save_path}")
    else:
        plt.show()

    plt.close()
