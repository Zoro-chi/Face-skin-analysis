"""
Single Image Prediction Script
Performs inference on a single image with confidence scoring and Grad-CAM.
"""

import torch
import cv2
import numpy as np
import argparse
import yaml
import logging
from pathlib import Path

import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_factory import create_model, load_checkpoint
from preprocessing.face_detection import FaceDetector
from preprocessing.augmentation import apply_clahe, get_augmentation_pipeline
from explainability.gradcam import GradCAM, get_target_layer, visualize_gradcam
from explainability.visualize import plot_predictions
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict skin conditions from image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--show-gradcam",
        action="store_true",
        help="Generate and save Grad-CAM visualizations",
    )
    parser.add_argument(
        "--use-mc-dropout",
        action="store_true",
        help="Use MC Dropout for uncertainty estimation",
    )
    return parser.parse_args()


def preprocess_image(image_path: str, config: dict) -> tuple:
    """
    Preprocess image for inference.

    Args:
        image_path: Path to input image
        config: Configuration dictionary

    Returns:
        Tuple of (processed_tensor, original_image)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect and crop face
    detector = FaceDetector(
        method=config["preprocessing"].get("face_detector", "opencv")
    )
    bbox = detector.detect_face(image)

    if bbox is None:
        logger.warning("No face detected, using full image")
        face = image_rgb
    else:
        face = detector.crop_face(image_rgb, bbox)

    # Resize
    image_size = config["preprocessing"]["image_size"]
    face_resized = cv2.resize(face, (image_size, image_size))

    # Apply CLAHE
    face_enhanced = apply_clahe(face_resized, config)

    # Get augmentation pipeline (validation mode)
    transform = get_augmentation_pipeline(config, is_training=False)

    # Transform
    face_tensor = transform(face_enhanced)

    # Add batch dimension
    face_tensor = face_tensor.unsqueeze(0)

    return face_tensor, face_resized


def predict_with_uncertainty(model, image_tensor, device, n_samples=20):
    """
    Predict with uncertainty estimation using MC Dropout.

    Args:
        model: Model
        image_tensor: Input tensor
        device: Device
        n_samples: Number of MC samples

    Returns:
        Dictionary with predictions and uncertainty
    """
    image_tensor = image_tensor.to(device)
    result = model.predict_with_uncertainty(image_tensor, n_samples=n_samples)

    return {
        "prediction": result["prediction"].cpu().numpy()[0],
        "uncertainty": result["uncertainty"].cpu().numpy()[0],
    }


def predict_standard(model, image_tensor, device):
    """
    Standard prediction without uncertainty.

    Args:
        model: Model
        image_tensor: Input tensor
        device: Device

    Returns:
        Prediction probabilities
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions.cpu().numpy()[0]


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    logger.info(f"Processing image: {args.image}")

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess image
    logger.info("Preprocessing image...")
    image_tensor, original_image = preprocess_image(args.image, config)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_model(config)
    model = load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device)

    # Get predictions
    condition_names = config["model"]["conditions"]

    if args.use_mc_dropout:
        logger.info("Predicting with MC Dropout...")
        n_samples = config["model"].get("mc_dropout_samples", 20)
        result = predict_with_uncertainty(model, image_tensor, args.device, n_samples)
        predictions = result["prediction"]
        uncertainties = result["uncertainty"]
    else:
        logger.info("Predicting...")
        predictions = predict_standard(model, image_tensor, args.device)
        uncertainties = None

    # Log results
    logger.info("\n=== Predictions ===")
    for name, prob in zip(condition_names, predictions):
        if uncertainties is not None:
            unc = uncertainties[list(condition_names).index(name)]
            logger.info(f"{name}: {prob:.4f} ± {unc:.4f}")
        else:
            logger.info(f"{name}: {prob:.4f}")

    # Save prediction plot
    plot_save_path = output_dir / f"{Path(args.image).stem}_predictions.png"
    plot_predictions(
        original_image,
        predictions,
        condition_names,
        uncertainties=uncertainties,
        save_path=str(plot_save_path),
    )

    # Generate Grad-CAM if requested
    if args.show_gradcam:
        logger.info("Generating Grad-CAM visualizations...")

        # Get target layer
        architecture = config["model"]["architecture"]
        target_layer = get_target_layer(model, architecture)

        # Create Grad-CAM
        gradcam = GradCAM(model, target_layer)

        # Generate CAM for each condition
        for i, condition in enumerate(condition_names):
            cam = gradcam.generate_cam(image_tensor.to(args.device), i)

            # Save visualization
            cam_save_path = (
                output_dir / f"{Path(args.image).stem}_{condition}_gradcam.png"
            )
            visualize_gradcam(
                original_image, cam, condition, save_path=str(cam_save_path)
            )

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
