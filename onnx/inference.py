"""
ONNX Inference Script
Performs inference using ONNX Runtime for optimized performance.
"""

import onnxruntime as ort
import numpy as np
import cv2
import argparse
import logging
from pathlib import Path

import sys

sys.path.append("..")

from preprocessing.face_detection import FaceDetector
from preprocessing.augmentation import apply_clahe
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ONNX model inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model", type=str, default="onnx/skin_model.onnx", help="Path to ONNX model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def preprocess_image_for_onnx(image_path, config):
    """
    Preprocess image for ONNX inference.

    Args:
        image_path: Path to input image
        config: Configuration dictionary

    Returns:
        Preprocessed image array
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

    if bbox is not None:
        face = detector.crop_face(image_rgb, bbox)
    else:
        logger.warning("No face detected, using full image")
        face = image_rgb

    # Resize
    image_size = config["preprocessing"]["image_size"]
    face_resized = cv2.resize(face, (image_size, image_size))

    # Apply CLAHE
    face_enhanced = apply_clahe(face_resized, config)

    # Normalize
    mean = np.array(config["preprocessing"]["mean"])
    std = np.array(config["preprocessing"]["std"])

    face_normalized = (face_enhanced.astype(np.float32) / 255.0 - mean) / std

    # Transpose to CHW format
    face_chw = face_normalized.transpose(2, 0, 1)

    # Add batch dimension
    face_batch = np.expand_dims(face_chw, axis=0).astype(np.float32)

    return face_batch


def run_onnx_inference(model_path, input_array):
    """
    Run inference with ONNX model.

    Args:
        model_path: Path to ONNX model
        input_array: Input array

    Returns:
        Model predictions
    """
    # Create inference session
    session = ort.InferenceSession(model_path)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_array})

    return outputs[0]


def main():
    """Main ONNX inference function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    logger.info(f"Processing image: {args.image}")
    logger.info(f"Using ONNX model: {args.model}")

    # Load configuration
    config = load_config(args.config)

    # Preprocess image
    logger.info("Preprocessing image...")
    input_array = preprocess_image_for_onnx(args.image, config)

    # Run inference
    logger.info("Running ONNX inference...")
    predictions = run_onnx_inference(args.model, input_array)

    # Get condition names
    condition_names = config["model"]["conditions"]

    # Log results
    logger.info("\n=== Predictions ===")
    for name, prob in zip(condition_names, predictions[0]):
        logger.info(f"{name}: {prob:.4f}")

    logger.info("\nInference complete!")


if __name__ == "__main__":
    main()
