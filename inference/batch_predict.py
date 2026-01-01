"""
Batch Prediction Script
Performs inference on multiple images.
"""

import torch
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_factory import create_model, load_checkpoint
from inference.predict import preprocess_image, predict_standard
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch prediction on multiple images")
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_predictions",
        help="Directory to save outputs",
    )
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
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    return parser.parse_args()


def main():
    """Main batch prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    logger.info(f"Processing images from: {args.input_dir}")

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    input_dir = Path(args.input_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        logger.error("No images found!")
        return

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_model(config)
    model = load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device)
    model.eval()

    # Process images
    condition_names = config["model"]["conditions"]
    results = []

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Preprocess
            image_tensor, _ = preprocess_image(str(img_path), config)

            # Predict
            predictions = predict_standard(model, image_tensor, args.device)

            # Store results
            result = {"image_path": str(img_path), "filename": img_path.name}

            for name, prob in zip(condition_names, predictions):
                result[name] = prob

            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_csv = output_dir / "batch_predictions.csv"
    df.to_csv(output_csv, index=False)

    logger.info(f"\nProcessed {len(df)} images")
    logger.info(f"Results saved to {output_csv}")

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    for condition in condition_names:
        mean_prob = df[condition].mean()
        positive_count = (df[condition] >= 0.5).sum()
        logger.info(f"{condition}:")
        logger.info(f"  Mean probability: {mean_prob:.4f}")
        logger.info(f"  Predicted positive: {positive_count}/{len(df)}")


if __name__ == "__main__":
    main()
