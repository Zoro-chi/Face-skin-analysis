"""
Main Evaluation Script
Evaluates trained model and performs bias analysis.
"""

import os
import sys
import torch
import numpy as np
import yaml
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_factory import create_model, load_checkpoint
from training.data_loader import create_data_loaders
from evaluation.metrics import MetricsCalculator
from evaluation.bias_analysis import BiasAnalyzer
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate skin condition detection model"
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
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--optimize_thresholds",
        action="store_true",
        help="Optimize per-class thresholds using validation set",
    )
    return parser.parse_args()


def collect_predictions(model, data_loader, device):
    """
    Collect predictions from model.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use

    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Collecting predictions"):
            images = images.to(device)

            # Get predictions
            outputs = model(images)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    return predictions, targets


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    logger.info("Starting evaluation pipeline...")
    logger.info(f"Using device: {args.device}")

    # Load configuration
    config = load_config(args.config)
    # Merge CLI and config flag for threshold optimization
    args.optimize_thresholds = args.optimize_thresholds or config.get(
        "evaluation", {}
    ).get("optimize_thresholds", False)

    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(config)
    data_loader = data_loaders[args.split]

    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
    model = create_model(config)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device)

    # Optionally optimize thresholds using validation split
    condition_names = config["model"]["conditions"]
    threshold = config["evaluation"].get("confidence_threshold", 0.5)

    optimized_thresholds = None
    if args.optimize_thresholds and args.split == "test":
        logger.info("Optimizing per-class thresholds on validation set...")
        val_loader = data_loaders["val"]
        val_preds, val_targets = collect_predictions(model, val_loader, args.device)

        # Grid search thresholds for each class to maximize F1
        grid = np.linspace(0.05, 0.95, 19)
        thresholds = []
        from sklearn.metrics import f1_score

        for i, cond in enumerate(condition_names):
            best_t = threshold
            best_f1 = -1.0
            y_true = val_targets[:, i]
            y_scores = val_preds[:, i]

            # Skip if no positives or no variation
            if y_true.sum() == 0 or np.allclose(y_scores, y_scores[0]):
                thresholds.append(best_t)
                continue

            for t in grid:
                y_pred = (y_scores >= t).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            thresholds.append(float(best_t))

        optimized_thresholds = np.array(thresholds, dtype=float)
        logger.info(
            f"Optimized thresholds: "
            + ", ".join(
                [f"{c}={t:.2f}" for c, t in zip(condition_names, optimized_thresholds)]
            )
        )

        # Save thresholds
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "thresholds.json", "w") as f:
            json.dump(
                {c: float(t) for c, t in zip(condition_names, optimized_thresholds)},
                f,
                indent=2,
            )

    # Collect predictions for selected split
    logger.info(f"Evaluating on {args.split} set...")
    predictions, targets = collect_predictions(model, data_loader, args.device)

    # Calculate metrics
    logger.info("Calculating metrics...")
    # Use optimized thresholds if available
    metrics_threshold = (
        optimized_thresholds if optimized_thresholds is not None else threshold
    )

    metrics_calc = MetricsCalculator(condition_names, threshold=metrics_threshold)
    metrics = metrics_calc.calculate_metrics(predictions, targets)

    # Log metrics
    logger.info("\n=== Evaluation Metrics ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Get classification report
    report = metrics_calc.get_classification_report(predictions, targets)
    logger.info("\n=== Classification Report ===")
    logger.info(f"\n{report}")

    # Save metrics
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_dir / 'metrics.json'}")

    # Bias analysis (if skin tone information available)
    if config["evaluation"].get("stratify_by_skin_tone", False):
        logger.info("\n=== Bias Analysis ===")
        # Load skin tones from split CSV to match loader order
        processed_dir = Path(config["data"]["processed_dir"])
        split_csv = processed_dir / f"{args.split}.csv"
        if split_csv.exists():
            df_split = pd.read_csv(split_csv)
            raw_tones = df_split["skin_tone"].astype(str).fillna("unknown").values

            # Map numeric Fitzpatrick to groups using config
            tone_map = config["data"].get("skin_tones", {})
            light = set(map(str, tone_map.get("light", [])))
            medium = set(map(str, tone_map.get("medium", [])))
            dark = set(map(str, tone_map.get("dark", [])))

            logger.info(
                f"Tone mapping - Light: {light}, Medium: {medium}, Dark: {dark}"
            )
            logger.info(f"Raw tone sample (first 10): {raw_tones[:10]}")

            def to_group(x: str) -> str:
                x_clean = x.strip()
                if x_clean in light:
                    return "light"
                if x_clean in medium:
                    return "medium"
                if x_clean in dark:
                    return "dark"
                return "unknown"

            skin_tones = np.array([to_group(x) for x in raw_tones])

            logger.info(f"Mapped groups (first 10): {skin_tones[:10]}")
            logger.info(
                f"Group distribution: {np.unique(skin_tones, return_counts=True)}"
            )

            # Filter out unknowns for fairness metrics
            known_mask = skin_tones != "unknown"
            if known_mask.sum() == 0:
                logger.info("No known skin tone labels found; skipping bias analysis.")
            else:
                skin_tone_groups = ["light", "medium", "dark"]
                bias_analyzer = BiasAnalyzer(
                    condition_names,
                    skin_tone_groups,
                    output_dir="outputs/bias_analysis",
                )
                # Align arrays to known labels
                bias_analyzer.analyze(
                    predictions[known_mask],
                    targets[known_mask],
                    skin_tones[known_mask],
                    metrics_threshold,
                )
        else:
            logger.info("Split CSV not found; skipping bias analysis.")
    else:
        logger.info("Bias analysis disabled in config (stratify_by_skin_tone=false)")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
