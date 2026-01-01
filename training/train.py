"""
Main Training Script
Orchestrates the training process with MLflow tracking.
Environment-aware: Works in both local and Colab environments.
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import mlflow
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_factory import create_model
from training.data_loader import create_data_loaders
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.config_loader import load_config
from utils.environment import get_environment, print_environment_info, is_colab
from utils.helpers import set_seed, get_device

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train skin condition detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def setup_mlflow(config):
    """Setup MLflow tracking."""
    mlflow_config = config.get("mlops", {}).get("mlflow", {})

    # Set tracking URI (use local mlruns if not specified)
    tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment
    experiment_name = mlflow_config.get("experiment_name", "skin-condition-detection")
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow experiment: {experiment_name}")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Failed to setup MLflow: {e}")
        logger.warning("⚠️  Continuing without MLflow tracking...")
        return False


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    # Print environment info
    print_environment_info()

    logger.info("Starting training pipeline...")

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config["data"].get("random_seed", 42))

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Setup MLflow (skip if in Colab without credentials)
    env = get_environment()
    mlflow_enabled = False

    if env == "local":
        mlflow_enabled = setup_mlflow(config)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_enabled = setup_mlflow(config)
    else:
        logger.warning(
            "⚠️  MLflow tracking disabled (no tracking URI configured). Training will continue without tracking."
        )

    # Start MLflow run (only if enabled)
    if mlflow_enabled:
        mlflow.start_run()

    try:
        # Log configuration
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "architecture": config["model"]["architecture"],
                    "batch_size": config["training"]["batch_size"],
                    "learning_rate": config["training"]["learning_rate"],
                    "num_epochs": config["training"]["num_epochs"],
                    "optimizer": config["training"]["optimizer"],
                    "loss": config["training"]["loss"],
                }
            )

        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(config)

        # Create model
        logger.info(f"Creating model: {config['model']['architecture']}")
        model = create_model(config)

        # Log model info
        model_info = model.get_model_info()
        logger.info(f"Model parameters: {model_info['num_parameters']:,}")
        if mlflow_enabled:
            mlflow.log_param("num_parameters", model_info["num_parameters"])

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint["model_state_dict"])

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(model, config, device=args.device)

        # Train model
        trainer.train(
            train_loader=data_loaders["train"], val_loader=data_loaders["val"]
        )

        # Log best model
        best_model_path = Path(config["training"]["checkpoint_dir"]) / "best_model.pth"
        if best_model_path.exists() and mlflow_enabled:
            mlflow.log_artifact(str(best_model_path))
            logger.info(f"Logged best model to MLflow: {best_model_path}")

        logger.info("Training pipeline complete!")

    finally:
        # End MLflow run if it was started
        if mlflow_enabled and mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()
