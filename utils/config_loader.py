"""
Configuration Loader
Loads and validates configuration files with environment awareness.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

try:
    from .environment import get_environment, setup_paths
except ImportError:
    from environment import get_environment, setup_paths

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment-aware path resolution.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary with resolved paths
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    # Resolve paths based on environment
    env = get_environment()
    if config.get("environment", {}).get("auto_detect", True):
        logger.info(f"Auto-detecting environment: {env}")

        # Get environment-specific paths
        paths = setup_paths()

        # Update config with resolved paths
        if env == "colab":
            base_dir = (
                config.get("environment", {})
                .get("colab", {})
                .get("base_dir", str(paths["base"]))
            )
            config["data"]["raw_dir"] = str(Path(base_dir) / "data" / "raw")
            config["data"]["processed_dir"] = str(Path(base_dir) / "data" / "processed")
            config["data"]["augmented_dir"] = str(Path(base_dir) / "data" / "augmented")
            config["training"]["checkpoint_dir"] = str(
                Path(base_dir) / "outputs" / "checkpoints"
            )
            config["logging"]["log_dir"] = str(Path(base_dir) / "outputs" / "logs")

            logger.info(f"Using Colab base directory: {base_dir}")
        else:
            # Use paths as-is for local
            logger.info(f"Using local base directory: {paths['base']}")

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Saved configuration to {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        "project",
        "data",
        "preprocessing",
        "model",
        "training",
        "evaluation",
    ]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False

    logger.info("Configuration is valid")
    return True
