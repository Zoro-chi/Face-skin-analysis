"""
Environment Detection and Path Management
Handles local vs Colab environment differences.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple


def is_colab() -> bool:
    """
    Check if running in Google Colab.

    Returns:
        True if in Colab, False otherwise
    """
    try:
        import google.colab

        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """
    Check if running in Kaggle.

    Returns:
        True if in Kaggle, False otherwise
    """
    return os.path.exists("/kaggle/working")


def get_environment() -> str:
    """
    Detect current environment.

    Returns:
        'colab', 'kaggle', or 'local'
    """
    if is_colab():
        return "colab"
    elif is_kaggle():
        return "kaggle"
    else:
        return "local"


def setup_paths(base_dir: str = None) -> Dict[str, Path]:
    """
    Setup paths based on environment.

    Args:
        base_dir: Base directory for the project (None for auto-detect)

    Returns:
        Dictionary of paths
    """
    env = get_environment()

    if env == "colab":
        if base_dir is None:
            # Default Colab setup with Google Drive
            base_dir = "/content/drive/MyDrive/Face-skin-analysis"
        base_path = Path(base_dir)

        # Ensure Google Drive is mounted
        if not Path("/content/drive").exists():
            print("⚠️  Google Drive not mounted. Mounting now...")
            from google.colab import drive

            drive.mount("/content/drive")

    elif env == "kaggle":
        if base_dir is None:
            base_dir = "/kaggle/working/Face-skin-analysis"
        base_path = Path(base_dir)

    else:  # local
        if base_dir is None:
            # Get project root (assumes this file is in utils/)
            base_path = Path(__file__).parent.parent
        else:
            base_path = Path(base_dir)

    # Define all paths
    paths = {
        "base": base_path,
        "data": base_path / "data",
        "raw": base_path / "data" / "raw",
        "processed": base_path / "data" / "processed",
        "augmented": base_path / "data" / "augmented",
        "configs": base_path / "configs",
        "models": base_path / "models",
        "outputs": base_path / "outputs",
        "checkpoints": base_path / "outputs" / "checkpoints",
        "logs": base_path / "outputs" / "logs",
        "mlruns": base_path / "mlruns",
        "onnx": base_path / "onnx",
    }

    return paths


def get_device_info() -> Dict[str, any]:
    """
    Get device information (CPU/GPU).

    Returns:
        Dictionary with device info
    """
    import torch

    info = {
        "environment": get_environment(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    return info


def print_environment_info():
    """Print detailed environment information."""
    env = get_environment()
    device_info = get_device_info()

    print("=" * 60)
    print("🌍 ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Environment: {env.upper()}")
    print(f"Device: {device_info['device'].upper()}")

    if device_info["cuda_available"]:
        print(f"GPU: {device_info['gpu_name']}")
        print(f"GPU Memory: {device_info['gpu_memory']}")
        print(f"GPU Count: {device_info['gpu_count']}")
    else:
        print("GPU: Not available (using CPU)")

    print("=" * 60)


def sync_from_drive(local_path: str, drive_path: str = None):
    """
    Sync files from Google Drive to Colab local storage.
    Useful for faster data access during training.

    Args:
        local_path: Local path in Colab (e.g., /content/data)
        drive_path: Path in Google Drive
    """
    if not is_colab():
        print("⚠️  This function only works in Google Colab")
        return

    import shutil

    if drive_path is None:
        drive_path = local_path.replace("/content/", "/content/drive/MyDrive/")

    print(f"📂 Syncing from Drive: {drive_path} -> {local_path}")

    if os.path.exists(drive_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copytree(drive_path, local_path, dirs_exist_ok=True)
        print("✅ Sync complete!")
    else:
        print(f"❌ Source path not found: {drive_path}")


def sync_to_drive(local_path: str, drive_path: str = None):
    """
    Sync files from Colab local storage to Google Drive.
    Useful for saving checkpoints and results.

    Args:
        local_path: Local path in Colab (e.g., /content/outputs)
        drive_path: Path in Google Drive
    """
    if not is_colab():
        print("⚠️  This function only works in Google Colab")
        return

    import shutil

    if drive_path is None:
        drive_path = local_path.replace("/content/", "/content/drive/MyDrive/")

    print(f"📂 Syncing to Drive: {local_path} -> {drive_path}")

    if os.path.exists(local_path):
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        shutil.copytree(local_path, drive_path, dirs_exist_ok=True)
        print("✅ Sync complete!")
    else:
        print(f"❌ Source path not found: {local_path}")


def install_colab_dependencies():
    """
    Install project dependencies in Colab.
    Call this at the start of your Colab notebook.
    """
    if not is_colab():
        print("⚠️  Not in Colab environment. Skipping installation.")
        return

    print("📦 Installing dependencies for Colab...")

    # Install from requirements.txt if available
    if os.path.exists("requirements.txt"):
        os.system("pip install -q -r requirements.txt")
    else:
        # Core dependencies
        packages = [
            "torch torchvision",
            "opencv-python albumentations",
            "timm transformers",
            "mlflow dagshub",
            "scikit-learn pandas",
            "matplotlib seaborn plotly",
            "onnx onnxruntime",
        ]

        for package in packages:
            os.system(f"pip install -q {package}")

    print("✅ Dependencies installed!")
