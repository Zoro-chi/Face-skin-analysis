"""
ONNX Export Script
Exports PyTorch model to ONNX format for optimized inference.
"""

import torch
import onnx
import argparse
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_factory import create_model, load_checkpoint
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
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
        "--output",
        type=str,
        default="onnx/skin_model.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--opset-version", type=int, default=13, help="ONNX opset version"
    )
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    return parser.parse_args()


def export_to_onnx(model, output_path, config, opset_version=13):
    """
    Export PyTorch model to ONNX.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        config: Configuration dictionary
        opset_version: ONNX opset version
    """
    # Set model to eval mode
    model.eval()

    # Create dummy input
    image_size = config["preprocessing"]["image_size"]
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export to ONNX
    logger.info(f"Exporting model to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(f"Model exported to {output_path}")


def verify_onnx_model(onnx_path, pytorch_model, config):
    """
    Verify ONNX model against PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        config: Configuration dictionary
    """
    import onnxruntime as ort
    import numpy as np

    logger.info("Verifying ONNX model...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX model is valid")

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Create test input
    image_size = config["preprocessing"]["image_size"]
    test_input = torch.randn(1, 3, image_size, image_size)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - ort_output).max()
    logger.info(f"Maximum difference between PyTorch and ONNX: {max_diff:.6f}")

    if max_diff < 1e-5:
        logger.info("✓ ONNX model matches PyTorch model")
    else:
        logger.warning(f"⚠ Large difference detected: {max_diff}")


def simplify_onnx_model(input_path, output_path):
    """
    Simplify ONNX model.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save simplified model
    """
    try:
        from onnxsim import simplify

        logger.info("Simplifying ONNX model...")

        # Load model
        model = onnx.load(input_path)

        # Simplify
        model_simplified, check = simplify(model)

        if check:
            # Save simplified model
            onnx.save(model_simplified, output_path)
            logger.info(f"✓ Simplified model saved to {output_path}")
        else:
            logger.warning("⚠ Simplification failed")

    except ImportError:
        logger.warning("onnx-simplifier not installed. Skipping simplification.")


def main():
    """Main export function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    logger.info("Starting ONNX export...")

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    logger.info(f"Loading PyTorch model from {args.checkpoint}")
    model = create_model(config)
    model = load_checkpoint(model, args.checkpoint, device="cpu")
    model.eval()

    # Export to ONNX
    export_to_onnx(model, str(output_path), config, opset_version=args.opset_version)

    # Verify ONNX model
    verify_onnx_model(str(output_path), model, config)

    # Simplify if requested
    if args.simplify:
        simplified_path = output_path.parent / f"{output_path.stem}_simplified.onnx"
        simplify_onnx_model(str(output_path), str(simplified_path))

    logger.info("\nONNX export complete!")

    # Print model info
    onnx_model = onnx.load(str(output_path))
    logger.info(f"\nModel info:")
    logger.info(f"  Input: {onnx_model.graph.input[0].name}")
    logger.info(f"  Output: {onnx_model.graph.output[0].name}")
    logger.info(f"  Opset version: {onnx_model.opset_import[0].version}")


if __name__ == "__main__":
    main()
