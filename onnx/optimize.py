"""
ONNX Optimization Script
Optimizes ONNX model for better inference performance.
"""

import onnx
import argparse
import logging
from pathlib import Path

import sys

sys.path.append("..")

from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize ONNX model")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save optimized model"
    )
    return parser.parse_args()


def optimize_onnx_model(input_path, output_path):
    """
    Optimize ONNX model.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
    """
    logger.info(f"Loading ONNX model from {input_path}")

    # Load model
    model = onnx.load(input_path)

    # Check model
    onnx.checker.check_model(model)
    logger.info("✓ Model is valid")

    # Optimize using ONNX optimizer
    try:
        from onnxoptimizer import optimize

        logger.info("Optimizing model...")

        # Available passes
        passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]

        optimized_model = optimize(model, passes)

        # Save optimized model
        onnx.save(optimized_model, output_path)
        logger.info(f"✓ Optimized model saved to {output_path}")

        # Compare sizes
        original_size = Path(input_path).stat().st_size / (1024 * 1024)
        optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

        logger.info(f"\nModel sizes:")
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  Optimized: {optimized_size:.2f} MB")
        logger.info(f"  Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")

    except ImportError:
        logger.warning("onnxoptimizer not installed. Skipping optimization.")
        logger.info("Install with: pip install onnxoptimizer")


def main():
    """Main optimization function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    logger.info("Starting ONNX optimization...")

    # Optimize model
    optimize_onnx_model(args.input, args.output)

    logger.info("\nOptimization complete!")


if __name__ == "__main__":
    main()
