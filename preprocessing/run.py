"""
Main Preprocessing Script
Orchestrates the complete preprocessing pipeline.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import logging
from typing import Dict, List, Tuple

from face_detection import FaceDetector, process_image
from augmentation import apply_clahe, get_augmentation_pipeline
from label_mapping import get_label_flags

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main preprocessing pipeline for skin condition analysis."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize preprocessor.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup directories
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.processed_dir = Path(self.config["data"]["processed_dir"])
        self.augmented_dir = Path(self.config["data"]["augmented_dir"])

        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_dir.mkdir(parents=True, exist_ok=True)

        # Initialize face detector
        detector_method = self.config["preprocessing"].get("face_detector", "opencv")
        self.face_detector = FaceDetector(method=detector_method)

        # Image size
        self.image_size = self.config["preprocessing"]["image_size"]

        # Max images per dataset
        self.max_images = self.config["data"].get("max_images_per_dataset", None)

        # Datasets to skip face detection
        self.skip_face_detection = self.config["data"].get("skip_face_detection", [])

        # Load Fitzpatrick17k metadata for skin tone labels
        self.fitzpatrick_metadata = self._load_fitzpatrick_metadata()

    def _load_fitzpatrick_metadata(self) -> pd.DataFrame:
        """Load Fitzpatrick17k metadata CSV for skin tone labels."""
        fitzpatrick_path = self.raw_dir / "fitzpatrick17k"

        # Look for metadata CSV
        if not fitzpatrick_path.exists():
            logger.warning(
                "Fitzpatrick17k folder not found; skin tones will be 'unknown'"
            )
            return pd.DataFrame()

        metadata_files = list(fitzpatrick_path.glob("*.csv"))

        if not metadata_files:
            logger.warning(
                "No Fitzpatrick17k metadata CSV found; skin tones will be 'unknown'"
            )
            return pd.DataFrame()

        # Load first CSV found
        metadata_csv = metadata_files[0]
        logger.info(f"Loading Fitzpatrick17k metadata from: {metadata_csv}")

        try:
            df = pd.read_csv(metadata_csv)

            # Build lookup by image hash (filename stem)
            if "md5hash" in df.columns and "fitzpatrick_scale" in df.columns:
                df["skin_tone"] = df["fitzpatrick_scale"].astype(str)
                # Map -1 (unknown) to 'unknown'
                df.loc[df["skin_tone"] == "-1", "skin_tone"] = "unknown"
                logger.info(
                    f"Loaded skin tone metadata for {len(df)} Fitzpatrick17k images"
                )
                return df[["md5hash", "skin_tone"]].set_index("md5hash")
            else:
                logger.warning(
                    "Fitzpatrick CSV missing required columns (md5hash, fitzpatrick_scale)"
                )
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Fitzpatrick metadata: {e}")
            return pd.DataFrame()

    def get_skin_tone(self, dataset: str, img_path: Path) -> str:
        """
        Get skin tone label for an image.

        Args:
            dataset: Dataset name
            img_path: Path to image file

        Returns:
            Skin tone label (1-6 for Fitzpatrick, 'unknown' otherwise)
        """
        if dataset.lower() == "fitzpatrick17k" and not self.fitzpatrick_metadata.empty:
            # Use filename stem as hash lookup
            img_hash = img_path.stem

            if img_hash in self.fitzpatrick_metadata.index:
                return self.fitzpatrick_metadata.loc[img_hash, "skin_tone"]

        # Default for all other datasets
        return "unknown"

    def process_single_image(
        self, image_path: str, save_path: str, skip_face_detection: bool = False
    ) -> bool:
        """
        Process a single image.

        Args:
            image_path: Path to input image
            save_path: Path to save processed image
            skip_face_detection: Skip face detection (for pre-aligned datasets)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if skip_face_detection:
                # Use entire image (already face-aligned)
                face = image
            else:
                # Detect and crop face
                bbox = self.face_detector.detect_face(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )

                if bbox is None:
                    logger.warning(f"No face detected in: {image_path}")
                    return False

                # Crop face
                face = self.face_detector.crop_face(image, bbox)

            # Resize
            face = cv2.resize(face, (self.image_size, self.image_size))

            # Apply CLAHE if enabled
            face = apply_clahe(face, self.config)

            # Save processed image
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

            return True

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False

    def process_dataset(self, dataset_name: str):
        """
        Process entire dataset.

        Args:
            dataset_name: Name of dataset to process
        """
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset directory
        dataset_dir = self.raw_dir / dataset_name
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return

        # Check if face detection should be skipped
        skip_face_detection = dataset_name in self.skip_face_detection
        if skip_face_detection:
            logger.info(
                f"Skipping face detection for {dataset_name} (pre-aligned dataset)"
            )

        # Find all images
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_dir.rglob(f"*{ext}"))

        # Apply limit if specified
        total_images = len(image_files)
        if self.max_images and total_images > self.max_images:
            logger.info(
                f"Limiting {dataset_name} from {total_images} to {self.max_images} images"
            )
            image_files = image_files[: self.max_images]

        logger.info(f"Found {len(image_files)} images to process")

        # Process each image
        successful = 0
        failed = 0

        for img_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
            # Create corresponding output path
            rel_path = img_path.relative_to(dataset_dir)
            output_path = self.processed_dir / dataset_name / rel_path

            if self.process_single_image(
                str(img_path), str(output_path), skip_face_detection
            ):
                successful += 1
            else:
                failed += 1

        logger.info(f"Processing complete: {successful} successful, {failed} failed")

    def create_metadata(self):
        """Create metadata CSV for processed images."""
        logger.info("Creating metadata file...")

        metadata = []

        # Scan processed directory
        for img_path in self.processed_dir.rglob("*.jpg"):
            rel_path = img_path.relative_to(self.processed_dir)
            parts = rel_path.parts

            # Extract information from path structure
            # Assuming structure: dataset_name/condition/skin_tone/image.jpg
            if len(parts) >= 3:
                dataset_name = parts[0]

                # Get skin tone from Fitzpatrick metadata if available
                skin_tone = self.get_skin_tone(dataset_name, img_path)

                # Build label flags using dataset-aware mapping
                flags = get_label_flags(
                    dataset=dataset_name,
                    relative_path=str(rel_path),
                    filename=img_path.name,
                )
                metadata.append(
                    {
                        "image_path": str(img_path),
                        "relative_path": str(rel_path),
                        "dataset": dataset_name,
                        "condition": parts[1] if len(parts) > 1 else "unknown",
                        "skin_tone": skin_tone,  # Use extracted Fitzpatrick labels
                        "filename": img_path.name,
                        # Multi-label targets expected by data loader
                        "has_acne": flags["has_acne"],
                        "has_pigmentation": flags["has_pigmentation"],
                        "has_wrinkles": flags["has_wrinkles"],
                    }
                )

        # Create DataFrame
        df = pd.DataFrame(metadata)

        # Save to CSV
        metadata_path = self.processed_dir / "metadata.csv"
        df.to_csv(metadata_path, index=False)

        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Total processed images: {len(df)}")

    def split_data(self):
        """Split data into train/val/test sets."""
        logger.info("Splitting data into train/val/test...")

        # Load metadata
        metadata_path = self.processed_dir / "metadata.csv"
        if not metadata_path.exists():
            logger.error("Metadata file not found. Run create_metadata first.")
            return

        df = pd.read_csv(metadata_path)

        # Get split ratios
        train_ratio = self.config["data"]["split_ratios"]["train"]
        val_ratio = self.config["data"]["split_ratios"]["val"]
        test_ratio = self.config["data"]["split_ratios"]["test"]

        # Shuffle data
        df = df.sample(
            frac=1, random_state=self.config["data"]["random_seed"]
        ).reset_index(drop=True)

        # Calculate split indices
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        # Split data
        train_df = df[:train_idx]
        val_df = df[train_idx:val_idx]
        test_df = df[val_idx:]

        # Save splits
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "val.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)

        logger.info(f"Data split complete:")
        logger.info(f"  Train: {len(train_df)} images")
        logger.info(f"  Val: {len(val_df)} images")
        logger.info(f"  Test: {len(test_df)} images")


def main():
    """Main preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline...")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Process datasets
    datasets = preprocessor.config["data"].get("datasets", [])

    for dataset in datasets:
        preprocessor.process_dataset(dataset)

    # Create metadata
    preprocessor.create_metadata()

    # Split data
    preprocessor.split_data()

    logger.info("Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()
