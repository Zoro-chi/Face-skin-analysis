"""
Face Detection Module
Handles face detection and alignment for preprocessing pipeline.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using OpenCV Haar Cascades or DNN-based detector."""

    def __init__(self, method: str = "opencv", conf_threshold: float = 0.5):
        """
        Initialize face detector.

        Args:
            method: Detection method ('opencv', 'dnn')
            conf_threshold: Confidence threshold for detection
        """
        self.method = method
        self.conf_threshold = conf_threshold

        if method == "opencv":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        elif method == "dnn":
            # Load DNN face detector
            self.detector = cv2.dnn.readNetFromCaffe(
                "models/deploy.prototxt",
                "models/res10_300x300_ssd_iter_140000.caffemodel",
            )
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image.

        Args:
            image: Input image (BGR format)

        Returns:
            Bounding box (x, y, w, h) or None if no face detected
        """
        if self.method == "opencv":
            return self._detect_opencv(image)
        elif self.method == "dnn":
            return self._detect_dnn(image)

    def _detect_opencv(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            logger.warning("No face detected")
            return None

        # Return largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)

    def _detect_dnn(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using DNN-based detector."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        self.detector.setInput(blob)
        detections = self.detector.forward()

        # Find detection with highest confidence
        best_detection = None
        max_confidence = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold and confidence > max_confidence:
                max_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                best_detection = (x1, y1, x2 - x1, y2 - y1)

        if best_detection is None:
            logger.warning("No face detected above confidence threshold")
            return None

        return best_detection

    def crop_face(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.2
    ) -> np.ndarray:
        """
        Crop face from image with margin.

        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            margin: Margin to add around face (as fraction of bbox size)

        Returns:
            Cropped face image
        """
        x, y, w, h = bbox

        # Add margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)

        return image[y1:y2, x1:x2]

    def align_face(self, image: np.ndarray) -> np.ndarray:
        """
        Align face based on facial landmarks.
        Note: Basic implementation - can be enhanced with dlib or similar.

        Args:
            image: Face image

        Returns:
            Aligned face image
        """
        # Placeholder for facial landmark-based alignment
        # For now, just return the image
        return image


def process_image(
    image_path: str, output_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Process single image: detect, crop, and resize face.

    Args:
        image_path: Path to input image
        output_size: Output image size

    Returns:
        Processed face image or None if processing fails
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    # Detect face
    detector = FaceDetector(method="opencv")
    bbox = detector.detect_face(image)

    if bbox is None:
        return None

    # Crop face
    face = detector.crop_face(image, bbox)

    # Align face
    face = detector.align_face(face)

    # Resize
    face = cv2.resize(face, output_size)

    return face
