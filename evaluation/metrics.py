"""
Evaluation Metrics Module
Calculates metrics for model evaluation.
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple, Union


class MetricsCalculator:
    """Calculator for classification metrics."""

    def __init__(
        self,
        condition_names: List[str],
        threshold: Union[float, List[float], np.ndarray] = 0.5,
    ):
        """
        Initialize metrics calculator.

        Args:
            condition_names: List of condition names
            threshold: Threshold for binary classification
        """
        self.condition_names = condition_names
        self.threshold = threshold

    def _binarize(self, predictions: np.ndarray) -> np.ndarray:
        """
        Binarize predictions using scalar or per-class thresholds.

        Args:
            predictions: (n_samples, n_classes)

        Returns:
            Binary array (n_samples, n_classes)
        """
        if isinstance(self.threshold, (list, np.ndarray)):
            thr = np.array(self.threshold, dtype=float).reshape(1, -1)
            return (predictions >= thr).astype(int)
        else:
            return (predictions >= float(self.threshold)).astype(int)

    def calculate_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            predictions: Model predictions (n_samples, n_classes)
            targets: Ground truth labels (n_samples, n_classes)

        Returns:
            Dictionary of metrics
        """
        # Binarize predictions
        pred_binary = self._binarize(predictions)

        # Calculate metrics
        metrics = {}

        # Overall metrics
        metrics["precision"] = precision_score(
            targets, pred_binary, average="macro", zero_division=0
        )
        metrics["recall"] = recall_score(
            targets, pred_binary, average="macro", zero_division=0
        )
        metrics["f1_score"] = f1_score(
            targets, pred_binary, average="macro", zero_division=0
        )

        # AUROC
        try:
            metrics["auroc"] = roc_auc_score(targets, predictions, average="macro")
        except ValueError:
            metrics["auroc"] = 0.0

        # Per-class metrics
        for i, condition in enumerate(self.condition_names):
            metrics[f"{condition}_precision"] = precision_score(
                targets[:, i], pred_binary[:, i], zero_division=0
            )
            metrics[f"{condition}_recall"] = recall_score(
                targets[:, i], pred_binary[:, i], zero_division=0
            )
            metrics[f"{condition}_f1"] = f1_score(
                targets[:, i], pred_binary[:, i], zero_division=0
            )

            try:
                metrics[f"{condition}_auroc"] = roc_auc_score(
                    targets[:, i], predictions[:, i]
                )
            except ValueError:
                metrics[f"{condition}_auroc"] = 0.0

        return metrics

    def get_confusion_matrices(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get confusion matrices for each class.

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary of confusion matrices
        """
        pred_binary = self._binarize(predictions)

        matrices = {}
        for i, condition in enumerate(self.condition_names):
            matrices[condition] = confusion_matrix(targets[:, i], pred_binary[:, i])

        return matrices

    def get_classification_report(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> str:
        """
        Get classification report.

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            Classification report string
        """
        pred_binary = self._binarize(predictions)

        report = classification_report(
            targets, pred_binary, target_names=self.condition_names, zero_division=0
        )

        return report


class FairnessMetrics:
    """Calculator for fairness metrics across skin tones."""

    def __init__(self, condition_names: List[str], skin_tone_groups: List[str]):
        """
        Initialize fairness metrics calculator.

        Args:
            condition_names: List of condition names
            skin_tone_groups: List of skin tone group names
        """
        self.condition_names = condition_names
        self.skin_tone_groups = skin_tone_groups

    def calculate_group_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        skin_tones: np.ndarray,
        threshold: Union[float, List[float], np.ndarray] = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics stratified by skin tone.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            skin_tones: Skin tone labels for each sample
            threshold: Classification threshold

        Returns:
            Dictionary of metrics per skin tone group
        """
        if isinstance(threshold, (list, np.ndarray)):
            thr = np.array(threshold, dtype=float).reshape(1, -1)
            pred_binary = (predictions >= thr).astype(int)
        else:
            pred_binary = (predictions >= float(threshold)).astype(int)

        group_metrics = {}

        for group in self.skin_tone_groups:
            # Get samples for this group
            mask = skin_tones == group

            if mask.sum() == 0:
                continue

            group_pred = pred_binary[mask]
            group_target = targets[mask]

            # Calculate per-condition metrics and average
            # Use samples averaging for multi-label classification
            # This averages across samples first, then across conditions
            try:
                # For multi-label, use 'samples' averaging which is more appropriate
                precision = precision_score(
                    group_target, group_pred, average="samples", zero_division=0
                )
                recall = recall_score(
                    group_target, group_pred, average="samples", zero_division=0
                )
                f1 = f1_score(
                    group_target, group_pred, average="samples", zero_division=0
                )
            except ValueError:
                # Fallback to weighted if samples fails (e.g., all zeros)
                precision = precision_score(
                    group_target, group_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    group_target, group_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    group_target, group_pred, average="weighted", zero_division=0
                )

            group_metrics[group] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "sample_count": mask.sum(),
            }

        return group_metrics

    def calculate_per_condition_group_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        skin_tones: np.ndarray,
        threshold: Union[float, List[float], np.ndarray] = 0.5,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate metrics per condition per skin tone group.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            skin_tones: Skin tone labels for each sample
            threshold: Classification threshold

        Returns:
            Dictionary of metrics per condition per group
            Format: {condition: {group: {metric: value}}}
        """
        if isinstance(threshold, (list, np.ndarray)):
            thr = np.array(threshold, dtype=float).reshape(1, -1)
            pred_binary = (predictions >= thr).astype(int)
        else:
            pred_binary = (predictions >= float(threshold)).astype(int)

        per_condition_metrics = {}

        for i, condition in enumerate(self.condition_names):
            per_condition_metrics[condition] = {}

            for group in self.skin_tone_groups:
                # Get samples for this group
                mask = skin_tones == group

                if mask.sum() == 0:
                    continue

                group_pred = pred_binary[mask, i]
                group_target = targets[mask, i]

                # Calculate metrics for this condition and group
                per_condition_metrics[condition][group] = {
                    "precision": precision_score(
                        group_target, group_pred, zero_division=0
                    ),
                    "recall": recall_score(group_target, group_pred, zero_division=0),
                    "f1_score": f1_score(group_target, group_pred, zero_division=0),
                    "sample_count": mask.sum(),
                    "positive_count": group_target.sum(),
                }

        return per_condition_metrics

    def calculate_fairness_gaps(
        self, group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate fairness gaps between groups.

        Args:
            group_metrics: Metrics per skin tone group

        Returns:
            Dictionary of fairness gaps
        """
        gaps = {}

        metric_names = ["precision", "recall", "f1_score"]

        for metric in metric_names:
            values = [group_metrics[group][metric] for group in group_metrics.keys()]

            if len(values) > 0:
                gaps[f"{metric}_gap"] = max(values) - min(values)
                gaps[f"{metric}_std"] = np.std(values)

        return gaps

    def calculate_false_rates(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        skin_tones: np.ndarray,
        threshold: Union[float, List[float], np.ndarray] = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate false positive and false negative rates per group.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            skin_tones: Skin tone labels
            threshold: Classification threshold

        Returns:
            Dictionary of false rates per group
        """
        if isinstance(threshold, (list, np.ndarray)):
            thr = np.array(threshold, dtype=float).reshape(1, -1)
            pred_binary = (predictions >= thr).astype(int)
        else:
            pred_binary = (predictions >= float(threshold)).astype(int)

        false_rates = {}

        for group in self.skin_tone_groups:
            mask = skin_tones == group

            if mask.sum() == 0:
                continue

            group_pred = pred_binary[mask]
            group_target = targets[mask]

            # Calculate confusion matrix components
            tp = ((group_pred == 1) & (group_target == 1)).sum()
            tn = ((group_pred == 0) & (group_target == 0)).sum()
            fp = ((group_pred == 1) & (group_target == 0)).sum()
            fn = ((group_pred == 0) & (group_target == 1)).sum()

            # Calculate rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            false_rates[group] = {
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
            }

        return false_rates
