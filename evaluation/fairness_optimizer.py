"""
Fairness-Aware Threshold Optimization
Optimizes decision thresholds per skin tone group to reduce bias.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class FairnessThresholdOptimizer:
    """Optimizes thresholds to improve fairness across skin tone groups."""

    def __init__(
        self,
        condition_names: List[str],
        skin_tone_groups: List[str] = ["light", "medium", "dark"],
    ):
        """
        Initialize optimizer.

        Args:
            condition_names: List of condition names
            skin_tone_groups: List of skin tone groups
        """
        self.condition_names = condition_names
        self.skin_tone_groups = skin_tone_groups

    def optimize_per_group_thresholds(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        skin_tones: np.ndarray,
        grid: np.ndarray = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize thresholds separately for each skin tone group.

        Args:
            predictions: Model predictions (n_samples, n_classes)
            targets: Ground truth labels (n_samples, n_classes)
            skin_tones: Skin tone labels (n_samples,)
            grid: Threshold grid to search (default: 0.05 to 0.95, step 0.05)

        Returns:
            Dictionary mapping {group: {condition: threshold}}
        """
        if grid is None:
            grid = np.linspace(0.05, 0.95, 19)

        group_thresholds = {}

        for group in self.skin_tone_groups:
            mask = skin_tones == group
            if mask.sum() == 0:
                logger.warning(f"No samples for group {group}; skipping")
                continue

            group_preds = predictions[mask]
            group_targets = targets[mask]

            thresholds = {}
            for i, condition in enumerate(self.condition_names):
                y_true = group_targets[:, i]
                y_scores = group_preds[:, i]

                # Skip if no positives
                if y_true.sum() == 0:
                    thresholds[condition] = 0.5
                    continue

                # Grid search for best F1
                best_t = 0.5
                best_f1 = -1.0
                for t in grid:
                    y_pred = (y_scores >= t).astype(int)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t

                thresholds[condition] = float(best_t)

            group_thresholds[group] = thresholds
            logger.info(f"{group}: {thresholds}")

        return group_thresholds

    def optimize_equalized_odds_thresholds(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        skin_tones: np.ndarray,
        grid: np.ndarray = None,
        target_fpr: float = 0.1,
        target_fnr: float = 0.1,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize thresholds to equalize false positive/negative rates across groups.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            skin_tones: Skin tone labels
            grid: Threshold grid
            target_fpr: Target false positive rate
            target_fnr: Target false negative rate

        Returns:
            Dictionary mapping {group: {condition: threshold}}
        """
        if grid is None:
            grid = np.linspace(0.05, 0.95, 19)

        group_thresholds = {}

        for group in self.skin_tone_groups:
            mask = skin_tones == group
            if mask.sum() == 0:
                continue

            group_preds = predictions[mask]
            group_targets = targets[mask]

            thresholds = {}
            for i, condition in enumerate(self.condition_names):
                y_true = group_targets[:, i]
                y_scores = group_preds[:, i]

                if y_true.sum() == 0 or (y_true == 0).sum() == 0:
                    thresholds[condition] = 0.5
                    continue

                # Find threshold minimizing distance to target rates
                best_t = 0.5
                best_score = float("inf")

                for t in grid:
                    y_pred = (y_scores >= t).astype(int)

                    # Calculate FPR and FNR
                    tp = ((y_pred == 1) & (y_true == 1)).sum()
                    tn = ((y_pred == 0) & (y_true == 0)).sum()
                    fp = ((y_pred == 1) & (y_true == 0)).sum()
                    fn = ((y_pred == 0) & (y_true == 1)).sum()

                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                    # Score: distance to target rates
                    score = (fpr - target_fpr) ** 2 + (fnr - target_fnr) ** 2

                    if score < best_score:
                        best_score = score
                        best_t = t

                thresholds[condition] = float(best_t)

            group_thresholds[group] = thresholds

        return group_thresholds

    def save_thresholds(
        self, group_thresholds: Dict[str, Dict[str, float]], output_path: Path
    ):
        """Save per-group thresholds to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(group_thresholds, f, indent=2)
        logger.info(f"Saved per-group thresholds to {output_path}")

    @staticmethod
    def load_thresholds(path: Path) -> Dict[str, Dict[str, float]]:
        """Load per-group thresholds from JSON."""
        with open(path, "r") as f:
            return json.load(f)


def analyze_skin_tone_distribution(data_csv: str, config: Dict) -> pd.DataFrame:
    """
    Analyze skin tone distribution in the dataset.

    Args:
        data_csv: Path to CSV file
        config: Configuration dictionary

    Returns:
        DataFrame with skin tone statistics
    """
    df = pd.read_csv(data_csv)

    # Map Fitzpatrick to groups
    tone_map = config["data"].get("skin_tones", {})
    light = set(map(str, tone_map.get("light", [])))
    medium = set(map(str, tone_map.get("medium", [])))
    dark = set(map(str, tone_map.get("dark", [])))

    def to_group(x):
        x_clean = str(x).strip()
        if x_clean in light:
            return "light"
        if x_clean in medium:
            return "medium"
        if x_clean in dark:
            return "dark"
        return "unknown"

    if "skin_tone" in df.columns:
        df["tone_group"] = df["skin_tone"].apply(to_group)

        # Get distribution
        stats = df["tone_group"].value_counts()
        logger.info(f"Skin tone distribution:\n{stats}")

        # Per-condition distribution
        conditions = config["model"]["conditions"]
        for cond in conditions:
            col = f"has_{cond}"
            if col in df.columns:
                logger.info(f"\n{cond} by skin tone:")
                cross = pd.crosstab(df["tone_group"], df[col])
                logger.info(cross)

        return df
    else:
        logger.warning("No skin_tone column found in dataset")
        return df
