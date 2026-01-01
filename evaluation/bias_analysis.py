"""
Bias Analysis Module
Analyzes model bias across different skin tones.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

from .metrics import FairnessMetrics

logger = logging.getLogger(__name__)


class BiasAnalyzer:
    """Analyzer for model bias across skin tones."""

    def __init__(
        self,
        condition_names: List[str],
        skin_tone_groups: List[str],
        output_dir: str = "outputs/bias_analysis",
    ):
        """
        Initialize bias analyzer.

        Args:
            condition_names: List of condition names
            skin_tone_groups: List of skin tone group names
            output_dir: Directory to save analysis results
        """
        self.condition_names = condition_names
        self.skin_tone_groups = skin_tone_groups
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fairness_metrics = FairnessMetrics(condition_names, skin_tone_groups)

    def analyze(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        skin_tones: np.ndarray,
        threshold: float = 0.5,
    ):
        """
        Perform comprehensive bias analysis.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            skin_tones: Skin tone labels
            threshold: Classification threshold
        """
        logger.info("Performing bias analysis...")

        # Calculate overall group metrics (averaged across conditions)
        group_metrics = self.fairness_metrics.calculate_group_metrics(
            predictions, targets, skin_tones, threshold
        )

        # Calculate per-condition per-group metrics for detailed analysis
        per_condition_metrics = (
            self.fairness_metrics.calculate_per_condition_group_metrics(
                predictions, targets, skin_tones, threshold
            )
        )

        # Calculate fairness gaps
        fairness_gaps = self.fairness_metrics.calculate_fairness_gaps(group_metrics)

        # Calculate false rates
        false_rates = self.fairness_metrics.calculate_false_rates(
            predictions, targets, skin_tones, threshold
        )

        # Log results
        self._log_results(
            group_metrics, fairness_gaps, false_rates, per_condition_metrics
        )

        # Visualize results
        self._visualize_group_metrics(group_metrics)
        self._visualize_per_condition_metrics(per_condition_metrics)
        self._visualize_false_rates(false_rates)
        self._visualize_fairness_gaps(fairness_gaps)

        logger.info(f"Bias analysis complete. Results saved to {self.output_dir}")

    def _log_results(
        self,
        group_metrics: Dict,
        fairness_gaps: Dict,
        false_rates: Dict,
        per_condition_metrics: Dict,
    ):
        """Log analysis results."""
        logger.info("\n=== Overall Group Metrics (averaged across conditions) ===")
        for group, metrics in group_metrics.items():
            logger.info(f"\n{group}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        logger.info("\n=== Per-Condition Per-Group Metrics ===")
        for condition, groups in per_condition_metrics.items():
            logger.info(f"\n{condition}:")
            for group, metrics in groups.items():
                logger.info(f"  {group}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, int):
                            logger.info(f"    {metric}: {value}")
                        else:
                            logger.info(f"    {metric}: {value:.4f}")

        logger.info("\n=== Fairness Gaps ===")
        for metric, value in fairness_gaps.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("\n=== False Rates ===")
        for group, rates in false_rates.items():
            logger.info(f"\n{group}:")
            for rate, value in rates.items():
                logger.info(f"  {rate}: {value:.4f}")

    def _visualize_group_metrics(self, group_metrics: Dict):
        """Visualize metrics across groups."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_to_plot = ["precision", "recall", "f1_score"]

        for i, metric in enumerate(metrics_to_plot):
            groups = list(group_metrics.keys())
            values = [group_metrics[g][metric] for g in groups]

            axes[i].bar(groups, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Skin Tone')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim([0, 1])
            axes[i].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "group_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _visualize_per_condition_metrics(self, per_condition_metrics: Dict):
        """Visualize metrics per condition per group."""
        metrics_to_plot = ["precision", "recall", "f1_score"]
        n_conditions = len(self.condition_names)

        fig, axes = plt.subplots(n_conditions, 3, figsize=(15, 5 * n_conditions))

        # Handle single condition case
        if n_conditions == 1:
            axes = axes.reshape(1, -1)

        for i, condition in enumerate(self.condition_names):
            condition_data = per_condition_metrics.get(condition, {})

            for j, metric in enumerate(metrics_to_plot):
                groups = list(condition_data.keys())
                if not groups:
                    axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
                    axes[i, j].set_title(
                        f'{condition} - {metric.replace("_", " ").title()}'
                    )
                    continue

                values = [condition_data[g][metric] for g in groups]

                axes[i, j].bar(groups, values)
                axes[i, j].set_title(
                    f'{condition} - {metric.replace("_", " ").title()}'
                )
                axes[i, j].set_ylabel(metric.replace("_", " ").title())
                axes[i, j].set_ylim([0, 1])
                axes[i, j].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "per_condition_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _visualize_false_rates(self, false_rates: Dict):
        """Visualize false positive and false negative rates."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        groups = list(false_rates.keys())
        fpr_values = [false_rates[g]["false_positive_rate"] for g in groups]
        fnr_values = [false_rates[g]["false_negative_rate"] for g in groups]

        # False Positive Rate
        ax1.bar(groups, fpr_values, color="coral")
        ax1.set_title("False Positive Rate by Skin Tone")
        ax1.set_ylabel("False Positive Rate")
        ax1.set_ylim([0, 1])
        ax1.grid(axis="y", alpha=0.3)

        # False Negative Rate
        ax2.bar(groups, fnr_values, color="skyblue")
        ax2.set_title("False Negative Rate by Skin Tone")
        ax2.set_ylabel("False Negative Rate")
        ax2.set_ylim([0, 1])
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "false_rates.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _visualize_fairness_gaps(self, fairness_gaps: Dict):
        """Visualize fairness gaps."""
        fig, ax = plt.subplots(figsize=(10, 6))

        gap_metrics = {k: v for k, v in fairness_gaps.items() if k.endswith("_gap")}

        metrics = list(gap_metrics.keys())
        values = list(gap_metrics.values())

        ax.barh(metrics, values, color="purple", alpha=0.7)
        ax.set_xlabel("Gap Value")
        ax.set_title("Fairness Gaps Across Skin Tone Groups")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "fairness_gaps.png", dpi=300, bbox_inches="tight")
        plt.close()
