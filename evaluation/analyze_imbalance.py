"""
Data Imbalance Analysis Script
Analyzes skin tone and condition distribution to identify bias sources.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config_loader import load_config
from utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def analyze_distribution(config):
    """Analyze and visualize data distribution."""
    processed_dir = Path(config["data"]["processed_dir"])

    # Load splits
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    # Map skin tones to groups
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

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "skin_tone" in df.columns:
            df["tone_group"] = df["skin_tone"].apply(to_group)

    # 1. Overall skin tone distribution
    logger.info("\n=== Overall Skin Tone Distribution ===")
    all_df = pd.concat([train_df, val_df, test_df])
    tone_counts = all_df["tone_group"].value_counts()
    logger.info(f"\n{tone_counts}")
    logger.info(f"\nPercentages:\n{tone_counts / len(all_df) * 100}")

    # 2. Per-split distribution
    logger.info("\n=== Per-Split Distribution ===")
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = df["tone_group"].value_counts()
        logger.info(f"\n{split_name}:\n{counts}")

    # 3. Condition distribution by skin tone
    logger.info("\n=== Condition Distribution by Skin Tone ===")
    conditions = config["model"]["conditions"]

    for cond in conditions:
        col = f"has_{cond}"
        if col in train_df.columns:
            logger.info(f"\n{cond.upper()}:")
            cross = pd.crosstab(
                train_df["tone_group"], train_df[col], margins=True, normalize="index"
            )
            logger.info(cross)

    # 4. Create visualizations
    output_dir = Path("outputs/data_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Skin tone distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (split_name, df) in enumerate(
        [("Train", train_df), ("Val", val_df), ("Test", test_df)]
    ):
        counts = df["tone_group"].value_counts()
        axes[idx].bar(counts.index, counts.values, color=["wheat", "tan", "sienna"])
        axes[idx].set_title(f"{split_name} Set Skin Tone Distribution")
        axes[idx].set_ylabel("Count")
        axes[idx].set_xlabel("Skin Tone Group")
        axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "skin_tone_distribution.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_dir / 'skin_tone_distribution.png'}")
    plt.close()

    # Plot 2: Condition prevalence by skin tone
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]

    for idx, cond in enumerate(conditions):
        col = f"has_{cond}"
        if col in train_df.columns:
            # Calculate prevalence
            prevalence = train_df.groupby("tone_group")[col].mean()
            axes[idx].bar(
                prevalence.index, prevalence.values, color=["wheat", "tan", "sienna"]
            )
            axes[idx].set_title(f"{cond.title()} Prevalence by Skin Tone")
            axes[idx].set_ylabel("Prevalence (proportion)")
            axes[idx].set_xlabel("Skin Tone Group")
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "condition_prevalence.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_dir / 'condition_prevalence.png'}")
    plt.close()

    # Plot 3: Heatmap of condition x skin tone
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap_data = []
    groups = ["light", "medium", "dark"]

    for cond in conditions:
        col = f"has_{cond}"
        if col in train_df.columns:
            row = []
            for group in groups:
                mask = train_df["tone_group"] == group
                if mask.sum() > 0:
                    prevalence = train_df[mask][col].mean()
                    row.append(prevalence)
                else:
                    row.append(0)
            heatmap_data.append(row)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        xticklabels=groups,
        yticklabels=[c.title() for c in conditions],
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Condition Prevalence Heatmap (Train Set)")
    ax.set_xlabel("Skin Tone Group")
    ax.set_ylabel("Condition")

    plt.tight_layout()
    plt.savefig(output_dir / "prevalence_heatmap.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_dir / 'prevalence_heatmap.png'}")
    plt.close()

    # 5. Identify imbalance severity
    logger.info("\n=== Imbalance Analysis ===")

    # Skin tone imbalance ratio
    tone_counts_train = train_df["tone_group"].value_counts()
    if len(tone_counts_train) > 0:
        max_count = tone_counts_train.max()
        min_count = tone_counts_train.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
        logger.info(f"Skin tone imbalance ratio: {imbalance_ratio:.2f}:1")

        # Recommendations
        logger.info("\n=== Recommendations ===")
        if imbalance_ratio > 3:
            logger.warning(
                "⚠️ HIGH IMBALANCE DETECTED! Recommendations:"
                "\n  1. Enable balanced sampling: set training.balance_skin_tones=true"
                "\n  2. Consider collecting more data for underrepresented groups"
                "\n  3. Use per-group threshold optimization"
                "\n  4. Consider group-aware loss weighting"
            )
        elif imbalance_ratio > 1.5:
            logger.info(
                "Moderate imbalance detected. Consider:"
                "\n  1. Balanced sampling for fairness"
                "\n  2. Per-group threshold optimization"
            )
        else:
            logger.info("✓ Skin tone distribution is relatively balanced")

    logger.info("\nAnalysis complete!")


def main():
    """Main analysis function."""
    setup_logger()

    logger.info("Starting data imbalance analysis...")

    # Load config
    config = load_config("configs/config.yaml")

    # Run analysis
    analyze_distribution(config)


if __name__ == "__main__":
    main()
