"""
Trainer Module
Handles model training logic with MLflow tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import logging
from typing import Dict, Any
from pathlib import Path

from .losses import create_loss_function

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for skin condition detection models."""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = "cuda"):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training config
        self.num_epochs = config["training"]["num_epochs"]
        self.lr = config["training"]["learning_rate"]

        # Loss function
        loss_type = config["training"].get("loss", "bce")
        class_weights = config["training"].get("class_weights")
        self.criterion = create_loss_function(loss_type, class_weights)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Early stopping
        self.early_stopping_config = config["training"].get("early_stopping", {})
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.config["training"].get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.config["training"].get("weight_decay", 1e-4),
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.config["training"].get("weight_decay", 1e-4),
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.config["training"].get("weight_decay", 1e-4),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config["training"].get("scheduler", "cosine").lower()

        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        else:
            return None

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        return {"train_loss": avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(val_loader)
        return {"val_loss": avg_loss}

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early.

        Args:
            val_loss: Validation loss

        Returns:
            True if should stop, False otherwise
        """
        if not self.early_stopping_config.get("enabled", False):
            return False

        patience = self.early_stopping_config.get("patience", 10)
        min_delta = self.early_stopping_config.get("min_delta", 0.001)

        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs")
                return True
            return False

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Log metrics
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Log to MLflow
            mlflow.log_metrics(metrics, step=epoch)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Save checkpoint
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            save_best_only = self.config["training"].get("save_best_only", True)
            if not save_best_only or is_best:
                self.save_checkpoint(epoch, metrics, is_best=is_best)

            # Early stopping
            if self.check_early_stopping(val_metrics["val_loss"]):
                logger.info("Training stopped early")
                break

        logger.info("Training complete!")
