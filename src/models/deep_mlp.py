"""
===============================================================================
Script Name   : deep_mlp.py
Description   : Defines the Multilayer Perceptron (MLP) architecture and training loop.
    
                Architecture:
                - Fully connected layers with ReLU activations.
                - Dropout for regularization.
                - Configurable depth and width via MLPConfig.
                
                Training Logic:
                - Uses PyTorch DataLoader for batch processing.
                - Implements Early Stopping based on Validation Loss.
                - Saves the best model checkpoint to 'models/<dataset>/deep/mlp/'.

Usage:
    Imported by scripts/train_deep_cicids2017.py (for both target and surrogate models).

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Nov-22

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.
===============================================================================
"""

# src/models/deep_mlp.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..data.io import load_npz
from ..utils.paths import get_splits_dir, get_models_root, get_outputs_root
from ..utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_dims: List[int]
    dropout: float = 0.3
    batch_size: int = 1024
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    seed: int = 42


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_dataloaders(
    dataset_name: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Load train/val/test splits from npz and wrap them in DataLoaders.
    Returns loaders and (input_dim, num_classes).
    """
    splits_dir = get_splits_dir(dataset_name)
    train = load_npz(splits_dir / "train.npz")
    val = load_npz(splits_dir / "val.npz")
    test = load_npz(splits_dir / "test.npz")

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_test, y_test = test["X"], test["y"]

    input_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1

    logger.info(
        "MLP data: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
        input_dim,
        num_classes,
    )

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, input_dim, num_classes


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a loader and return (loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def train_mlp(
    dataset_name: str,
    run_name: str,
    cfg: MLPConfig,
) -> Dict[str, Any]:
    """
    Train an MLP on the CICIDS-style splits with early stopping on val loss.

    Returns a dict with basic metrics and the path to the best checkpoint.
    """
    set_global_seed(cfg.seed)

    train_loader, val_loader, test_loader, input_dim, num_classes = _build_dataloaders(
        dataset_name, cfg.batch_size
    )

    # Update input_dim/num_classes in case config placeholders were used
    cfg.input_dim = input_dim
    cfg.num_classes = num_classes

    device = torch.device("cpu")
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    models_root = get_models_root(dataset_name)
    mlp_dir = models_root / "deep" / "mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = mlp_dir / f"{run_name}_best.pt"

    outputs_root = get_outputs_root(dataset_name)
    run_dir = outputs_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "mlp_metrics.txt"

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    logger.info("Starting MLP training for up to %d epochs", cfg.max_epochs)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

        train_loss = epoch_loss / max(1, total_samples)
        val_loss, val_acc = _evaluate_model(model, val_loader, device)

        logger.info(
            "[MLP][Epoch %d] train_loss=%.6f val_loss=%.6f val_acc=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_acc,
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
            logger.info("New best val_loss=%.6f, resetting patience", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    patience_counter,
                )
                break

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final test evaluation
    test_loss, test_acc = _evaluate_model(model, test_loader, device)
    logger.info("[MLP] Final test_loss=%.6f test_acc=%.4f", test_loss, test_acc)

    # Save best checkpoint
    torch.save(model.state_dict(), best_ckpt_path)
    logger.info("Saved best MLP checkpoint to %s", best_ckpt_path)

    # Save simple metrics text file
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc: {test_acc:.6f}\n")

    return {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "checkpoint": str(best_ckpt_path),
    }
