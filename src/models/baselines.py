"""
===============================================================================
Script Name   : utils.py
Description   : Implements classical machine learning baselines for the IDS evaluation.
                Models:
                1. Random Forest (RF): A robust ensemble method using decision trees.
                2. Logistic Regression (LR): A linear baseline for comparison.
                
                Key Functions:
                - train_random_forest_baseline: Trains RF using scikit-learn and saves the model.
                - train_logreg_baseline: Trains LR using scikit-learn and saves the model.
                - _evaluate_model: Computes classification metrics (Accuracy, F1, Confusion Matrix).

Usage:
    Imported by scripts/train_baselines_cicids2017.py.

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

# src/models/baselines.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from ..data.io import load_npz
from ..utils.paths import get_models_root, get_outputs_root
from ..utils.seed import set_global_seed
from ..utils.paths import get_splits_dir

logger = logging.getLogger(__name__)


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_depth: int | None = None
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class LRConfig:
    max_iter: int = 1000
    C: float = 1.0
    solver: str = "lbfgs"
    n_jobs: int = -1
    random_state: int = 42


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def _save_metrics(
    dataset_name: str,
    run_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Save classification report and confusion matrix to outputs/<dataset>/runs/<run_name>/.
    """
    outputs_root = get_outputs_root(dataset_name)
    run_dir = outputs_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    report_str = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Save report as text
    report_path = run_dir / f"{model_name}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info("Saved classification report to %s", report_path)

    # Save confusion matrix as .npy for later plotting
    cm_path = run_dir / f"{model_name}_confusion_matrix.npy"
    np.save(cm_path, cm)
    logger.info("Saved confusion matrix to %s", cm_path)


def train_random_forest_baseline(
    dataset_name: str,
    run_name: str = "baseline_rf_lr",
    rf_cfg: RFConfig | None = None,
    seed: int = 42,
) -> RandomForestClassifier:
    """
    Train a RandomForest baseline on preprocessed CICIDS-style splits.

    Uses:
      data/processed/<dataset_name>/splits/train.npz
      data/processed/<dataset_name>/splits/val.npz
      data/processed/<dataset_name>/splits/test.npz
    """
    set_global_seed(seed)

    if rf_cfg is None:
        rf_cfg = RFConfig(random_state=seed)

    models_root = get_models_root(dataset_name)
    rf_dir = models_root / "baselines" / "rf"
    rf_dir.mkdir(parents=True, exist_ok=True)
    model_path = rf_dir / "rf_model.pkl"

    # Load data
    splits_dir = get_splits_dir(dataset_name)
    train = load_npz(splits_dir / "train.npz")
    val = load_npz(splits_dir / "val.npz")
    test = load_npz(splits_dir / "test.npz")

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_test, y_test = test["X"], test["y"]

    logger.info(
        "Training RandomForest on %d train samples, %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    model = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        n_jobs=rf_cfg.n_jobs,
        random_state=rf_cfg.random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate on val and test
    logger.info("Evaluating RandomForest on validation set")
    y_val_pred = model.predict(X_val)
    _save_metrics(dataset_name, run_name, "rf_val", y_val, y_val_pred)

    logger.info("Evaluating RandomForest on test set")
    y_test_pred = model.predict(X_test)
    _save_metrics(dataset_name, run_name, "rf_test", y_test, y_test_pred)

    # Save model
    _save_model(model, model_path)

    return model


def train_logreg_baseline(
    dataset_name: str,
    run_name: str = "baseline_rf_lr",
    lr_cfg: LRConfig | None = None,
    seed: int = 42,
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline on preprocessed CICIDS-style splits.
    """
    set_global_seed(seed)

    if lr_cfg is None:
        lr_cfg = LRConfig(random_state=seed)

    models_root = get_models_root(dataset_name)
    lr_dir = models_root / "baselines" / "lr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    model_path = lr_dir / "lr_model.pkl"

    # Load data
    splits_dir = Path("data") / "processed" / dataset_name / "splits"
    train = load_npz(splits_dir / "train.npz")
    val = load_npz(splits_dir / "val.npz")
    test = load_npz(splits_dir / "test.npz")

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_test, y_test = test["X"], test["y"]

    logger.info(
        "Training LogisticRegression on %d train samples, %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    model = LogisticRegression(
        max_iter=lr_cfg.max_iter,
        C=lr_cfg.C,
        solver=lr_cfg.solver,
        n_jobs=lr_cfg.n_jobs,
        random_state=lr_cfg.random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate on val and test
    logger.info("Evaluating LogisticRegression on validation set")
    y_val_pred = model.predict(X_val)
    _save_metrics(dataset_name, run_name, "lr_val", y_val, y_val_pred)

    logger.info("Evaluating LogisticRegression on test set")
    y_test_pred = model.predict(X_test)
    _save_metrics(dataset_name, run_name, "lr_test", y_test, y_test_pred)

    # Save model
    _save_model(model, model_path)

    return model
