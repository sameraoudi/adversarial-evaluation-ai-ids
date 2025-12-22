"""
===============================================================================
Script Name   : splits.py
Description   : Implements stratified splitting logic for imbalanced intrusion detection datasets.
                Function: stratified_train_val_test_split
                - Ensures that minority attack classes are represented proportionally 
                  across Train, Validation, and Test sets.
                - Prevents "zero-shot" issues where a rare attack appears in Test but not Train.

Usage:
    Used by preprocess.py to generate the 70/10/20 splits.

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

# src/data/splits.py
from __future__ import annotations

import logging
from typing import Tuple, Dict

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified train/val/test splits.

    The split is done in two stages:
      1) train+val vs test
      2) train vs val (inside train+val)
    so that val_size is relative to the full dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    test_size : float
        Fraction of data to reserve for test.
    val_size : float
        Fraction of data to reserve for validation (of the full dataset).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
          "train": (X_train, y_train),
          "val":   (X_val, y_val),
          "test":  (X_test, y_test)
        }
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0,1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    logger.info(
        "Creating stratified train/val/test splits (test_size=%.2f, val_size=%.2f)",
        test_size, val_size
    )

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Compute val size relative to the remaining temp set
    val_ratio_temp = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_temp,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
