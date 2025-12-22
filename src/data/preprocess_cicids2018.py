"""
===============================================================================
Script Name   : preprocess_cicids2018.py
Description   : pecialized preprocessing pipeline for the CICIDS2018 (Friday 02-03-2018) slice.
                Differences from standard preprocessing:
                - Handles specific 2018 column drops (e.g., 'Timestamp').
                - Maps diverse 2018 attack labels into a binary format (0=Benign, 1=Attack) 
                  for cross-dataset evaluation.
                - Fits a separate StandardScaler/MinMaxScaler specific to this distribution 
                  (or applies the one from 2017 depending on configuration).
                - Saves the processed test set for the `run_attacks_cicids2018_friday.py` script.

Usage:
    Called by scripts/prepare_cicids2018_friday.py.

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

# src/data/preprocess_cicids2018.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from ..utils.paths import get_project_root, get_splits_dir, get_artifacts_dir
from ..data.io import save_npz

logger = logging.getLogger(__name__)


@dataclass
class CICIDS2018Config:
    dataset_name: str = "cicids2018_friday"
    raw_rel_path: str = (
        "data/raw/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    )
    label_column: str = "Label"
    benign_labels: Tuple[str, ...] = ("benign", "benign ")  # normalized to lower
    test_size: float = 0.15
    val_size: float = 0.15  # fraction of train
    random_state: int = 42
    drop_columns: Tuple[str, ...] = ("Timestamp",)  # can extend if needed


def _resolve_raw_path(cfg: CICIDS2018Config) -> Path:
    root = get_project_root()
    raw_path = root / cfg.raw_rel_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CICIDS2018 CSV not found at {raw_path}")
    return raw_path


def _load_raw_cicids2018(cfg: CICIDS2018Config) -> pd.DataFrame:
    raw_path = _resolve_raw_path(cfg)
    logger.info("Loading CICIDS2018 raw CSV from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded CICIDS2018 slice with shape %s", df.shape)
    return df


def _clean_and_split(
    df: pd.DataFrame,
    cfg: CICIDS2018Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Clean raw CICIDS2018 slice and produce X_train, X_val, X_test, y_train, y_val, y_test.
    Binary label: 0 = Benign, 1 = Attack.
    """
    # Drop columns we do not want as features
    for col in cfg.drop_columns:
        if col in df.columns:
            logger.info("Dropping non-numeric column: %s", col)
            df = df.drop(columns=[col])

    if cfg.label_column not in df.columns:
        raise KeyError(f"Label column '{cfg.label_column}' not found in CICIDS2018 CSV")

    # Extract labels and features
    labels_raw = df[cfg.label_column].astype(str).str.strip()
    X_df = df.drop(columns=[cfg.label_column])

    # Identify numeric feature columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X_df.columns if c not in numeric_cols]

    if non_numeric_cols:
        logger.warning(
            "Dropping non-numeric feature columns in CICIDS2018: %s",
            non_numeric_cols,
        )
        X_df = X_df[numeric_cols]

    logger.info("Using %d numeric features for CICIDS2018 slice", len(numeric_cols))

    # Replace inf / -inf with NaN and drop rows with NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    n_before = len(X_df)
    mask_valid = ~X_df.isna().any(axis=1)
    X_df = X_df[mask_valid]
    labels_raw = labels_raw[mask_valid]
    n_after = len(X_df)

    if n_after < n_before:
        logger.info("Dropped %d rows with NaN/inf in CICIDS2018 slice", n_before - n_after)

    # Binary label mapping: 0 = benign, 1 = attack
    labels_norm = labels_raw.str.lower()
    benign_mask = labels_norm.isin(cfg.benign_labels)
    y_binary = np.where(benign_mask, 0, 1).astype(np.int64)

    # Optional: record original string mapping for documentation
    unique_labels = sorted(labels_raw.unique())
    label_mapping = {0: "Benign", 1: "Attack (any)"}
    logger.info(
        "CICIDS2018 label summary: %s",
        {k: int((y_binary == k).sum()) for k in [0, 1]},
    )
    logger.info("Original unique string labels in slice: %s", unique_labels)

    X = X_df.to_numpy(dtype=np.float32)

    # Train/val/test split (stratified by binary label)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_binary,
        test_size=cfg.test_size + cfg.val_size,
        random_state=cfg.random_state,
        stratify=y_binary,
    )

    relative_test = cfg.test_size / (cfg.test_size + cfg.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        random_state=cfg.random_state,
        stratify=y_temp,
    )

    logger.info(
        "CICIDS2018 splits: train=%d, val=%d, test=%d",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, label_mapping


def _fit_and_apply_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, scaler


def prepare_cicids2018_friday_slice(cfg: CICIDS2018Config | None = None) -> None:
    """
    Full preprocessing pipeline for the Friday 02-03-2018 CICIDS2018 CSV slice.

    - Loads raw CSV
    - Drops non-numeric / timestamp columns
    - Encodes labels to binary (Benign vs Attack)
    - Creates stratified train/val/test splits
    - Fits a StandardScaler on train
    - Saves splits as NPZ and scaler + label mapping under artifacts
    """
    if cfg is None:
        cfg = CICIDS2018Config()

    df = _load_raw_cicids2018(cfg)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_mapping,
    ) = _clean_and_split(df, cfg)

    X_train_s, X_val_s, X_test_s, scaler = _fit_and_apply_scaler(
        X_train, X_val, X_test
    )

    # Save splits
    splits_dir = get_splits_dir(cfg.dataset_name)
    splits_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving CICIDS2018 splits to %s", splits_dir)

    save_npz(
        splits_dir / "train.npz",
        X=X_train_s,
        y=y_train,
    )
    save_npz(
        splits_dir / "val.npz",
        X=X_val_s,
        y=y_val,
    )
    save_npz(
        splits_dir / "test.npz",
        X=X_test_s,
        y=y_test,
    )

    # Save scaler and label mapping
    artifacts_dir = get_artifacts_dir(cfg.dataset_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = artifacts_dir / "scaler.pkl"
    labels_path = artifacts_dir / "label_mapping_cicids2018_friday.json"

    joblib.dump(scaler, scaler_path)
    pd.Series(label_mapping).to_json(labels_path)

    logger.info("Saved CICIDS2018 scaler to %s", scaler_path)
    logger.info("Saved CICIDS2018 label mapping to %s", labels_path)
    logger.info("CICIDS2018 Friday slice preprocessing completed.")
