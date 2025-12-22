"""
===============================================================================
Script Name   : preprocess.py
Description   : Implements the core preprocessing pipeline for CICIDS-style datasets (specifically CICIDS2017).
    
                Workflow:
                1. Loads raw CSV configuration from YAML.
                2. Cleans column names and identifies the Label column.
                3. Coerces non-numeric features to NaN and drops invalid rows (Inf/NaN).
                4. Encodes string labels to integers (LabelEncoder).
                5. Performs a stratified Train (70%) / Validation (10%) / Test (20%) split.
                6. Fits a MinMaxScaler on the Training set and applies it to all splits.
                7. Saves artifacts: Scaled Data (.npz), Scaler (.pkl), Label Mapping (.csv), 
                   and Feature Constraints (.json).

Usage:
    Called by scripts/prepare_cicids2017.py.
    
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

# src/data/preprocess.py
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

from .io import (
    load_raw_csvs_from_config,
    save_npz,
    save_label_mapping,
    save_feature_list,
)
from .splits import stratified_train_val_test_split
from .constraints import build_default_constraints
from ..utils.paths import get_artifacts_dir, get_splits_dir
from ..utils.config import load_dataset_config

logger = logging.getLogger(__name__)


def _identify_label_column(df: pd.DataFrame) -> str:
    """
    Identify the label column in CICIDS-style datasets when no explicit
    label column is configured.

    Strategy:
      - Strip whitespace from column names and lower-case them.
      - Look for a column whose normalized name is exactly 'label'.
    """
    normalized_to_original = {
        col.strip().lower(): col for col in df.columns
    }

    for key, original in normalized_to_original.items():
        if key == "label":
            return original

    raise ValueError(
        f"Could not find a label column in columns: {list(df.columns)}"
    )


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from column names.

    CICIDS2017 CSVs often have leading spaces, e.g. ' Flow Duration'.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _convert_features_to_numeric(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Convert feature columns to numeric, coercing errors to NaN.

    Non-numeric artefacts (e.g., 'Infinity', 'NaN', or mixed-type columns)
    are coerced to NaN and then removed in a later cleaning step.

    The label column is left as string.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != label_col]

    # Coerce all feature columns to numeric
    df_features = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Ensure label column is treated as string
    df[label_col] = df[label_col].astype(str)
    df[feature_cols] = df_features

    return df


def _remove_invalid_rows(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Drop rows with NaN or infinite values in features, or missing labels.

    This uses numpy's isfinite to ensure no +/-inf get through.
    """
    df = df.copy()
    initial_rows = df.shape[0]

    # Separate features for validity check
    features_df = df.drop(columns=[label_col])

    # Convert to numpy for finite check
    features_values = features_df.to_numpy(dtype=np.float64)

    # Finite mask: True only if all feature values in the row are finite
    finite_mask = np.isfinite(features_values).all(axis=1)

    # Label mask: non-missing labels
    label_mask = df[label_col].notna().to_numpy()

    mask_valid = finite_mask & label_mask

    cleaned = df.loc[mask_valid].copy()
    dropped = initial_rows - cleaned.shape[0]

    if dropped > 0:
        logger.info(
            "Dropped %d rows with NaN/inf/invalid values (remaining=%d)",
            dropped,
            cleaned.shape[0],
        )
    else:
        logger.info("No rows dropped for NaN/inf/invalid values")

    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def prepare_dataset_cicids_style(
    dataset_name: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    End-to-end preprocessing pipeline for CICIDS-style datasets,
    driven by configs/datasets/<dataset_name>.yaml.

    Expected dataset config structure (YAML):

        dataset_name: cicids2017

        raw:
          csv_root: data/raw/cicids2017/CSVs
          csv_subdir: MachineLearningCVE
          pattern: "*.csv"

        label:
          column: "Label"

        splits:
          test_size: 0.2
          val_size: 0.1
          random_state: 42

        scaling:
          type: "minmax"

    Steps:
      1. Load dataset config.
      2. Load all raw CSVs according to the configured paths.
      3. Clean column names (strip whitespace).
      4. Determine label column (from config or via heuristic).
      5. Convert features to numeric (coerce errors to NaN).
      6. Drop rows with invalid (NaN/inf) features or labels.
      7. Encode labels using LabelEncoder.
      8. Stratified train/val/test split.
      9. Fit MinMaxScaler on train only; transform all splits.
     10. Save splits as .npz, plus scaler, label mapping, feature list,
         and default constraints JSON.

    Returns
    -------
    dict
        Splits with arrays:
        {
          "train": (X_train, y_train),
          "val": (X_val, y_val),
          "test": (X_test, y_test)
        }
    """
    logger.info("=== Preparing dataset '%s' ===", dataset_name)

    # ------------------------------------------------------------------
    # 1) Load dataset config
    # ------------------------------------------------------------------
    dataset_cfg = load_dataset_config(dataset_name)

    splits_cfg = dataset_cfg.get("splits", {})
    test_size = float(splits_cfg.get("test_size", 0.2))
    val_size = float(splits_cfg.get("val_size", 0.1))
    random_state = int(splits_cfg.get("random_state", 42))

    # ------------------------------------------------------------------
    # 2) Load raw CSVs according to config (MachineLearningCVE, etc.)
    # ------------------------------------------------------------------
    df_raw = load_raw_csvs_from_config(dataset_cfg)
    logger.info("Raw shape: %s", (df_raw.shape[0], df_raw.shape[1]))

    # ------------------------------------------------------------------
    # 3) Clean column names (strip whitespace)
    # ------------------------------------------------------------------
    df = _clean_column_names(df_raw)

    # ------------------------------------------------------------------
    # 4) Identify label column (explicit from config or heuristic)
    # ------------------------------------------------------------------
    label_cfg = dataset_cfg.get("label", {})
    explicit_label_col = label_cfg.get("column")

    if explicit_label_col is not None:
        label_col = explicit_label_col.strip()
        if label_col not in df.columns:
            raise ValueError(
                f"Configured label column '{explicit_label_col}' not found in columns: {list(df.columns)}"
            )
        logger.info("Using configured label column: %s", label_col)
    else:
        label_col = _identify_label_column(df)
        logger.info("Using inferred label column: %s", label_col)

    # ------------------------------------------------------------------
    # 5) Convert features to numeric (coerce to NaN where needed)
    # ------------------------------------------------------------------
    df = _convert_features_to_numeric(df, label_col=label_col)

    # ------------------------------------------------------------------
    # 6) Drop invalid rows (NaN/inf features or missing label)
    # ------------------------------------------------------------------
    df = _remove_invalid_rows(df, label_col=label_col)

    # ------------------------------------------------------------------
    # Separate features and labels
    # ------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != label_col]
    X_df = df[feature_cols]
    y_series = df[label_col]

    logger.info(
        "After cleaning: %d rows, %d features",
        X_df.shape[0],
        X_df.shape[1],
    )

    # ------------------------------------------------------------------
    # 7) Label encoding
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series.values)

    # Mapping from original label string -> integer ID
    label_mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}
    save_label_mapping(dataset_name, label_mapping)
    logger.info("Label mapping: %s", label_mapping)

    # Convert features to numpy arrays
    X_full = X_df.to_numpy(dtype=np.float32)
    y_full = y_encoded.astype(np.int64)

    # ------------------------------------------------------------------
    # 8) Stratified train/val/test split
    # ------------------------------------------------------------------
    splits = stratified_train_val_test_split(
        X_full,
        y_full,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # ------------------------------------------------------------------
    # 9) Fit MinMaxScaler on train; transform all splits
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Save scaler
    artifacts_dir = get_artifacts_dir(dataset_name)
    scaler_path = artifacts_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved MinMaxScaler to %s", scaler_path)

    # ------------------------------------------------------------------
    # 10) Save feature order & default constraints metadata
    # ------------------------------------------------------------------
    save_feature_list(dataset_name, feature_cols)
    build_default_constraints(dataset_name, feature_cols)

    # Save splits
    splits_dir = get_splits_dir(dataset_name)
    save_npz(splits_dir / "train.npz", X=X_train_scaled, y=y_train)
    save_npz(splits_dir / "val.npz", X=X_val_scaled, y=y_val)
    save_npz(splits_dir / "test.npz", X=X_test_scaled, y=y_test)

    logger.info("Finished preparing dataset '%s'", dataset_name)

    return {
        "train": (X_train_scaled, y_train),
        "val": (X_val_scaled, y_val),
        "test": (X_test_scaled, y_test),
    }
