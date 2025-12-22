"""
===============================================================================
Script Name   : io.py
Description   : Input/Output utilities for the data pipeline.
    
    Functions:
    - load_raw_csvs_from_config: Aggregates multiple raw CSV files (e.g., per-day PCAPs).
    - save_npz / load_npz: Efficiently stores numpy arrays for training.
    - save_label_mapping: Persists the LabelEncoder mapping for reproducibility.
    - save_feature_list: Saves the canonical order of features to ensure 
      model inputs align with the dataset columns.

Usage:
    Used by preprocess.py and preprocess_cicids2018.py.

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

# src/data/io.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ..utils.paths import get_project_root, get_artifacts_dir

logger = logging.getLogger(__name__)


def load_raw_csvs_from_config(dataset_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and concatenate all raw CSV files using the dataset config.

    Expects in dataset_cfg:
      raw.csv_root   -> e.g. "data/raw/cicids2017/CSVs"
      raw.csv_subdir -> e.g. "MachineLearningCVE"
      raw.pattern    -> e.g. "*.csv"
    """
    raw_cfg = dataset_cfg.get("raw", {})
    csv_root = raw_cfg.get("csv_root")
    csv_subdir = raw_cfg.get("csv_subdir", "")
    pattern = raw_cfg.get("pattern", "*.csv")

    if csv_root is None:
        raise ValueError("dataset_cfg['raw']['csv_root'] must be set")

    project_root = get_project_root()
    base_dir = project_root / csv_root
    csv_dir = base_dir / csv_subdir if csv_subdir else base_dir

    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_files: List[Path] = sorted(csv_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir} with pattern {pattern}")

    logger.info("Loading %d CSV files from %s", len(csv_files), csv_dir)
    dfs = []
    for f in csv_files:
        logger.info("Reading %s", f)
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info("Loaded raw dataset: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """
    Save multiple numpy arrays into a compressed .npz file.

    Example:
        save_npz(file, X=X, y=y)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    logger.info("Saved npz file to %s", path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """
    Load arrays from a .npz file.

    Returns
    -------
    dict
        Mapping from array-name to numpy array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"npz file not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    """
    Save a DataFrame to CSV with index disabled.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved DataFrame to %s", path)


def save_label_mapping(dataset_name: str, mapping: Dict[str, int]) -> Path:
    """
    Save label mapping (string label -> integer id) to
    data/artifacts/<dataset_name>/label_mapping.csv
    """
    artifacts_dir = get_artifacts_dir(dataset_name)
    path = artifacts_dir / "label_mapping.csv"
    df = pd.DataFrame(
        [{"label": label, "id": idx} for label, idx in mapping.items()]
    ).sort_values("id")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved label mapping to %s", path)
    return path


def save_feature_list(dataset_name: str, feature_names: List[str]) -> Path:
    """
    Save the canonical feature order to
    data/artifacts/<dataset_name>/features.csv.
    """
    artifacts_dir = get_artifacts_dir(dataset_name)
    path = artifacts_dir / "features.csv"
    df = pd.DataFrame({"feature": feature_names})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved feature list to %s", path)
    return path
