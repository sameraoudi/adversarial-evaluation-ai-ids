"""
===============================================================================
Script Name   : paths.py
Description   : Defines the project's directory structure dynamically.
                Purpose:
                - Eliminates hardcoded absolute paths (e.g., C:/Users/...).
                - Uses Python's `pathlib` to resolve paths relative to the project root.
                - Provides getter functions for standard directories:
                  - get_project_root(): The base folder of the repo.
                  - get_data_root(): 'data/'
                  - get_models_root(): 'models/<dataset>/'
                  - get_outputs_root(): 'outputs/<dataset>/'

Usage:
    Imported by almost every script in the project to resolve file locations.

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

# src/utils/paths.py
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# PROJECT_ROOT = .../security-testing-IDS
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the project root folder."""
    return PROJECT_ROOT


def get_data_root() -> Path:
    """Return the root data directory."""
    return PROJECT_ROOT / "data"


def get_raw_dir(dataset_name: str) -> Path:
    """
    data/raw/<dataset_name>
    Example: data/raw/cicids2017
    """
    path = get_data_root() / "raw" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_interim_dir(dataset_name: str) -> Path:
    """
    data/interim/<dataset_name>
    """
    path = get_data_root() / "interim" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_dir(dataset_name: str) -> Path:
    """
    data/processed/<dataset_name>
    """
    path = get_data_root() / "processed" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_splits_dir(dataset_name: str) -> Path:
    """
    data/processed/<dataset_name>/splits
    """
    path = get_processed_dir(dataset_name) / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_attack_subsets_dir(dataset_name: str) -> Path:
    """
    data/processed/<dataset_name>/attack_subsets
    """
    path = get_processed_dir(dataset_name) / "attack_subsets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifacts_dir(dataset_name: str) -> Path:
    """
    data/artifacts/<dataset_name>
    """
    path = get_data_root() / "artifacts" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_root(dataset_name: str) -> Path:
    """
    models/<dataset_name>
    """
    path = PROJECT_ROOT / "models" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_root(dataset_name: str) -> Path:
    """
    logs/<dataset_name>
    """
    path = PROJECT_ROOT / "logs" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_outputs_root(dataset_name: str) -> Path:
    """
    outputs/<dataset_name>
    """
    path = PROJECT_ROOT / "outputs" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path
