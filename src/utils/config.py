"""
===============================================================================
Script Name   : config.py
Description   : Utility module for loading configuration files.
                Function: load_dataset_config
                - Reads YAML files from 'configs/datasets/' (e.g., cicids2017.yaml).
                - Returns a dictionary containing dataset-specific parameters like 
                  raw file paths, split ratios, and label column names.

Usage:
    Used by src/data/preprocess.py to configure the data pipeline.

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

# src/utils/config.py
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

from .paths import get_project_root


def load_dataset_config(name: str) -> Dict[str, Any]:
    """
    Load dataset config from configs/datasets/<name>.yaml,
    where <name> is e.g. 'cicids2017'.
    """
    root = get_project_root()
    path = root / "configs" / "datasets" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
