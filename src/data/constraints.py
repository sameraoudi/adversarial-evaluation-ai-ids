"""
===============================================================================
Script Name   : constraints.py
Description   : Manages semantic feature constraints for adversarial perturbation.
                To generate realistic adversarial examples, we cannot simply add noise to all features.
                This module:
                1. Defines valid ranges for features (e.g., [0, 1] after scaling).
                2. (Optional) Defines "Immutable" features that attackers cannot modify 
                   (e.g., Protocol ID, Destination Port) to preserve protocol validity.
                3. Provides projection functions to enforce these constraints after 
                   gradient updates.

Usage:
    Used by src/attacks/utils.py during FGSM/PGD generation.

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

# src/data/constraints.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from ..utils.paths import get_artifacts_dir

logger = logging.getLogger(__name__)


@dataclass
class FeatureConstraint:
    """
    Metadata describing how a single feature may be perturbed.
    """
    name: str
    index: int
    perturbable: bool = True
    feature_type: str = "continuous"  # placeholder; can be "count", "time", etc.
    min_value: float = 0.0
    max_value: float = 1.0


def build_default_constraints(dataset_name: str, feature_names: List[str]) -> Path:
    """
    Build and save default constraints for a dataset, assuming all
    features are continuous and scaled to [0,1].

    Later, you can manually edit the generated JSON to encode
    protocol-aware rules (non-perturbable fields, integer counts, etc.).
    """
    constraints: List[FeatureConstraint] = []
    for idx, name in enumerate(feature_names):
        constraints.append(
            FeatureConstraint(
                name=name,
                index=idx,
                perturbable=True,
                feature_type="continuous",
                min_value=0.0,
                max_value=1.0,
            )
        )

    artifacts_dir = get_artifacts_dir(dataset_name)
    path = artifacts_dir / "constraints.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"features": [asdict(c) for c in constraints]},
            f,
            indent=2,
        )

    logger.info("Saved default constraints to %s", path)
    return path


def load_constraints(dataset_name: str) -> Dict[str, Any]:
    """
    Load constraints JSON for a dataset.
    """
    artifacts_dir = get_artifacts_dir(dataset_name)
    path = artifacts_dir / "constraints.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"constraints.json not found for dataset '{dataset_name}' at {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def project_to_unit_interval(x: np.ndarray) -> np.ndarray:
    """
    Clip array values to [0, 1]. This assumes MinMax scaling was used.
    """
    return np.clip(x, 0.0, 1.0)


def apply_feature_constraints(
    x_adv: np.ndarray,
    constraints: Dict[str, Any],
) -> np.ndarray:
    """
    Apply simple per-feature min/max constraints and non-perturbable flags.

    Parameters
    ----------
    x_adv : np.ndarray
        Adversarial samples, shape (N, D).
    constraints : dict
        Loaded constraints.json dictionary.

    Returns
    -------
    np.ndarray
        Projected adversarial samples that respect min/max and
        non-perturbable features (to the extent encoded).
    """
    feats = constraints.get("features", [])
    x_proj = x_adv.copy()

    for feat in feats:
        idx = feat["index"]
        if idx >= x_proj.shape[1]:
            continue

        if not feat.get("perturbable", True):
            # In later attack code, we can enforce "no change" by
            # restoring the original value. Here we simply clip.
            # The attack modules will be responsible for respecting this.
            continue

        min_v = feat.get("min_value", 0.0)
        max_v = feat.get("max_value", 1.0)
        x_proj[:, idx] = np.clip(x_proj[:, idx], min_v, max_v)

    # Final global projection to [0,1]
    x_proj = np.clip(x_proj, 0.0, 1.0)
    return x_proj
