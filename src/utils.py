"""
===============================================================================
Script Name   : run_attacks_cicids2018_friday.py
Description   : Utility functions for adversarial perturbation constraints and projections.
                Key Functions:
                - project_lp_ball: Projects perturbations back into the L-inf or L2 ball.
                - apply_feature_constraints: Enforces domain-specific constraints (e.g., 
                  clipping features to [0,1], handling immutable features).
                - compute_linf_norm / compute_l2_norm: Metric calculations for evaluation.

Usage:
    Used by all attack scripts to ensure generated samples are valid.

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

# src/attacks/utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch


@dataclass
class FeatureConstraints:
    """
    Feature-level constraints for adversarial perturbations.

    clip_min and clip_max can be either scalars or per-feature arrays.

    Attributes
    ----------
    clip_min : float or np.ndarray
    clip_max : float or np.ndarray
    mutable_mask : Optional[np.ndarray]
        Boolean mask of length D; False indicates a feature that must not change.
    """
    clip_min: float | np.ndarray = 0.0
    clip_max: float | np.ndarray = 1.0
    mutable_mask: Optional[np.ndarray] = None


def apply_feature_constraints(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    constraints: FeatureConstraints,
) -> torch.Tensor:
    """
    Apply feature-level constraints:
      - Enforce immutability (if mutable_mask provided).
      - Clip each feature to its allowed range.

    Supports both scalar and per-feature clip bounds.
    """
    x_new = x_adv

    # Enforce immutability
    if constraints.mutable_mask is not None:
        mask = torch.from_numpy(constraints.mutable_mask.astype(bool)).to(x_new.device)
        while mask.dim() < x_new.dim():
            mask = mask.unsqueeze(0)
        x_new = torch.where(mask, x_new, x_orig)

    # Clip
    if isinstance(constraints.clip_min, np.ndarray):
        cmin = torch.from_numpy(constraints.clip_min.astype(np.float32)).to(x_new.device)
        while cmin.dim() < x_new.dim():
            cmin = cmin.unsqueeze(0)
        x_new = torch.maximum(x_new, cmin)
    else:
        x_new = torch.clamp(x_new, min=float(constraints.clip_min))

    if isinstance(constraints.clip_max, np.ndarray):
        cmax = torch.from_numpy(constraints.clip_max.astype(np.float32)).to(x_new.device)
        while cmax.dim() < x_new.dim():
            cmax = cmax.unsqueeze(0)
        x_new = torch.minimum(x_new, cmax)
    else:
        x_new = torch.clamp(x_new, max=float(constraints.clip_max))

    return x_new


def compute_linf_norm(delta: np.ndarray) -> np.ndarray:
    """
    Compute L-infinity norm per-sample.

    Parameters
    ----------
    delta : np.ndarray
        Perturbation array of shape (N, D).

    Returns
    -------
    np.ndarray
        L-infinity norm for each sample, shape (N,).
    """
    return np.max(np.abs(delta), axis=1)


def compute_l2_norm(delta: np.ndarray) -> np.ndarray:
    """
    Compute L2 norm per-sample.

    Parameters
    ----------
    delta : np.ndarray
        Perturbation array of shape (N, D).

    Returns
    -------
    np.ndarray
        L2 norm for each sample, shape (N,).
    """
    return np.linalg.norm(delta, ord=2, axis=1)


def project_lp_ball(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    eps: float,
    norm: str = "linf",
) -> torch.Tensor:
    """
    Project adversarial examples to an Lp ball around the original inputs.

    Parameters
    ----------
    x_adv : torch.Tensor
        Current adversarial samples.
    x_orig : torch.Tensor
        Clean samples.
    eps : float
        Radius of the ball.
    norm : {"linf", "l2"}
        Norm type.

    Returns
    -------
    torch.Tensor
        Projected adversarial samples.
    """
    delta = x_adv - x_orig
    if norm == "linf":
        delta = torch.clamp(delta, min=-eps, max=eps)
    elif norm == "l2":
        # Flatten per sample
        delta_flat = delta.view(delta.size(0), -1)
        norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-12
        factor = torch.clamp(eps / norms, max=1.0)
        delta_flat = delta_flat * factor
        delta = delta_flat.view_as(delta)
    else:
        raise ValueError(f"Unsupported norm '{norm}', expected 'linf' or 'l2'")
    return x_orig + delta


def apply_feature_constraints(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    constraints: FeatureConstraints,
) -> torch.Tensor:
    """
    Apply simple feature-level constraints:
      - Enforce immutable features (if mutable_mask is given).
      - Clip to [clip_min, clip_max].

    Parameters
    ----------
    x_adv : torch.Tensor
        Adversarial samples.
    x_orig : torch.Tensor
        Original clean samples.
    constraints : FeatureConstraints
        Constraint specification.

    Returns
    -------
    torch.Tensor
        Constrained adversarial samples.
    """
    x_new = x_adv

    # Enforce immutability
    if constraints.mutable_mask is not None:
        # mutable_mask: shape (D,)
        mask = torch.from_numpy(constraints.mutable_mask.astype(bool)).to(x_new.device)
        # Expand to batch shape (N, D)
        while mask.dim() < x_new.dim():
            mask = mask.unsqueeze(0)
        x_new = torch.where(mask, x_new, x_orig)

    # Global clipping
    x_new = torch.clamp(x_new, min=constraints.clip_min, max=constraints.clip_max)
    return x_new


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Detach and move a tensor to CPU as a NumPy array.
    """
    return x.detach().cpu().numpy()
