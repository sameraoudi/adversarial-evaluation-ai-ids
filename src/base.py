"""
===============================================================================
Script Name   : base.py
Description   : efines the core data structures for the adversarial attack framework.
              Classes:
              - AttackConfig: Configuration dataclass for attack parameters (epsilon, 
                norm, steps, batch size).
              - AttackResult: Container dataclass for attack outputs (adversarial examples, 
                prediction logs, success masks, and perturbation norms).

Usage:
    Imported by specific attack implementations (fgsm.py, pgd.py) and the orchestrator.

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

# src/attacks/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class AttackConfig:
    """
    Generic configuration for gradient-based attacks.

    Attributes
    ----------
    norm : {"linf", "l2"}
        Norm used to constrain perturbations.
    eps : float
        Maximum perturbation magnitude (radius of Lp ball).
    step_size : float
        Step size for iterative attacks (PGD). For FGSM, this is typically
        ignored and eps is used directly.
    num_steps : int
        Number of optimization steps (PGD); 1 step degenerates to FGSM-like.
    targeted : bool
        If True, perform targeted attack; otherwise untargeted.
        (Targeted behaviour is not yet implemented but kept for extensibility.)
    clip_min : float
        Lower bound for feature values (typically 0.0 after MinMax scaling).
    clip_max : float
        Upper bound for feature values (typically 1.0 after MinMax scaling).
    batch_size : int
        Batch size for generating adversarial examples.
    """
    norm: str = "linf"
    eps: float = 0.03
    step_size: float = 0.01
    num_steps: int = 40
    targeted: bool = False
    clip_min: float = 0.0
    clip_max: float = 1.0
    batch_size: int = 1024


@dataclass
class AttackResult:
    """
    Container for the outputs of an adversarial attack.

    Attributes
    ----------
    x_adv : np.ndarray
        Adversarial examples (same shape as clean inputs).
    y_true : np.ndarray
        True labels.
    y_pred_clean : np.ndarray
        Model predictions on clean inputs.
    y_pred_adv : np.ndarray
        Model predictions on adversarial inputs.
    success : np.ndarray
        Boolean mask indicating whether the attack succeeded for each sample.
    norms : Dict[str, np.ndarray]
        Dictionary containing perturbation norms (e.g., {"linf": ..., "l2": ...}).
    meta : Dict[str, Any]
        Optional metadata (e.g., eps, model_name, attack_type).
    """
    x_adv: np.ndarray
    y_true: np.ndarray
    y_pred_clean: np.ndarray
    y_pred_adv: np.ndarray
    success: np.ndarray
    norms: Dict[str, np.ndarray]
    meta: Optional[Dict[str, Any]] = None
