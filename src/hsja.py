"""
===============================================================================
Script Name   : hsja.py
Description   : Implementation of the HopSkipJump Attack (HSJA).
                Type: Decision-Based Black-Box Attack
                Paper Reference: Chen et al. (2020)
                
                HSJA generates adversarial examples using only the final class decision (hard label),
                without access to gradients or confidence scores. It relies on binary search 
                to find the decision boundary and estimates the gradient direction via random queries.
                Used here to attack RF, LR, MLP, and CNN directly.

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

# src/attacks/hsja.py
from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np

from .base import AttackConfig, AttackResult
from .utils import (
    FeatureConstraints,
    project_lp_ball,
    apply_feature_constraints,
    compute_linf_norm,
    compute_l2_norm,
)


def _find_initial_adversarial(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y_true: int,
    max_tries: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """
    Find an initial adversarial example by random perturbation within [0,1]^D.
    """
    if rng is None:
        rng = np.random.default_rng()
    D = x.shape[0]
    for _ in range(max_tries):
        x_rand = rng.random(D, dtype=np.float32)
        y_rand = predict(x_rand[None, :])[0]
        if y_rand != y_true:
            return x_rand
    return None


def hsja_attack(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    config: AttackConfig,
    constraints: FeatureConstraints | None = None,
    max_iters: int = 30,
    num_binary_search_steps: int = 10,
    step_size: float = 0.01,
    random_state: int = 42,
) -> AttackResult:
    """
    Simplified HSJA-style decision-based black-box attack.

    This implementation is designed to be conceptually faithful but
    computationally lighter. It is appropriate for smaller subsets.

    Parameters
    ----------
    predict : callable
        f(X) -> labels of shape (N,).
    X : np.ndarray
        Clean inputs, shape (N, D).
    y : np.ndarray
        True labels.
    config : AttackConfig
        Norm / eps parameters.
    constraints : FeatureConstraints, optional
        Constraints applied after each update.
    max_iters : int
        Maximum refinement iterations.
    num_binary_search_steps : int
        Number of binary search steps to approximate the decision boundary.
    step_size : float
        Step size for random perturbations.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    AttackResult
    """
    rng = np.random.default_rng(random_state)
    N, D = X.shape

    if constraints is None:
        constraints = FeatureConstraints(clip_min=config.clip_min, clip_max=config.clip_max)

    x_adv = X.copy()
    y_pred_clean = predict(X)

    # 1) Find initial adversarial examples for each correctly classified point
    for i in range(N):
        if y_pred_clean[i] != y[i]:
            continue
        init_adv = _find_initial_adversarial(predict, X[i], y[i], rng=rng)
        if init_adv is not None:
            x_adv[i] = init_adv

    # 2) Refine using binary search + random steps
    import torch
    X_t = torch.from_numpy(X.astype(np.float32))

    for _ in range(max_iters):
        x_adv_t = torch.from_numpy(x_adv.astype(np.float32))

        # Binary search towards decision boundary
        for i in range(N):
            if predict(X[i:i+1])[0] != y[i] and predict(x_adv[i:i+1])[0] == y[i]:
                # edge case: we lost adversariality, skip
                continue

            lo = X_t[i].clone()
            hi = x_adv_t[i].clone()

            # Ensure hi is adversarial
            if predict(hi[None, :].numpy())[0] == y[i]:
                continue

            for _ in range(num_binary_search_steps):
                mid = (lo + hi) / 2.0
                y_mid = predict(mid[None, :].numpy())[0]
                if y_mid == y[i]:
                    lo = mid
                else:
                    hi = mid

            x_adv_t[i] = hi

        # Random perturbations around current adversarial point
        noise = torch.randn_like(x_adv_t) * step_size
        x_adv_t = x_adv_t + noise

        x_adv_t = project_lp_ball(x_adv_t, X_t, eps=config.eps, norm=config.norm)
        x_adv_t = apply_feature_constraints(x_adv_t, X_t, constraints)
        x_adv = x_adv_t.cpu().numpy()

    y_pred_adv = predict(x_adv)
    y_true = y.copy()
    success = (y_pred_clean == y_true) & (y_pred_adv != y_true)
    delta = x_adv - X
    norms = {
        "linf": compute_linf_norm(delta),
        "l2": compute_l2_norm(delta),
    }

    meta: Dict[str, Any] = {
        "attack": "hsja",
        "norm": config.norm,
        "eps": config.eps,
        "max_iters": max_iters,
        "num_binary_search_steps": num_binary_search_steps,
        "step_size": step_size,
    }

    return AttackResult(
        x_adv=x_adv,
        y_true=y_true,
        y_pred_clean=y_pred_clean,
        y_pred_adv=y_pred_adv,
        success=success,
        norms=norms,
        meta=meta,
    )
