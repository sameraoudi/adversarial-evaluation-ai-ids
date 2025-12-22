"""
===============================================================================
Script Name   : zoo.py
Description   : Implementation of Zeroth Order Optimization (ZOO).
                Type: Score-Based Black-Box Attack
                Paper Reference: Chen et al. (2017)
                
                ZOO estimates gradients by querying the target model's confidence scores 
                (probabilities) and using coordinate-wise finite differences. It is 
                gradient-free but requires access to model confidence.
                Used here to attack RF and LR directly.

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

# src/attacks/zoo.py
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


def zoo_attack(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    config: AttackConfig,
    constraints: FeatureConstraints | None = None,
    max_iters: int = 50,
    fd_epsilon: float = 1e-3,
    coords_per_iter: int = 8,
    random_state: int = 42,
) -> AttackResult:
    """
    Simplified ZOO-style score-based black-box attack.

    This implementation is CPU-friendly and intended for small subsets of
    the test set. It estimates gradients via finite differences on a small
    number of randomly selected coordinates per iteration.

    Parameters
    ----------
    predict_proba : callable
        f(X) -> probabilities of shape (N, C).
    X : np.ndarray
        Clean inputs, shape (N, D).
    y : np.ndarray
        True labels.
    config : AttackConfig
        Norm / eps parameters (step_size is used here).
    constraints : FeatureConstraints, optional
        Feature constraints applied after each update.
    max_iters : int
        Max iterations.
    fd_epsilon : float
        Finite-difference step for gradient estimation.
    coords_per_iter : int
        Number of coordinates to sample per iteration.
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
    # initial predictions
    probs_clean = predict_proba(X)
    y_pred_clean = np.argmax(probs_clean, axis=1)

    for it in range(max_iters):
        # estimate gradient for each sample on a subset of coordinates
        grad = np.zeros_like(x_adv, dtype=np.float32)

        for i in range(N):
            # only bother attacking samples that are currently correctly classified
            if y_pred_clean[i] != y[i]:
                continue

            idx_coords = rng.choice(D, size=min(coords_per_iter, D), replace=False)
            for j in idx_coords:
                e = np.zeros(D, dtype=np.float32)
                e[j] = fd_epsilon

                x_plus = x_adv[i] + e
                x_minus = x_adv[i] - e

                x_plus = np.clip(x_plus, config.clip_min, config.clip_max)
                x_minus = np.clip(x_minus, config.clip_min, config.clip_max)

                p_plus = predict_proba(x_plus[None, :])[0]
                p_minus = predict_proba(x_minus[None, :])[0]

                # untargeted: increase loss on true class
                grad[i, j] = (p_plus[y[i]] - p_minus[y[i]]) / (2.0 * fd_epsilon)

        # gradient step (ascending to worsen true-class prob)
        if config.norm == "linf":
            step = config.step_size * np.sign(grad)
        elif config.norm == "l2":
            g_flat = grad.reshape(N, -1)
            g_norm = np.linalg.norm(g_flat, ord=2, axis=1, keepdims=True) + 1e-12
            step_flat = config.step_size * g_flat / g_norm
            step = step_flat.reshape_as(grad)
        else:
            raise ValueError(f"Unsupported norm '{config.norm}' for ZOO.")

        x_adv = x_adv + step

        # Lp projection
        # (use small torch helper via project_lp_ball)
        import torch
        x_t = torch.from_numpy(x_adv.astype(np.float32))
        x0_t = torch.from_numpy(X.astype(np.float32))
        x_t = project_lp_ball(x_t, x0_t, eps=config.eps, norm=config.norm)
        x_t = apply_feature_constraints(x_t, x0_t, constraints)
        x_adv = x_t.cpu().numpy()

    probs_adv = predict_proba(x_adv)
    y_pred_adv = np.argmax(probs_adv, axis=1)

    y_true = y.copy()
    success = (y_pred_clean == y_true) & (y_pred_adv != y_true)
    delta = x_adv - X
    norms = {
        "linf": compute_linf_norm(delta),
        "l2": compute_l2_norm(delta),
    }
    meta: Dict[str, Any] = {
        "attack": "zoo",
        "norm": config.norm,
        "eps": config.eps,
        "step_size": config.step_size,
        "max_iters": max_iters,
        "coords_per_iter": coords_per_iter,
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
