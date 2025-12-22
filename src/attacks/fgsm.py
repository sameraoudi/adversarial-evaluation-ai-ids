"""
===============================================================================
Script Name   : fgsm.py
Description   : Implementation of the Fast Gradient Sign Method (FGSM).
                Type: White-Box Gradient Attack (Single Step)
                Paper Reference: Goodfellow et al. (2014)
                
                This function computes the gradient of the loss w.r.t the input and takes 
                a single step of size epsilon in the direction of the sign of the gradient.
                Used here to attack the Surrogate MLP.

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

# src/attacks/fgsm.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import AttackConfig, AttackResult
from .utils import (
    FeatureConstraints,
    project_lp_ball,
    apply_feature_constraints,
    compute_linf_norm,
    compute_l2_norm,
    to_numpy,
)


def fgsm_attack(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: AttackConfig,
    constraints: FeatureConstraints | None = None,
    device: torch.device | str = "cpu",
) -> AttackResult:
    """
    Generate adversarial examples using FGSM on a differentiable surrogate model.

    Parameters
    ----------
    model : torch.nn.Module
        Surrogate model used to compute gradients. Should output logits.
    X : np.ndarray
        Clean inputs, shape (N, D), already scaled (e.g., [0, 1]).
    y : np.ndarray
        True labels, shape (N,).
    config : AttackConfig
        Attack configuration (eps, norm, etc.). For FGSM, eps and norm are used.
    constraints : FeatureConstraints, optional
        Feature-level constraints. If None, defaults to [0,1] and all mutable.
    device : torch.device or str
        Device on which to perform computations.

    Returns
    -------
    AttackResult
        Container with adversarial examples, predictions, success mask, and norms.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if constraints is None:
        constraints = FeatureConstraints(
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            mutable_mask=None,
        )

    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y.astype(np.int64)).to(device)

    x_adv = X_tensor.clone().detach()

    x_adv.requires_grad = True
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y_tensor)
    model.zero_grad()
    loss.backward()

    grad = x_adv.grad.detach()

    if config.norm == "linf":
        # FGSM step in sign of gradient
        step = config.eps * torch.sign(grad)
        x_adv = X_tensor + step
    elif config.norm == "l2":
        # Normalize gradient and scale by eps
        g_flat = grad.view(grad.size(0), -1)
        g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-12
        step_flat = config.eps * g_flat / g_norm
        step = step_flat.view_as(grad)
        x_adv = X_tensor + step
    else:
        raise ValueError(f"Unsupported norm '{config.norm}' for FGSM.")

    # Project to Lp ball just to be safe (should already be within eps)
    x_adv = project_lp_ball(x_adv, X_tensor, eps=config.eps, norm=config.norm)

    # Apply feature-level constraints
    x_adv = apply_feature_constraints(x_adv, X_tensor, constraints)

    # Predictions
    with torch.no_grad():
        logits_clean = model(X_tensor)
        logits_adv = model(x_adv)
        y_pred_clean = torch.argmax(logits_clean, dim=1)
        y_pred_adv = torch.argmax(logits_adv, dim=1)

    y_true_np = y.copy()
    y_pred_clean_np = to_numpy(y_pred_clean)
    y_pred_adv_np = to_numpy(y_pred_adv)
    x_adv_np = to_numpy(x_adv)

    success = (y_pred_clean_np == y_true_np) & (y_pred_adv_np != y_true_np)
    delta = x_adv_np - X
    norms = {
        "linf": compute_linf_norm(delta),
        "l2": compute_l2_norm(delta),
    }

    meta: Dict[str, Any] = {
        "attack": "fgsm",
        "norm": config.norm,
        "eps": config.eps,
    }

    return AttackResult(
        x_adv=x_adv_np,
        y_true=y_true_np,
        y_pred_clean=y_pred_clean_np,
        y_pred_adv=y_pred_adv_np,
        success=success,
        norms=norms,
        meta=meta,
    )
