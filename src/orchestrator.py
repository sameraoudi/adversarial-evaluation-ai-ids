"""
===============================================================================
Script Name   : orchestrator.py
Description   : The central engine for running the adversarial evaluation suite.
                It maps the configuration from the YAML files to the specific attack 
                implementations. It handles:
                1. Loading the correct test data subset.
                2. Loading all target models (RF, LR, MLP, CNN).
                3. Iterating through attack types (FGSM, PGD, ZOO, HSJA).
                4. Saving the results to CSV files in `outputs/`.

Usage:
    Imported and executed by scripts/run_attacks_cicids2017.py.

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

# src/attacks/orchestrator.py
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch

from .base import AttackConfig
from .fgsm import fgsm_attack
from .pgd import pgd_attack
from .zoo import zoo_attack
from .hsja import hsja_attack
from .utils import FeatureConstraints
from .transfer import evaluate_transfer_with_clean
from ..data.io import load_npz
from ..models.wrappers import (
    load_surrogate_mlp_wrapper,
    load_rf_wrapper,
    load_lr_wrapper,
    load_mlp_wrapper,
    load_cnn1d_wrapper,
)
from ..utils.paths import get_splits_dir, get_outputs_root

logger = logging.getLogger(__name__)


@dataclass
class AttackJob:
    """
    Configuration for a single family of attacks.

    attack_type :
        One of {"fgsm", "pgd", "zoo", "hsja"}.
        - "fgsm" / "pgd" use the surrogate MLP (white-box) and evaluate transfer
          to RF, LR, MLP, and CNN1D.
        - "zoo" performs a score-based black-box attack directly on RF and LR.
        - "hsja" performs a decision-based black-box attack directly on RF, LR,
          MLP, and CNN1D.
    norm :
        "linf" or "l2".
    eps_list :
        List of epsilon values (attack budgets) to evaluate.
    num_steps :
        For PGD: number of iterations.
        For ZOO/HSJA: used as max_iters.
    step_size :
        For PGD: gradient step size.
        For HSJA: random perturbation step size.
    max_samples :
        Maximum number of test samples to include in the attack subset.
    """
    attack_type: str           # "fgsm" | "pgd" | "zoo" | "hsja"
    norm: str                  # "linf" or "l2"
    eps_list: List[float]
    num_steps: int = 40
    step_size: float = 0.01
    max_samples: int = 20000


def _load_test_subset(
    dataset_name: str,
    max_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the test split and optionally subsample to max_samples.
    """
    splits_dir = get_splits_dir(dataset_name)
    test = load_npz(splits_dir / "test.npz")
    X_test, y_test = test["X"], test["y"]

    n = X_test.shape[0]
    if max_samples is not None and max_samples < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
        logger.info(
            "Subsampled test set from %d to %d samples for attacks",
            n,
            X_test.shape[0],
        )
    else:
        logger.info("Using full test set of %d samples for attacks", n)

    return X_test, y_test


def run_attack_suite(
    dataset_name: str,
    job: AttackJob,
    run_name: str,
    seed: int = 42,
) -> Path:
    """
    Run a suite of attacks specified by `job` on the CICIDS2017 test subset.

    For "fgsm" and "pgd", this runs a white-box attack on the surrogate MLP and
    evaluates transferability to RF, LR, MLP, and CNN1D.

    For "zoo", this runs a score-based black-box attack directly against RF and LR.

    For "hsja", this runs a decision-based black-box attack directly against RF,
    LR, MLP, and CNN1D.

    Results are written to a CSV file and the path is returned.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) Load data subset
    X_clean, y = _load_test_subset(
        dataset_name=dataset_name,
        max_samples=job.max_samples,
        seed=seed,
    )

    # 2) Load models
    surrogate = load_surrogate_mlp_wrapper(dataset_name)
    rf = load_rf_wrapper(dataset_name)
    lr = load_lr_wrapper(dataset_name)
    mlp = load_mlp_wrapper(dataset_name)
    cnn1d = load_cnn1d_wrapper(dataset_name)

    # Predict-only callables for transfer and HSJA
    target_models_pred = {
        "rf": rf.predict,
        "lr": lr.predict,
        "mlp": mlp.predict,
        "cnn1d": cnn1d.predict,
    }

    # Score-based callables for ZOO (requires predict_proba)
    target_models_proba = {
        "rf": lambda X: rf.model.predict_proba(X),
        "lr": lambda X: lr.model.predict_proba(X),
    }

    # 3) Prepare output CSV
    outputs_root = get_outputs_root(dataset_name)
    adv_root = outputs_root / "adv" / "runs" / run_name
    adv_root.mkdir(parents=True, exist_ok=True)

    csv_path = adv_root / f"{job.attack_type}_{job.norm}.csv"
    logger.info("Writing attack summary to %s", csv_path)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "attack",
                "norm",
                "eps",
                "model",
                "clean_accuracy",
                "adv_accuracy",
                "attack_success_rate",
                "num_steps",
                "step_size",
                "max_samples",
            ]
        )

        # Shared constraints (currently: all features mutable, [0,1] clipping)
        constraints = FeatureConstraints(
            clip_min=0.0,
            clip_max=1.0,
            mutable_mask=None,
        )

        # 5) Loop over epsilons
        for eps in job.eps_list:
            logger.info(
                "Running %s attack with norm=%s, eps=%.5f on %d samples",
                job.attack_type,
                job.norm,
                eps,
                X_clean.shape[0],
            )

            cfg = AttackConfig(
                norm=job.norm,
                eps=float(eps),
                step_size=job.step_size,
                num_steps=job.num_steps,
                clip_min=0.0,
                clip_max=1.0,
                batch_size=job.max_samples,  # single batch for now
            )

            attack_type = job.attack_type.lower()

            if attack_type in {"fgsm", "pgd"}:
                # White-box on surrogate + transfer evaluation
                if attack_type == "fgsm":
                    attack_result = fgsm_attack(
                        model=surrogate.model,
                        X=X_clean,
                        y=y,
                        config=cfg,
                        constraints=constraints,
                        device=surrogate.device,
                    )
                else:  # "pgd"
                    attack_result = pgd_attack(
                        model=surrogate.model,
                        X=X_clean,
                        y=y,
                        config=cfg,
                        constraints=constraints,
                        device=surrogate.device,
                        random_start=True,
                    )

                transfer_result = evaluate_transfer_with_clean(
                    X_clean=X_clean,
                    attack_result=attack_result,
                    target_models=target_models_pred,
                )

                for model_name in target_models_pred.keys():
                    writer.writerow(
                        [
                            attack_type,
                            job.norm,
                            f"{eps:.6f}",
                            model_name,
                            f"{transfer_result.clean_accuracy[model_name]:.6f}",
                            f"{transfer_result.adv_accuracy[model_name]:.6f}",
                            f"{transfer_result.asr[model_name]:.6f}",
                            job.num_steps,
                            f"{job.step_size:.6f}",
                            job.max_samples,
                        ]
                    )

            elif attack_type == "zoo":
                # Score-based black-box on RF and LR
                for model_name, proba_fn in target_models_proba.items():
                    logger.info("Running ZOO against %s", model_name)

                    attack_result = zoo_attack(
                        predict_proba=proba_fn,
                        X=X_clean,
                        y=y,
                        config=cfg,
                        constraints=constraints,
                        max_iters=job.num_steps,
                        fd_epsilon=1e-3,
                        coords_per_iter=8,
                        random_state=seed,
                    )

                    y_true = attack_result.y_true
                    y_clean_pred = attack_result.y_pred_clean
                    y_adv_pred = attack_result.y_pred_adv
                    clean_acc = float(np.mean(y_clean_pred == y_true))
                    adv_acc = float(np.mean(y_adv_pred == y_true))
                    originally_correct = (y_clean_pred == y_true)
                    asr = float(
                        np.sum(attack_result.success) / max(1, np.sum(originally_correct))
                    )

                    writer.writerow(
                        [
                            attack_type,
                            job.norm,
                            f"{eps:.6f}",
                            model_name,
                            f"{clean_acc:.6f}",
                            f"{adv_acc:.6f}",
                            f"{asr:.6f}",
                            job.num_steps,
                            f"{job.step_size:.6f}",
                            job.max_samples,
                        ]
                    )

            elif attack_type == "hsja":
                # Decision-based black-box on RF, LR, MLP, CNN1D
                for model_name, pred_fn in target_models_pred.items():
                    logger.info("Running HSJA against %s", model_name)

                    attack_result = hsja_attack(
                        predict=pred_fn,
                        X=X_clean,
                        y=y,
                        config=cfg,
                        constraints=constraints,
                        max_iters=job.num_steps,
                        num_binary_search_steps=10,
                        step_size=job.step_size,
                        random_state=seed,
                    )

                    y_true = attack_result.y_true
                    y_clean_pred = attack_result.y_pred_clean
                    y_adv_pred = attack_result.y_pred_adv
                    clean_acc = float(np.mean(y_clean_pred == y_true))
                    adv_acc = float(np.mean(y_adv_pred == y_true))
                    originally_correct = (y_clean_pred == y_true)
                    asr = float(
                        np.sum(attack_result.success) / max(1, np.sum(originally_correct))
                    )

                    writer.writerow(
                        [
                            attack_type,
                            job.norm,
                            f"{eps:.6f}",
                            model_name,
                            f"{clean_acc:.6f}",
                            f"{adv_acc:.6f}",
                            f"{asr:.6f}",
                            job.num_steps,
                            f"{job.step_size:.6f}",
                            job.max_samples,
                        ]
                    )
            else:
                raise ValueError(f"Unsupported attack type '{job.attack_type}'.")

    logger.info("Finished attack suite. Results saved to %s", csv_path)
    return csv_path


# Backwards-compatible alias for earlier code that only handled FGSM/PGD.
def run_whitebox_and_transfer_suite(
    dataset_name: str,
    job: AttackJob,
    run_name: str,
    seed: int = 42,
) -> Path:
    """
    Backwards-compatible wrapper that now delegates to run_attack_suite.
    It supports FGSM, PGD, ZOO, and HSJA using the same AttackJob structure.
    """
    return run_attack_suite(dataset_name=dataset_name, job=job, run_name=run_name, seed=seed)
