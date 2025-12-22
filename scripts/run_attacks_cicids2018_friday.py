"""
===============================================================================
Script Name   : run_attacks_cicids2018_friday.py
Description   : Performs Cross-Dataset Adversarial Robustness Evaluation.
                This script addresses the "Generalization under Distribution Shift" research question.
                It evaluates how well adversarial examples generated using a surrogate trained on 
                CICIDS2017 transfer to the CICIDS2018 (Friday slice) dataset.
    
                Workflow:
                1. Loads the CICIDS2018 Friday test set (Target Data).
                2. Loads IDS models (RF, LR, MLP, CNN, Surrogate) trained on CICIDS2017 (Source Models).
                3. Generates FGSM and PGD attacks using the CICIDS2017 Surrogate.
                4. Evaluates the robustness of the CICIDS2017 models against these attacks 
                   on the CICIDS2018 data.

Usage:
    Run from the project root:
    $ python scripts/run_attacks_cicids2018_friday.py

Inputs:
    - data/processed/cicids2018_friday/splits/test.npz (Target Data)
    - models/cicids2017/* (Models trained on Source Data)

Outputs:
    - outputs/cicids2018_friday/adv/runs/cross_dataset_from_cicids2017/fgsm_linf.csv
    - outputs/cicids2018_friday/adv/runs/cross_dataset_from_cicids2017/pgd_linf.csv
    - logs/cicids2018_friday/cross_dataset_from_cicids2017.log

Dependencies:
    - src.attacks (base, fgsm, pgd, transfer)
    - src.models.wrappers
    - numpy, torch

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

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.attacks.base import AttackConfig
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.transfer import evaluate_transfer_with_clean
from src.attacks.utils import FeatureConstraints
from src.data.io import load_npz
from src.models.wrappers import (
    load_surrogate_mlp_wrapper,
    load_rf_wrapper,
    load_lr_wrapper,
    load_mlp_wrapper,
    load_cnn1d_wrapper,
)
from src.utils.paths import get_splits_dir, get_outputs_dir
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _load_cicids2018_test(
    dataset_name: str = "cicids2018_friday",
    max_samples: int | None = None,
    seed: int = 42,
):
    """
    Load the CICIDS2018 Friday test split (binary labels: 0=Benign, 1=Attack)
    and optionally subsample it for faster evaluation.

    Returns
    -------
    X_test : np.ndarray, shape (N, D)
    y_test : np.ndarray, shape (N,)
    """
    splits_dir = get_splits_dir(dataset_name)
    test_path = splits_dir / "test.npz"
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    data = load_npz(test_path)
    X = data["X"]
    y = data["y"]

    n = X.shape[0]
    if max_samples is not None and max_samples < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        logger.info(
            "Subsampled CICIDS2018 test from %d to %d samples",
            n,
            X.shape[0],
        )
    else:
        logger.info("Using full CICIDS2018 test set with %d samples", n)

    return X, y


def _get_outputs_dir(
    target_dataset: str,
    run_name: str,
) -> Path:
    # Uses the shared utility to find outputs/<dataset>/adv/runs/<run_name>
    # Note: Ensure get_outputs_dir is imported correctly from src.utils.paths
    # If using the version I provided earlier, it might be named get_outputs_root.
    # We construct the specific subpath here.
    from src.utils.paths import get_outputs_root
    root = get_outputs_root(target_dataset)
    out_dir = root / "adv" / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_cross_dataset_attacks_cicids2018_friday() -> None:
    """
    Main Execution Function for Cross-Dataset Evaluation.
    """
    # Configuration
    target_dataset = "cicids2018_friday"
    source_dataset = "cicids2017"  # Models are loaded from here
    run_name = "cross_dataset_from_cicids2017"
    seed = 42

    # Subsample to 20k to match the paper's methodology for fair comparison
    max_samples = 20000

    eps_list = [0.01, 0.03, 0.05]
    norm = "linf"

    # Set up logging and seeds
    configure_logging(dataset_name=target_dataset, run_name=run_name)
    set_global_seed(seed)

    logger.info("Running cross-dataset attacks: %s (source) -> %s (target)",
                source_dataset, target_dataset)

    # 1. Load CICIDS2018 Friday test slice (Target Data)
    X_clean, y = _load_cicids2018_test(
        dataset_name=target_dataset,
        max_samples=max_samples,
        seed=seed,
    )

    # 2. Load models trained on CICIDS2017 (Source Models)
    surrogate = load_surrogate_mlp_wrapper(source_dataset)
    rf = load_rf_wrapper(source_dataset)
    lr = load_lr_wrapper(source_dataset)
    mlp = load_mlp_wrapper(source_dataset)
    cnn1d = load_cnn1d_wrapper(source_dataset)

    # Map model instances to their predict methods
    target_models_pred: Dict[str, callable] = {
        "rf": rf.predict,
        "lr": lr.predict,
        "mlp": mlp.predict,
        "cnn1d": cnn1d.predict,
    }

    # 3. Define Constraints
    # Ensure adversarial examples remain valid (0-1 range)
    constraints = FeatureConstraints(
        clip_min=0.0,
        clip_max=1.0,
        mutable_mask=None,
    )

    out_dir = _get_outputs_dir(target_dataset, run_name)

    # Helper to write CSV summaries
    def _open_csv(name: str):
        path = out_dir / name
        f = path.open("w", newline="", encoding="utf-8")
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
        return f, writer, path

    # 4. Execute FGSM Attack Transfer
    fgsm_file, fgsm_writer, fgsm_path = _open_csv("fgsm_linf.csv")
    logger.info("Writing FGSM results to %s", fgsm_path)

    for eps in eps_list:
        logger.info("FGSM: norm=%s, eps=%.5f on %d CICIDS2018 samples", norm, eps, X_clean.shape[0])

        cfg = AttackConfig(
            norm=norm,
            eps=float(eps),
            step_size=float(eps),  # FGSM takes one step of size eps
            num_steps=1,
            clip_min=0.0,
            clip_max=1.0,
            batch_size=max_samples,
        )

        # Generate attack on Surrogate (Source)
        attack_result = fgsm_attack(
            model=surrogate.model,
            X=X_clean,
            y=y,
            config=cfg,
            constraints=constraints,
            device=surrogate.device,
        )

        # Evaluate transfer to Targets (Source models on Target data)
        transfer_result = evaluate_transfer_with_clean(
            X_clean=X_clean,
            attack_result=attack_result,
            target_models=target_models_pred,
        )

        for model_name in target_models_pred.keys():
            fgsm_writer.writerow(
                [
                    "fgsm",
                    norm,
                    f"{eps:.6f}",
                    model_name,
                    f"{transfer_result.clean_accuracy[model_name]:.6f}",
                    f"{transfer_result.adv_accuracy[model_name]:.6f}",
                    f"{transfer_result.asr[model_name]:.6f}",
                    1,
                    f"{cfg.step_size:.6f}",
                    X_clean.shape[0],
                ]
            )

    fgsm_file.close()
    logger.info("Finished FGSM cross-dataset attacks; results saved to %s", fgsm_path)

    # 5. Execute PGD Attack Transfer
    pgd_file, pgd_writer, pgd_path = _open_csv("pgd_linf.csv")
    logger.info("Writing PGD results to %s", pgd_path)

    for eps in eps_list:
        logger.info("PGD: norm=%s, eps=%.5f on %d CICIDS2018 samples", norm, eps, X_clean.shape[0])

        num_steps = 20
        step_size = eps / 5.0

        cfg = AttackConfig(
            norm=norm,
            eps=float(eps),
            step_size=float(step_size),
            num_steps=num_steps,
            clip_min=0.0,
            clip_max=1.0,
            batch_size=max_samples,
        )

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
            pgd_writer.writerow(
                [
                    "pgd",
                    norm,
                    f"{eps:.6f}",
                    model_name,
                    f"{transfer_result.clean_accuracy[model_name]:.6f}",
                    f"{transfer_result.adv_accuracy[model_name]:.6f}",
                    f"{transfer_result.asr[model_name]:.6f}",
                    num_steps,
                    f"{step_size:.6f}",
                    X_clean.shape[0],
                ]
            )

    pgd_file.close()
    logger.info("Finished PGD cross-dataset attacks; results saved to %s", pgd_path)


if __name__ == "__main__":
    run_cross_dataset_attacks_cicids2018_friday()
