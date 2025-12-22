"""
===============================================================================
Script Name   : train_baselines_cicids2017.py
Description   : Trains the classical machine learning baselines for the IDS evaluation:
                1. Random Forest (RF): A nonlinear ensemble baseline[cite: 129].
                2. Logistic Regression (LR): A transparent linear baseline[cite: 132].
                
                The script loads hyperparameters from 'configs/training/baselines.yaml',
                trains both models on the CICIDS2017 training split, and evaluates them
                on the validation set. Trained models and performance metrics (confusion 
                matrices, classification reports) are saved to disk.

Usage:
    Run from the project root:
    $ python scripts/train_baselines_cicids2017.py

Inputs:
    - configs/training/baselines.yaml (Hyperparameters for RF and LR)
    - data/processed/cicids2017/splits/train.npz
    - data/processed/cicids2017/splits/val.npz

Outputs:
    - models/cicids2017/baselines/rf/rf_model.pkl
    - models/cicids2017/baselines/lr/lr_model.pkl
    - logs/cicids2017/baseline_rf_lr.log
    - outputs/cicids2017/runs/baseline_rf_lr/ (Metrics & Reports)

Dependencies:
    - src.models.baselines
    - src.utils (logging, seeding)
    - scikit-learn
    - pyyaml

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

import yaml
from pathlib import Path

from src.models.baselines import (
    RFConfig,
    LRConfig,
    train_random_forest_baseline,
    train_logreg_baseline,
)
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_global_seed


def load_baselines_config() -> dict:
    """
    Load training config for baselines from configs/training/baselines.yaml.

    Expected structure (example):

        dataset_name: cicids2017
        run_name: baseline_rf_lr
        seed: 42

        rf:
          n_estimators: 200
          max_depth: null
          n_jobs: -1
          class_weight: "balanced_subsample"

        lr:
          max_iter: 1000
          C: 1.0
          solver: "lbfgs"
          n_jobs: -1
          class_weight: "balanced"
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "training" / "baselines.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def build_rf_config(cfg: dict, seed: int) -> RFConfig:
    rf_cfg = cfg.get("rf", {}) or {}

    return RFConfig(
        n_estimators=int(rf_cfg.get("n_estimators", 100)),
        max_depth=rf_cfg.get("max_depth", None),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=seed,
    )


def build_lr_config(cfg: dict, seed: int) -> LRConfig:
    lr_cfg = cfg.get("lr", {}) or {}

    return LRConfig(
        max_iter=int(lr_cfg.get("max_iter", 1000)),
        C=float(lr_cfg.get("C", 1.0)),
        solver=str(lr_cfg.get("solver", "lbfgs")),
        n_jobs=int(lr_cfg.get("n_jobs", -1)),
        random_state=seed,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Load training config
    # ------------------------------------------------------------------
    cfg = load_baselines_config()

    dataset_name = cfg.get("dataset_name", "cicids2017")
    run_name = cfg.get("run_name", "baseline_rf_lr")
    seed = int(cfg.get("seed", 42))

    # ------------------------------------------------------------------
    # 2) Configure logging & global seed
    # ------------------------------------------------------------------
    configure_logging(dataset_name, run_name=run_name)
    set_global_seed(seed)

    # ------------------------------------------------------------------
    # 3) Build model configs from YAML
    # ------------------------------------------------------------------
    rf_config = build_rf_config(cfg, seed=seed)
    lr_config = build_lr_config(cfg, seed=seed)

    # ------------------------------------------------------------------
    # 4) Train RandomForest baseline
    # ------------------------------------------------------------------
    rf_model = train_random_forest_baseline(
        dataset_name=dataset_name,
        run_name=run_name,
        rf_cfg=rf_config,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 5) Train Logistic Regression baseline
    # ------------------------------------------------------------------
    lr_model = train_logreg_baseline(
        dataset_name=dataset_name,
        run_name=run_name,
        lr_cfg=lr_config,
        seed=seed,
    )
