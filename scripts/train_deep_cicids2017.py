"""
===============================================================================
Script Name   : train_deep_cicids2017.py
Description   : Orchestrates the training of Deep Learning models for the IDS evaluation.
                This script handles three distinct training phases based on 'configs/training/deep.yaml':
                1. Multilayer Perceptron (MLP): A fully connected deep network target.
                2. 1D-CNN (CNN1D): A convolutional network target for local feature extraction.
                3. Surrogate Model: A separate shadow model (typically MLP) used to generate 
                   transfer-based adversarial examples (FGSM/PGD) without accessing the target 
                   model's gradients.

Usage:
    Run from the project root:
    $ python scripts/train_deep_cicids2017.py

Inputs:
    - configs/training/deep.yaml (Hyperparameters for all deep models)
    - data/processed/cicids2017/splits/train.npz
    - data/processed/cicids2017/splits/val.npz

Outputs:
    - models/cicids2017/deep/mlp/deep_mlp_cnn_best.pt
    - models/cicids2017/deep/cnn1d/deep_mlp_cnn_best.pt
    - models/cicids2017/surrogate/surrogate_mlp_best.pt
    - logs/cicids2017/deep_mlp_cnn.log

Dependencies:
    - src.models.deep_mlp
    - src.models.deep_cnn1d
    - src.models.surrogate
    - src.utils (logging, seeding)
    - pyyaml
    - torch

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

from src.models.deep_mlp import MLPConfig, train_mlp
from src.models.deep_cnn1d import CNN1DConfig, train_cnn1d
from src.models.surrogate import SurrogateConfig, train_surrogate
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_global_seed


def load_deep_config() -> dict:
    """
    Load the deep learning training configuration from YAML.
    
    This reads 'configs/training/deep.yaml' to retrieve hyperparameters
    (learning rate, batch size, architecture details) for MLP, CNN, and Surrogate.
    
    Returns:
        dict: A dictionary containing 'mlp', 'cnn1d', and 'surrogate' config blocks.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "training" / "deep.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Deep training config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


if __name__ == "__main__":
    # 1. Load Configuration
    cfg = load_deep_config()

    dataset_name = cfg.get("dataset_name", "cicids2017")
    run_name = cfg.get("run_name", "deep_mlp_cnn")
    seed = int(cfg.get("seed", 42))

    # 2. Setup Environment
    # Configure logging to file and console; set global random seeds for reproducibility
    configure_logging(dataset_name, run_name=run_name)
    set_global_seed(seed)

    # ------------------------------------------------------------------
    # 3. Train Target MLP (Multilayer Perceptron)
    # ------------------------------------------------------------------
    mlp_cfg_raw = cfg.get("mlp", {}) or {}
    
    # Check if MLP training is enabled in the config
    if mlp_cfg_raw.get("enabled", True):
        # input_dim and num_classes are set to -1 because they will be
        # inferred dynamically from the dataset shape inside 'train_mlp'
        mlp_cfg = MLPConfig(
            input_dim=-1,  
            num_classes=-1, 
            hidden_dims=list(mlp_cfg_raw.get("hidden_dims", [256, 128])),
            dropout=float(mlp_cfg_raw.get("dropout", 0.3)),
            batch_size=int(mlp_cfg_raw.get("batch_size", 1024)),
            max_epochs=int(mlp_cfg_raw.get("max_epochs", 30)),
            lr=float(mlp_cfg_raw.get("lr", 1e-3)),
            weight_decay=float(mlp_cfg_raw.get("weight_decay", 1e-4)),
            early_stopping_patience=int(mlp_cfg_raw.get("early_stopping_patience", 5)),
            seed=seed,
        )
        train_mlp(dataset_name=dataset_name, run_name=run_name, cfg=mlp_cfg)

    # ------------------------------------------------------------------
    # 4. Train Target CNN-1D (Convolutional Neural Network)
    # ------------------------------------------------------------------
    cnn_cfg_raw = cfg.get("cnn1d", {}) or {}
    
    if cnn_cfg_raw.get("enabled", True):
        # Similar dynamic inference for input dimensions
        cnn_cfg = CNN1DConfig(
            input_dim=-1,
            num_classes=-1,
            channels=list(cnn_cfg_raw.get("channels", [32, 64])),
            kernel_sizes=list(cnn_cfg_raw.get("kernel_sizes", [3, 3])),
            dropout=float(cnn_cfg_raw.get("dropout", 0.3)),
            batch_size=int(cnn_cfg_raw.get("batch_size", 1024)),
            max_epochs=int(cnn_cfg_raw.get("max_epochs", 30)),
            lr=float(cnn_cfg_raw.get("lr", 1e-3)),
            weight_decay=float(cnn_cfg_raw.get("weight_decay", 1e-4)),
            early_stopping_patience=int(cnn_cfg_raw.get("early_stopping_patience", 5)),
            seed=seed,
        )
        train_cnn1d(dataset_name=dataset_name, run_name=run_name, cfg=cnn_cfg)

    # ------------------------------------------------------------------
    # 5. Train Surrogate Model (Shadow Model for Black-Box Attacks)
    # ------------------------------------------------------------------
    # This model is critical for generating transfer-based attacks (FGSM/PGD).
    # The attacker calculates gradients on this model and applies them to the targets.
    surrogate_raw = cfg.get("surrogate", {}) or {}
    
    surrogate_cfg = SurrogateConfig(
        base=str(surrogate_raw.get("base", "mlp")).lower(),
        max_epochs=int(surrogate_raw.get("max_epochs", 30)),
        lr=float(surrogate_raw.get("lr", 1e-3)),
        weight_decay=float(surrogate_raw.get("weight_decay", 1e-4)),
        early_stopping_patience=int(surrogate_raw.get("early_stopping_patience", 5)),
        dropout=float(surrogate_raw.get("dropout", 0.3)) if "dropout" in surrogate_raw else 0.3,
        hidden_dims=tuple(surrogate_raw.get("hidden_dims", [256, 128])),
        batch_size=int(surrogate_raw.get("batch_size", 1024)),
        seed=seed,
    )

    train_surrogate(dataset_name=dataset_name, run_name=run_name, cfg=surrogate_cfg)
