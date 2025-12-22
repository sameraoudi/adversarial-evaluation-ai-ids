"""
===============================================================================
Script Name   : surrogate.py
Description   : Manages the training of the Surrogate Model for Black-Box attacks.
              Concept:
              In a black-box scenario, the attacker does not have access to the target model's 
              parameters. Instead, they train a "Surrogate" (shadow) model on similar data 
              and generate adversarial examples using the surrogate's gradients.
              
              Implementation:
              - Currently wraps the MLP architecture (deep_mlp.py).
              - Saves the trained model to a dedicated 'models/surrogate/' directory to distinguish 
                it from the target classifiers.

Usage:
    Imported by scripts/train_deep_cicids2017.py.

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

# src/models/surrogate.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

from .deep_mlp import MLPConfig, train_mlp
from ..utils.paths import get_models_root

logger = logging.getLogger(__name__)


@dataclass
class SurrogateConfig:
    """
    For now, the surrogate is an MLP with its own training hyperparameters.
    """
    base: str = "mlp"  # reserved for future (cnn1d, etc.)
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    dropout: float = 0.3
    hidden_dims: tuple[int, ...] = (256, 128)
    batch_size: int = 1024
    seed: int = 42


def train_surrogate(
    dataset_name: str,
    run_name: str,
    cfg: SurrogateConfig,
) -> Dict[str, Any]:
    """
    Train a surrogate model (currently an MLP) to be used as the
    gradient-based attack source.
    """
    if cfg.base.lower() != "mlp":
        raise NotImplementedError("Currently only MLP-based surrogate is implemented.")

    logger.info(
        "Training surrogate model (MLP) for dataset='%s', run='%s'",
        dataset_name,
        run_name,
    )

    # MLPConfig's input_dim/num_classes will be filled by train_mlp
    mlp_cfg = MLPConfig(
        input_dim=-1,  # placeholder
        num_classes=-1,  # placeholder
        hidden_dims=list(cfg.hidden_dims),
        dropout=cfg.dropout,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        early_stopping_patience=cfg.early_stopping_patience,
        seed=cfg.seed,
    )

    # Use a specific run name under models/surrogate
    surrogate_run_name = f"{run_name}_surrogate"

    # Reuse train_mlp, but save checkpoint under models/<dataset>/surrogate/
    result = train_mlp(
        dataset_name=dataset_name,
        run_name=surrogate_run_name,
        cfg=mlp_cfg,
    )

    # Move checkpoint to surrogate folder for clarity
    models_root = get_models_root(dataset_name)
    src_ckpt = models_root / "deep" / "mlp" / f"{surrogate_run_name}_best.pt"
    dst_dir = models_root / "surrogate"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_ckpt = dst_dir / "surrogate_mlp_best.pt"

    if src_ckpt.is_file():
        dst_ckpt.write_bytes(src_ckpt.read_bytes())
        logger.info("Copied surrogate checkpoint to %s", dst_ckpt)
        result["checkpoint"] = str(dst_ckpt)
    else:
        logger.warning(
            "Expected surrogate checkpoint %s not found; using original path %s",
            dst_ckpt,
            result.get("checkpoint"),
        )

    return result
