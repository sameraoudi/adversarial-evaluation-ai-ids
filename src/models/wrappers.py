"""
===============================================================================
Script Name   : wrappers.py
Description   : Provides unified wrappers for Scikit-Learn and PyTorch models.
                Purpose:
                The attack orchestrator (src/attacks/orchestrator.py) needs a consistent 
                interface to generate predictions, regardless of whether the underlying 
                model is a PyTorch tensor-based network or a Scikit-Learn numpy-based estimator.
                
                Classes:
                - SklearnModelWrapper: Wraps RF/LR. Exposes a .predict(X) method accepting numpy arrays.
                - TorchModelWrapper: Wraps MLP/CNN. Handles tensor conversion, device management (CPU/GPU), 
                  and .eval() mode toggling automatically.

Usage:
    Used by src/attacks/transfer.py and src/attacks/orchestrator.py to load models for evaluation.

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

# src/models/wrappers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, Tuple

import joblib
import numpy as np
import torch
import yaml

from .baselines import RFConfig, LRConfig  # not strictly needed but ok to keep
from .deep_mlp import MLP
from .deep_cnn1d import CNN1D
from ..data.io import load_npz
from ..utils.paths import get_models_root, get_splits_dir


@dataclass
class SklearnModelWrapper:
    model: Any
    name: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class TorchModelWrapper:
    model: torch.nn.Module
    name: str
    device: torch.device = torch.device("cpu")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()


def _get_input_dim_num_classes(dataset_name: str) -> Tuple[int, int]:
    """
    Inspect the training split to determine input_dim and num_classes.
    """
    splits_dir = get_splits_dir(dataset_name)
    train = load_npz(splits_dir / "train.npz")
    X_train, y_train = train["X"], train["y"]
    input_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    return input_dim, num_classes


def _load_deep_config() -> Dict[str, Any]:
    """
    Load configs/training/deep.yaml if it exists.
    Returns an empty dict if not found.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "configs" / "training" / "deep.yaml"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_rf_wrapper(dataset_name: str) -> SklearnModelWrapper:
    models_root = get_models_root(dataset_name)
    rf_path = models_root / "baselines" / "rf" / "rf_model.pkl"
    if not rf_path.is_file():
        raise FileNotFoundError(f"RandomForest model not found at {rf_path}")
    model = joblib.load(rf_path)
    return SklearnModelWrapper(model=model, name="rf")


def load_lr_wrapper(dataset_name: str) -> SklearnModelWrapper:
    models_root = get_models_root(dataset_name)
    lr_path = models_root / "baselines" / "lr" / "lr_model.pkl"
    if not lr_path.is_file():
        raise FileNotFoundError(f"LogisticRegression model not found at {lr_path}")
    model = joblib.load(lr_path)
    return SklearnModelWrapper(model=model, name="lr")


def load_mlp_wrapper(dataset_name: str, run_name: str = "deep_mlp_cnn") -> TorchModelWrapper:
    input_dim, num_classes = _get_input_dim_num_classes(dataset_name)
    deep_cfg = _load_deep_config()
    mlp_cfg = deep_cfg.get("mlp", {}) if isinstance(deep_cfg, dict) else {}

    hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
    dropout = float(mlp_cfg.get("dropout", 0.3))

    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    models_root = get_models_root(dataset_name)
    ckpt_path = models_root / "deep" / "mlp" / f"{run_name}_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return TorchModelWrapper(model=model, name="mlp", device=torch.device("cpu"))


def load_cnn1d_wrapper(dataset_name: str, run_name: str = "deep_mlp_cnn") -> TorchModelWrapper:
    input_dim, num_classes = _get_input_dim_num_classes(dataset_name)
    deep_cfg = _load_deep_config()
    cnn_cfg = deep_cfg.get("cnn1d", {}) if isinstance(deep_cfg, dict) else {}

    channels = list(cnn_cfg.get("channels", [32, 64]))
    kernel_sizes = list(cnn_cfg.get("kernel_sizes", [3, 3]))
    dropout = float(cnn_cfg.get("dropout", 0.3))

    model = CNN1D(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=channels,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    )

    models_root = get_models_root(dataset_name)
    ckpt_path = models_root / "deep" / "cnn1d" / f"{run_name}_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"CNN1D checkpoint not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return TorchModelWrapper(model=model, name="cnn1d", device=torch.device("cpu"))


def load_surrogate_mlp_wrapper(dataset_name: str) -> TorchModelWrapper:
    """
    Load the surrogate MLP used for gradient-based attacks.
    """
    input_dim, num_classes = _get_input_dim_num_classes(dataset_name)
    deep_cfg = _load_deep_config()
    surrogate_cfg = deep_cfg.get("surrogate", {}) if isinstance(deep_cfg, dict) else {}
    hidden_dims = list(surrogate_cfg.get("hidden_dims", [256, 128]))
    dropout = float(surrogate_cfg.get("dropout", 0.3))

    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    models_root = get_models_root(dataset_name)
    ckpt_path = models_root / "surrogate" / "surrogate_mlp_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Surrogate MLP checkpoint not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return TorchModelWrapper(model=model, name="surrogate_mlp", device=torch.device("cpu"))
