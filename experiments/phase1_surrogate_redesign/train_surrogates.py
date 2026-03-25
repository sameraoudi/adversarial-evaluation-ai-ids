"""
Phase 1: Train Three Architecturally-Distinct Surrogate Models

Surrogates:
  1. LR-Surrogate:    Single-layer PyTorch linear (differentiable logistic regression)
  2. GBT-Distill:     XGBoost trained, then distilled into 2-hidden-layer net (512->256, GELU)
  3. CNN-v2:          3-conv-layer CNN (16/32/64 filters, kernel=5, LeakyReLU, BatchNorm, GlobalAvgPool)

All are architecturally distinct from ALL 4 target models (RF, LR-sklearn, MLP, CNN1D).
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_npz
from src.utils.paths import get_splits_dir, get_models_root
from src.utils.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results" / "train_surrogates.log"),
    ]
)
logger = logging.getLogger("phase1.train_surrogates")


# ---------------------------------------------------------------------------
# Surrogate 1: Differentiable Logistic Regression (single linear layer)
# ---------------------------------------------------------------------------

class LinearSurrogate(nn.Module):
    """
    Single linear layer: Linear(input_dim, num_classes).
    Architecturally distinct from all targets:
      - Target RF: ensemble of decision trees
      - Target LR: sklearn (non-differentiable)
      - Target MLP: multiple hidden layers (256->128, ReLU)
      - Target CNN1D: convolutional layers
    Used ONLY to provide gradients for FGSM/PGD; not used as an IDS target.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Surrogate 2: GBT-Distilled Network (512->256, GELU)
# ---------------------------------------------------------------------------

class GBTDistillNet(nn.Module):
    """
    2-hidden-layer network (512->256) with GELU activations.
    Trained via knowledge distillation from XGBoost (GBT).
    Architecturally distinct from target MLP (256->128, ReLU, standard cross-entropy training)
    and from target CNN1D (convolutional architecture).
    The distillation training objective captures GBT's tree-like decision boundaries.
    """
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Surrogate 3: CNN-v2 (3 conv layers, kernel=5, LeakyReLU, BatchNorm, GlobalAvgPool)
# ---------------------------------------------------------------------------

class CNNv2Surrogate(nn.Module):
    """
    3-conv-layer 1D CNN:
      Conv1d(1, 16, 5, same) -> BN -> LeakyReLU -> Dropout
      Conv1d(16, 32, 5, same) -> BN -> LeakyReLU -> Dropout
      Conv1d(32, 64, 5, same) -> BN -> LeakyReLU -> Dropout
      GlobalAvgPool -> Linear(64, fc_dim) -> Linear(fc_dim, num_classes)

    Architecturally distinct from target CNN1D:
      Target CNN1D: 2 conv layers (32, 64 channels), kernel=3, ReLU, BatchNorm, AdaptiveAvgPool -> FC(64)
      This:         3 conv layers (16, 32, 64 channels), kernel=5, LeakyReLU, BatchNorm, GlobalAvgPool -> FC(64)
    Key differences: depth (3 vs 2), channels (16/32/64 vs 32/64), kernel (5 vs 3), activation (LeakyReLU vs ReLU)
    """
    def __init__(self, input_dim: int, num_classes: int,
                 channels: List[int] = None, kernel_size: int = 5,
                 fc_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]
        conv_layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding="same"),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], fc_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) -> (B, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_splits(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splits_dir = get_splits_dir(dataset_name)
    train = load_npz(splits_dir / "train.npz")
    val = load_npz(splits_dir / "val.npz")
    test = load_npz(splits_dir / "test.npz")
    return (train["X"], train["y"], val["X"], val["y"], test["X"], test["y"])


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int, device: torch.device):
    X_tr = torch.from_numpy(X_train.astype(np.float32))
    y_tr = torch.from_numpy(y_train.astype(np.int64))
    X_v = torch.from_numpy(X_val.astype(np.float32))
    y_v = torch.from_numpy(y_val.astype(np.int64))
    pin = device.type == "cuda"
    workers = 0
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                              shuffle=True, pin_memory=pin, num_workers=workers)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size,
                            shuffle=False, pin_memory=pin, num_workers=workers)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Generic PyTorch training loop (cross-entropy, early stopping)
# ---------------------------------------------------------------------------

def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    model_name: str,
    save_path: Path,
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    pat_counter = 0

    logger.info("[%s] Starting training for up to %d epochs", model_name, max_epochs)

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        n_total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * X.size(0)
            n_total += X.size(0)

        train_loss = ep_loss / max(1, n_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += X.size(0)
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        logger.info("[%s][Epoch %d] train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    model_name, epoch, train_loss, val_loss, val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_counter = 0
        else:
            pat_counter += 1
            if pat_counter >= patience:
                logger.info("[%s] Early stopping at epoch %d", model_name, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(best_state or model.state_dict(), save_path)
    logger.info("[%s] Saved checkpoint to %s (best_val_loss=%.4f)", model_name, save_path, best_val_loss)
    return {"best_val_loss": best_val_loss, "checkpoint": str(save_path)}


# ---------------------------------------------------------------------------
# Distillation training loop (soft labels from XGBoost)
# ---------------------------------------------------------------------------

def train_distill_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train_soft: np.ndarray,   # soft labels from XGBoost: (N, C) float
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    temperature: float,
    batch_size: int,
    save_path: Path,
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_soft_t = torch.from_numpy(y_train_soft.astype(np.float32))
    train_ds = TensorDataset(X_t, y_soft_t)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    ce = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    pat_counter = 0

    logger.info("[GBT-Distill] Starting distillation training (T=%.1f) for up to %d epochs",
                temperature, max_epochs)

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        n_total = 0
        for X, y_soft in train_loader:
            X, y_soft = X.to(device), y_soft.to(device)
            optimizer.zero_grad()
            logits = model(X)
            # KL divergence against soft labels (teacher)
            soft_targets = F.softmax(y_soft / temperature, dim=1)
            log_probs = F.log_softmax(logits / temperature, dim=1)
            loss = -(soft_targets * log_probs).sum(dim=1).mean() * (temperature ** 2)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * X.size(0)
            n_total += X.size(0)

        train_loss = ep_loss / max(1, n_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += ce(logits, y).item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += X.size(0)
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        logger.info("[GBT-Distill][Epoch %d] train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch, train_loss, val_loss, val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_counter = 0
        else:
            pat_counter += 1
            if pat_counter >= patience:
                logger.info("[GBT-Distill] Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(best_state or model.state_dict(), save_path)
    logger.info("[GBT-Distill] Saved checkpoint to %s (best_val_loss=%.4f)", save_path, best_val_loss)
    return {"best_val_loss": best_val_loss, "checkpoint": str(save_path)}


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    seed = cfg["seed"]
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    surrogate_dir = results_dir / "surrogate_checkpoints"
    surrogate_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("Loading data splits for dataset=%s", dataset_name)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(dataset_name)
    input_dim = X_train.shape[1]
    num_classes = int(y_train.max()) + 1
    logger.info("Data: train=%d, val=%d, test=%d, features=%d, classes=%d",
                len(X_train), len(X_val), len(X_test), input_dim, num_classes)

    results: Dict[str, Dict] = {}
    # Build a shared val_loader once (used by GBT-Distill when LR is skipped)
    _shared_val_loader = None
    def get_val_loader(batch_size: int = 2048):
        nonlocal _shared_val_loader
        if _shared_val_loader is None:
            _, _shared_val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size, device)
        return _shared_val_loader

    # ----------------------------------------------------------------
    # Surrogate 1: LR-Surrogate (differentiable linear model)
    # ----------------------------------------------------------------
    lr_ckpt = surrogate_dir / "lr_surrogate_best.pt"
    if lr_ckpt.exists():
        logger.info("[LR-Surrogate] SKIPPED — checkpoint exists at %s", lr_ckpt)
        results["lr_surrogate"] = {"checkpoint": str(lr_ckpt), "skipped": True}
    else:
        logger.info("=" * 60)
        logger.info("Training Surrogate 1: LR-Surrogate")
        logger.info("Architecture: Linear(%d, %d) — single layer, no hidden units", input_dim, num_classes)
        lr_cfg = cfg["surrogates"]["lr_surrogate"]
        lr_model = LinearSurrogate(input_dim, num_classes)
        train_loader, _shared_val_loader = make_loaders(X_train, y_train, X_val, y_val,
                                                        int(lr_cfg["batch_size"]), device)
        lr_result = train_torch_model(
            lr_model, train_loader, get_val_loader(), device,
            lr=float(lr_cfg["lr"]), weight_decay=float(lr_cfg["weight_decay"]),
            max_epochs=int(lr_cfg["max_epochs"]), patience=int(lr_cfg["early_stopping_patience"]),
            model_name="LR-Surrogate",
            save_path=lr_ckpt,
        )
        results["lr_surrogate"] = lr_result
        logger.info("LR-Surrogate done: %s", lr_result)

    # ----------------------------------------------------------------
    # Surrogate 2: GBT-Distill (XGBoost -> distilled net)
    # ----------------------------------------------------------------
    gbt_ckpt = surrogate_dir / "gbt_distill_best.pt"
    hgbt_path = surrogate_dir / "hgbt_teacher.pkl"
    if gbt_ckpt.exists():
        logger.info("[GBT-Distill] SKIPPED — checkpoint exists at %s", gbt_ckpt)
        results["gbt_distill_surrogate"] = {"checkpoint": str(gbt_ckpt), "skipped": True}
    else:
        logger.info("=" * 60)
        logger.info("Training Surrogate 2: GBT-Distill")
        gbt_cfg = cfg["surrogates"]["gbt_distill_surrogate"]

        # Step 2a: Train HistGradientBoostingClassifier (CPU-only, no CUDA dependency)
        if hgbt_path.exists():
            logger.info("[GBT-Distill] Loading saved HGBT teacher from %s", hgbt_path)
            with hgbt_path.open("rb") as f_pkl:
                hgbt_model = pickle.load(f_pkl)
            hgbt_val_pred = hgbt_model.predict(X_val)
            hgbt_val_acc = float(np.mean(hgbt_val_pred == y_val))
            logger.info("[GBT-Distill] Loaded HGBT val accuracy: %.4f", hgbt_val_acc)
        else:
            logger.info("[GBT-Distill] Step 1: Training HistGradientBoostingClassifier (CPU-only)")
            hgbt_model = HistGradientBoostingClassifier(
                max_iter=int(gbt_cfg["n_estimators"]),
                max_depth=int(gbt_cfg["max_depth"]),
                learning_rate=float(gbt_cfg["learning_rate"]),
                random_state=seed,
                verbose=1,
            )
            hgbt_model.fit(X_train, y_train)
            hgbt_val_pred = hgbt_model.predict(X_val)
            hgbt_val_acc = float(np.mean(hgbt_val_pred == y_val))
            logger.info("[GBT-Distill] HGBT val accuracy: %.4f", hgbt_val_acc)
            with hgbt_path.open("wb") as f_pkl:
                pickle.dump(hgbt_model, f_pkl)
            logger.info("[GBT-Distill] Saved HGBT to %s", hgbt_path)

        # Step 2b: Soft labels
        logger.info("[GBT-Distill] Step 2: Generating soft labels from HGBT teacher")
        chunk_size = 50000
        soft_label_chunks = []
        for start in tqdm(range(0, len(X_train), chunk_size), desc="HGBT soft labels"):
            proba = hgbt_model.predict_proba(X_train[start:start + chunk_size])
            log_odds = np.log(np.clip(proba, 1e-7, 1 - 1e-7))
            soft_label_chunks.append(log_odds.astype(np.float32))
        y_train_soft = np.vstack(soft_label_chunks)
        logger.info("[GBT-Distill] Soft labels shape: %s", y_train_soft.shape)

        # Step 2c: Distillation net
        logger.info("[GBT-Distill] Step 3: Training distillation network (512->256, GELU)")
        gbt_net = GBTDistillNet(
            input_dim, num_classes,
            hidden_dims=gbt_cfg["hidden_dims"],
            dropout=gbt_cfg["dropout"],
        )
        gbt_result = train_distill_model(
            model=gbt_net,
            X_train=X_train,
            y_train_soft=y_train_soft,
            val_loader=get_val_loader(int(gbt_cfg["distill_batch_size"])),
            device=device,
            lr=float(gbt_cfg["distill_lr"]),
            weight_decay=float(gbt_cfg["distill_weight_decay"]),
            max_epochs=int(gbt_cfg["distill_max_epochs"]),
            patience=int(gbt_cfg["early_stopping_patience"]),
            temperature=float(gbt_cfg["temperature"]),
            batch_size=int(gbt_cfg["distill_batch_size"]),
            save_path=gbt_ckpt,
        )
        results["gbt_distill_surrogate"] = {**gbt_result, "hgbt_val_acc": hgbt_val_acc}
        logger.info("GBT-Distill done: %s", gbt_result)

    # ----------------------------------------------------------------
    # Surrogate 3: CNN-v2
    # ----------------------------------------------------------------
    cnn_ckpt = surrogate_dir / "cnn_v2_best.pt"
    if cnn_ckpt.exists():
        logger.info("[CNN-v2] SKIPPED — checkpoint exists at %s", cnn_ckpt)
        results["cnn_v2_surrogate"] = {"checkpoint": str(cnn_ckpt), "skipped": True}
    else:
        logger.info("=" * 60)
        logger.info("Training Surrogate 3: CNN-v2")
        logger.info("Architecture: 3 conv layers (16/32/64), kernel=5, LeakyReLU, BatchNorm, GlobalAvgPool->FC(64)")
        cnn_cfg = cfg["surrogates"]["cnn_v2_surrogate"]
        cnn_model = CNNv2Surrogate(
            input_dim, num_classes,
            channels=cnn_cfg["channels"],
            kernel_size=cnn_cfg["kernel_size"],
            fc_dim=cnn_cfg["fc_dim"],
            dropout=cnn_cfg["dropout"],
        )
        train_loader_cnn, val_loader_cnn = make_loaders(X_train, y_train, X_val, y_val,
                                                         int(cnn_cfg["batch_size"]), device)
        cnn_result = train_torch_model(
            cnn_model, train_loader_cnn, val_loader_cnn, device,
            lr=float(cnn_cfg["lr"]), weight_decay=float(cnn_cfg["weight_decay"]),
            max_epochs=int(cnn_cfg["max_epochs"]), patience=int(cnn_cfg["early_stopping_patience"]),
            model_name="CNN-v2-Surrogate",
            save_path=cnn_ckpt,
        )
        results["cnn_v2_surrogate"] = cnn_result
        logger.info("CNN-v2 done: %s", cnn_result)

    # Save summary
    import json
    summary_path = results_dir / "surrogate_training_summary.json"
    with summary_path.open("w") as f_json:
        json.dump(results, f_json, indent=2)
    logger.info("=" * 60)
    logger.info("All surrogates trained. Summary saved to %s", summary_path)
    logger.info("Results: %s", results)
    return results


if __name__ == "__main__":
    main()
