"""
Phase 6: Model training utilities.

Trains RF, LR, MLP, CNN1D (targets) and LR-Surr, GBT-Distill, CNN-v2 (surrogates)
on each new dataset. Reuses src/models/ architecture classes with dataset-adaptive
input dimensions.

All deep models run on GPU (device=cuda). XGBoost (GBT teacher) runs on CPU.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger("phase6.models")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _loader(X, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=True, num_workers=0)
    return _loader(X_train, y_train, True), _loader(X_val, y_val, False), \
           _loader(X_test, y_test, False)


def _eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def _train_torch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    model_name: str,
    ckpt_path: Path,
    use_amp: bool = True,
) -> Dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_val = 0.0
    pat = 0
    curves = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                loss = F.cross_entropy(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            n += len(y)

        val_acc = _eval_loader(model, val_loader, device)
        curves.append({
            "epoch": epoch,
            "train_loss": round(total_loss / max(n, 1), 5),
            "val_acc": round(val_acc, 4),
            "epoch_time_s": round(time.time() - t0, 1),
        })
        logger.info("[%s] Epoch %d/%d  loss=%.4f  val_acc=%.4f  (%.1fs)",
                    model_name, epoch, max_epochs,
                    total_loss / max(n, 1), val_acc, time.time() - t0)

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            pat = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            pat += 1
            if pat >= patience:
                logger.info("[%s] Early stopping at epoch %d", model_name, epoch)
                break

    # Load best weights
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return {"best_val_acc": round(best_val, 4), "curves": curves}


# ---------------------------------------------------------------------------
# Target: Random Forest
# ---------------------------------------------------------------------------

def train_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    n_estimators: int, seed: int, save_path: Path,
) -> Dict:
    if save_path.exists():
        logger.info("RF checkpoint found at %s — loading", save_path)
        model = joblib.load(save_path)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return {"model": model, "test_acc": round(test_acc, 4)}

    logger.info("Training RF (n=%d, n_trees=%d) ...", len(y_train), n_estimators)
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    logger.info("RF  test_acc=%.4f  (%.0fs)", test_acc, time.time() - t0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return {"model": model, "test_acc": round(test_acc, 4)}


# ---------------------------------------------------------------------------
# Target: Logistic Regression
# ---------------------------------------------------------------------------

def train_lr(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    C: float, seed: int, save_path: Path,
) -> Dict:
    if save_path.exists():
        logger.info("LR checkpoint found at %s — loading", save_path)
        model = joblib.load(save_path)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return {"model": model, "test_acc": round(test_acc, 4)}

    logger.info("Training LR (n=%d, C=%.2f) ...", len(y_train), C)
    t0 = time.time()
    model = LogisticRegression(
        C=C, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=seed
    )
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    logger.info("LR  test_acc=%.4f  (%.0fs)", test_acc, time.time() - t0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return {"model": model, "test_acc": round(test_acc, 4)}


# ---------------------------------------------------------------------------
# Target: MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train, y_train, X_val, y_val, X_test, y_test,
    input_dim: int, num_classes: int,
    hidden_dims: List[int], dropout: float,
    lr: float, weight_decay: float, batch_size: int,
    max_epochs: int, patience: int, seed: int,
    device: torch.device, save_path: Path,
) -> Dict:
    torch.manual_seed(seed)
    if save_path.exists():
        logger.info("MLP checkpoint found — loading from %s", save_path)
        model = MLP(input_dim, num_classes, hidden_dims, dropout).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_loader = _make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size)[2]
        test_acc = _eval_loader(model, test_loader, device)
        return {"model": model, "test_acc": round(test_acc, 4)}

    model = MLP(input_dim, num_classes, hidden_dims, dropout).to(device)
    tr_loader, val_loader, test_loader = _make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    info = _train_torch(model, tr_loader, val_loader, device, lr, weight_decay,
                        max_epochs, patience, "MLP", save_path)
    test_acc = _eval_loader(model, test_loader, device)
    logger.info("MLP  test_acc=%.4f", test_acc)
    return {"model": model, "test_acc": round(test_acc, 4), **info}


# ---------------------------------------------------------------------------
# Target: CNN1D
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 channels: List[int], kernel_sizes: List[int], dropout: float = 0.3):
        super().__init__()
        convs: List[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(channels, kernel_sizes):
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding="same"),
                nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True), nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train_cnn1d(
    X_train, y_train, X_val, y_val, X_test, y_test,
    input_dim: int, num_classes: int,
    channels: List[int], kernel_sizes: List[int], dropout: float,
    lr: float, weight_decay: float, batch_size: int,
    max_epochs: int, patience: int, seed: int,
    device: torch.device, save_path: Path,
) -> Dict:
    torch.manual_seed(seed)
    if save_path.exists():
        logger.info("CNN1D checkpoint found — loading from %s", save_path)
        model = CNN1D(input_dim, num_classes, channels, kernel_sizes, dropout).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_loader = _make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size)[2]
        test_acc = _eval_loader(model, test_loader, device)
        return {"model": model, "test_acc": round(test_acc, 4)}

    model = CNN1D(input_dim, num_classes, channels, kernel_sizes, dropout).to(device)
    tr_loader, val_loader, test_loader = _make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    info = _train_torch(model, tr_loader, val_loader, device, lr, weight_decay,
                        max_epochs, patience, "CNN1D", save_path)
    test_acc = _eval_loader(model, test_loader, device)
    logger.info("CNN1D  test_acc=%.4f", test_acc)
    return {"model": model, "test_acc": round(test_acc, 4), **info}


# ---------------------------------------------------------------------------
# Surrogate 1: Linear (differentiable LR)
# ---------------------------------------------------------------------------

class LinearSurrogate(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_lr_surrogate(
    X_train, y_train, X_val, y_val,
    input_dim: int, num_classes: int,
    lr: float, batch_size: int, epochs: int, patience: int, seed: int,
    device: torch.device, save_path: Path,
) -> Dict:
    torch.manual_seed(seed)
    if save_path.exists():
        logger.info("LR-Surr checkpoint found — loading from %s", save_path)
        model = LinearSurrogate(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        return {"model": model}

    model = LinearSurrogate(input_dim, num_classes).to(device)
    X_dummy = X_train[:10]; y_dummy = y_train[:10]
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=batch_size, shuffle=False,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _train_torch(model, tr_loader, val_loader, device, lr, 0.0, epochs, patience,
                 "LR-Surr", save_path)
    return {"model": model}


# ---------------------------------------------------------------------------
# Surrogate 2: GBT-Distilled Network (512 → 256, GELU)
# ---------------------------------------------------------------------------

class GBTDistillNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gbt_distill(
    X_train, y_train, X_val, y_val,
    input_dim: int, num_classes: int,
    xgb_n_estimators: int, xgb_max_depth: int, xgb_lr: float,
    distill_lr: float, distill_batch: int, distill_epochs: int,
    distill_patience: int, temperature: float, seed: int,
    device: torch.device, save_path: Path,
) -> Dict:
    torch.manual_seed(seed)
    if save_path.exists():
        logger.info("GBT-Distill checkpoint found — loading from %s", save_path)
        model = GBTDistillNet(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        return {"model": model}

    # Step 1: Train GBT teacher (CPU)
    logger.info("Training GBT teacher (n=%d) ...", len(y_train))
    t0 = time.time()
    gbt = HistGradientBoostingClassifier(
        max_iter=xgb_n_estimators, max_depth=xgb_max_depth,
        learning_rate=xgb_lr, random_state=seed,
    )
    gbt.fit(X_train, y_train)
    teacher_acc = accuracy_score(y_val, gbt.predict(X_val))
    logger.info("GBT teacher val_acc=%.4f (%.0fs)", teacher_acc, time.time() - t0)

    # Step 2: Distillation
    soft_labels = torch.from_numpy(
        gbt.predict_proba(X_train).astype(np.float32)
    )  # (N, C)

    student = GBTDistillNet(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=distill_lr)
    best_val = 0.0
    pat = 0

    ds = TensorDataset(torch.from_numpy(X_train), soft_labels,
                       torch.from_numpy(y_train.astype(np.int64)))
    tr_loader = DataLoader(ds, batch_size=distill_batch, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=distill_batch,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, distill_epochs + 1):
        student.train()
        for X_b, soft_b, y_b in tr_loader:
            X_b = X_b.to(device); soft_b = soft_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = student(X_b)
            log_prob = F.log_softmax(logits / temperature, dim=1)
            soft_target = F.softmax(soft_b / temperature, dim=1)
            loss = F.kl_div(log_prob, soft_target, reduction="batchmean")
            loss.backward()
            optimizer.step()

        val_acc = _eval_loader(student, val_loader, device)
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            pat = 0
            torch.save(student.state_dict(), save_path)
        else:
            pat += 1
            if pat >= distill_patience:
                logger.info("GBT-Distill early stopping at epoch %d", epoch)
                break
        if epoch % 5 == 0:
            logger.info("GBT-Distill epoch %d  val_acc=%.4f", epoch, val_acc)

    if save_path.exists():
        student.load_state_dict(torch.load(save_path, map_location=device))
    logger.info("GBT-Distill best_val=%.4f", best_val)
    return {"model": student}


# ---------------------------------------------------------------------------
# Surrogate 3: CNN-v2 (3 conv layers, kernel=5, LeakyReLU, BN, GAP)
# ---------------------------------------------------------------------------

class CNNv2Surrogate(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 channels: List[int] = None):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]
        convs: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding="same"),
                nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1, inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train_cnn_v2(
    X_train, y_train, X_val, y_val,
    input_dim: int, num_classes: int,
    lr: float, batch_size: int, epochs: int, patience: int, seed: int,
    device: torch.device, save_path: Path,
) -> Dict:
    torch.manual_seed(seed)
    if save_path.exists():
        logger.info("CNN-v2 checkpoint found — loading from %s", save_path)
        model = CNNv2Surrogate(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        return {"model": model}

    model = CNNv2Surrogate(input_dim, num_classes).to(device)
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=batch_size, shuffle=False,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _train_torch(model, tr_loader, val_loader, device, lr, 0.0, epochs, patience,
                 "CNN-v2", save_path)
    return {"model": model}


# ---------------------------------------------------------------------------
# Model prediction wrappers (unified interface)
# ---------------------------------------------------------------------------

def sklearn_predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def torch_predict(model: nn.Module, X: np.ndarray, device: torch.device,
                  batch_size: int = 4096) -> np.ndarray:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = torch.from_numpy(X[i:i+batch_size]).to(device)
            preds = model(X_b).argmax(1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)
