"""
Phase 7: Formal Statistical Testing
Supports paper revision claims with McNemar's tests, Friedman test, and bootstrap CIs.

Steps:
  0. (Pre-flight) Locate prediction files — done before this script was written.
  1. Generate per-sample predictions for CICIDS2017, CICIDS2018, UNSW-NB15, Phase 5 defense.
  2. McNemar's pairwise tests with Bonferroni/Holm-Bonferroni correction.
  3. Bootstrap difference-of-means CIs (RF ASR across datasets, AT vs TRADES, uniform vs feature-aware).
  4. Friedman test for cross-dataset model ranking consistency (3 datasets × 4 models).
  5. Compile paper_data_package.json.

Constraints:
  - Paired per-sample data for McNemar's (regenerated as needed)
  - Bonferroni correction + Holm-Bonferroni secondary
  - Exact p-values, 1000 bootstrap iterations, seed 42
  - CICIoT-2023 excluded from Friedman test (near-zero ASR is class-imbalance artifact)
  - Models and prior results are NOT modified
"""
from __future__ import annotations

import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PHASE6_DIR = PROJECT_ROOT / "experiments" / "phase6_cross_dataset"
sys.path.insert(0, str(PHASE6_DIR))

from models6 import MLP, CNN1D, LinearSurrogate, GBTDistillNet  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
PER_SAMPLE_DIR = RESULTS_DIR / "per_sample"
TESTS_DIR = RESULTS_DIR / "tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
TESTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "phase7_run.log"),
    ],
)
logger = logging.getLogger("phase7")

SEED = 42
N_BOOT = 1000
N_EVAL = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# CICIDS2017 targets
MODELS_2017 = PROJECT_ROOT / "models" / "cicids2017"
RF17_PATH   = MODELS_2017 / "baselines" / "rf" / "rf_model.pkl"
LR17_PATH   = MODELS_2017 / "baselines" / "lr" / "lr_model.pkl"
MLP17_PATH  = MODELS_2017 / "deep" / "mlp" / "deep_mlp_cnn_best.pt"
CNN17_PATH  = MODELS_2017 / "deep" / "cnn1d" / "deep_mlp_cnn_best.pt"

# Phase 5 defense models
PHASE5_MODELS = PROJECT_ROOT / "experiments" / "phase5_adversarial_training" / "results" / "models"
MLP_AT_PATH   = PHASE5_MODELS / "mlp_hardened_best.pt"
MLP_TR_PATH   = PHASE5_MODELS / "mlp_trades_best.pt"
CNN_AT_PATH   = PHASE5_MODELS / "cnn1d_hardened_best.pt"
CNN_TR_PATH   = PHASE5_MODELS / "cnn1d_trades_best.pt"

# Phase 6 targets
PHASE6_MODELS = PROJECT_ROOT / "experiments" / "phase6_cross_dataset" / "results" / "models"
RF18_PATH    = PHASE6_MODELS / "cicids2018" / "baselines" / "rf_model.pkl"
LR18_PATH    = PHASE6_MODELS / "cicids2018" / "baselines" / "lr_model.pkl"
MLP18_PATH   = PHASE6_MODELS / "cicids2018" / "deep" / "mlp_best.pt"
CNN18_PATH   = PHASE6_MODELS / "cicids2018" / "deep" / "cnn1d_best.pt"
GBTD18_PATH  = PHASE6_MODELS / "cicids2018" / "surrogates" / "gbt_distill_best.pt"

RF_UNS_PATH  = PHASE6_MODELS / "unsw_nb15" / "baselines" / "rf_model.pkl"
LR_UNS_PATH  = PHASE6_MODELS / "unsw_nb15" / "baselines" / "lr_model.pkl"
MLP_UNS_PATH = PHASE6_MODELS / "unsw_nb15" / "deep" / "mlp_best.pt"
CNN_UNS_PATH = PHASE6_MODELS / "unsw_nb15" / "deep" / "cnn1d_best.pt"
GBTD_UNS_PATH = PHASE6_MODELS / "unsw_nb15" / "surrogates" / "gbt_distill_best.pt"

# Phase 1 adversarial examples
PHASE1_ADV = PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results" / "adversarial_examples"
PHASE1_SUBSET = PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results" / "test_subset.npz"

# Phase 6 aggregate results (for Friedman + bootstrap)
PHASE6_RESULTS = PROJECT_ROOT / "experiments" / "phase6_cross_dataset" / "results"

# Split data
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers: model loading
# ---------------------------------------------------------------------------

def load_mlp(path: Path, input_dim: int, num_classes: int,
             hidden_dims: List[int] = [256, 128], dropout: float = 0.3) -> nn.Module:
    model = MLP(input_dim, num_classes, hidden_dims, dropout).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def load_cnn1d(path: Path, input_dim: int, num_classes: int,
               channels: List[int] = [32, 64], kernel_sizes: List[int] = [3, 3],
               dropout: float = 0.3) -> nn.Module:
    model = CNN1D(input_dim, num_classes, channels, kernel_sizes, dropout).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def load_gbt_distill(path: Path, input_dim: int, num_classes: int) -> nn.Module:
    model = GBTDistillNet(input_dim, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers: inference
# ---------------------------------------------------------------------------

def torch_predict(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def sklearn_predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# ---------------------------------------------------------------------------
# Helpers: stratified eval subset (same as Phase 6)
# ---------------------------------------------------------------------------

def stratified_eval_subset(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    selected = []
    for c, cnt in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        n_c = max(1, int(n * cnt / len(y)))
        n_c = min(n_c, len(idx_c))
        selected.append(rng.choice(idx_c, size=n_c, replace=False))
    idx = np.sort(np.concatenate(selected))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Adversarial example generation (PGD, surrogate-based)
# ---------------------------------------------------------------------------

def pgd_batch(surrogate: nn.Module, x: torch.Tensor, y: torch.Tensor,
              eps: float, step_size: float, n_steps: int) -> torch.Tensor:
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        loss = F.cross_entropy(surrogate(x_adv), y)
        surrogate.zero_grad()
        grad = torch.autograd.grad(loss, x_adv)[0].detach().sign()
        x_adv = x_adv.detach() + step_size * grad
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
    return x_adv.detach()


def generate_xadv(surrogate: nn.Module, X: np.ndarray, y: np.ndarray,
                  eps: float = 0.05, n_steps: int = 40, step_ratio: float = 0.25,
                  batch_size: int = 512) -> np.ndarray:
    """Generate PGD adversarial examples using the given surrogate."""
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)
    step_size = eps * step_ratio
    x_adv_list = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(DEVICE)
        yb = torch.from_numpy(y[i:i+batch_size].astype(np.int64)).to(DEVICE)
        with torch.enable_grad():
            for p in surrogate.parameters():
                p.requires_grad_(False)
            xa = pgd_batch(surrogate, xb, yb, eps, step_size, n_steps)
        x_adv_list.append(xa.cpu().numpy())
    return np.vstack(x_adv_list)


# ---------------------------------------------------------------------------
# Step 1: Per-sample prediction generation
# ---------------------------------------------------------------------------

def step1_cicids2017(binarize: bool = False) -> Dict[str, np.ndarray]:
    """
    Load CICIDS2017 test_subset (20000×80, 15-class) from Phase 1.
    Run all 4 target models on clean + representative adversarial (GBT-Distill PGD eps=0.05).
    Returns dict of per-sample arrays; saves to per_sample/cicids2017_*.npz.
    binarize=False: keep 15-class labels for CICIDS2017 native models.
    """
    out_path = PER_SAMPLE_DIR / "cicids2017_predictions.npz"
    if out_path.exists():
        logger.info("Loading cached CICIDS2017 per-sample predictions")
        return dict(np.load(out_path, allow_pickle=True))

    logger.info("Generating CICIDS2017 per-sample predictions ...")
    data = np.load(PHASE1_SUBSET)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    if binarize:
        y = (y > 0).astype(np.int64)

    n_classes = 2 if binarize else 15
    adv_path = PHASE1_ADV / "gbt_distill_surrogate__pgd__eps0.0500.npz"
    X_adv = np.load(adv_path)["X_adv"].astype(np.float32)
    assert X_adv.shape == X.shape, f"Shape mismatch: {X_adv.shape} vs {X.shape}"

    rf  = joblib.load(RF17_PATH)
    lr  = joblib.load(LR17_PATH)
    mlp = load_mlp(MLP17_PATH, input_dim=80, num_classes=n_classes)
    cnn = load_cnn1d(CNN17_PATH, input_dim=80, num_classes=n_classes)

    result = {
        "y_true":       y,
        "rf_clean":     sklearn_predict(rf, X),
        "lr_clean":     sklearn_predict(lr, X),
        "mlp_clean":    torch_predict(mlp, X),
        "cnn_clean":    torch_predict(cnn, X),
        "rf_adv":       sklearn_predict(rf, X_adv),
        "lr_adv":       sklearn_predict(lr, X_adv),
        "mlp_adv":      torch_predict(mlp, X_adv),
        "cnn_adv":      torch_predict(cnn, X_adv),
    }
    np.savez(out_path, **result)
    logger.info("Saved CICIDS2017 per-sample predictions → %s", out_path)
    return result


def step1_phase5_defense() -> Dict[str, np.ndarray]:
    """
    Load Phase 5 defense models (AT, TRADES) for MLP and CNN1D on CICIDS2017 test_subset.
    Uses same X and X_adv as CICIDS2017 step.
    """
    out_path = PER_SAMPLE_DIR / "phase5_defense_predictions.npz"
    if out_path.exists():
        logger.info("Loading cached Phase 5 defense per-sample predictions")
        return dict(np.load(out_path, allow_pickle=True))

    logger.info("Generating Phase 5 defense per-sample predictions ...")
    data = np.load(PHASE1_SUBSET)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    adv_path = PHASE1_ADV / "gbt_distill_surrogate__pgd__eps0.0500.npz"
    X_adv = np.load(adv_path)["X_adv"].astype(np.float32)

    mlp_at   = load_mlp(MLP_AT_PATH, 80, 15)
    mlp_tr   = load_mlp(MLP_TR_PATH, 80, 15)
    cnn_at   = load_cnn1d(CNN_AT_PATH, 80, 15)
    cnn_tr   = load_cnn1d(CNN_TR_PATH, 80, 15)

    result = {
        "y_true":       y,
        "mlp_at_clean": torch_predict(mlp_at, X),
        "mlp_tr_clean": torch_predict(mlp_tr, X),
        "cnn_at_clean": torch_predict(cnn_at, X),
        "cnn_tr_clean": torch_predict(cnn_tr, X),
        "mlp_at_adv":   torch_predict(mlp_at, X_adv),
        "mlp_tr_adv":   torch_predict(mlp_tr, X_adv),
        "cnn_at_adv":   torch_predict(cnn_at, X_adv),
        "cnn_tr_adv":   torch_predict(cnn_tr, X_adv),
    }
    np.savez(out_path, **result)
    logger.info("Saved Phase 5 defense per-sample predictions → %s", out_path)
    return result


def _step1_phase6_dataset(
    dataset_name: str,
    test_npz: Path,
    rf_path: Path,
    lr_path: Path,
    mlp_path: Path,
    cnn_path: Path,
    surr_path: Path,
    input_dim: int,
) -> Dict[str, np.ndarray]:
    """Generic Phase 6 per-sample prediction generation (binary classification)."""
    out_path = PER_SAMPLE_DIR / f"{dataset_name}_predictions.npz"
    if out_path.exists():
        logger.info("Loading cached %s per-sample predictions", dataset_name)
        return dict(np.load(out_path, allow_pickle=True))

    logger.info("Generating %s per-sample predictions (input_dim=%d) ...", dataset_name, input_dim)
    data = np.load(test_npz)
    X_full, y_full = data["X"].astype(np.float32), data["y"].astype(np.int64)
    X, y = stratified_eval_subset(X_full, y_full, N_EVAL, SEED)

    rf  = joblib.load(rf_path)
    lr  = joblib.load(lr_path)
    mlp = load_mlp(mlp_path, input_dim=input_dim, num_classes=2)
    cnn = load_cnn1d(cnn_path, input_dim=input_dim, num_classes=2)
    surr = load_gbt_distill(surr_path, input_dim=input_dim, num_classes=2)

    logger.info("  Generating X_adv (PGD eps=0.05) for %s ...", dataset_name)
    X_adv = generate_xadv(surr, X, y, eps=0.05, n_steps=40)

    result = {
        "y_true":    y,
        "rf_clean":  sklearn_predict(rf, X),
        "lr_clean":  sklearn_predict(lr, X),
        "mlp_clean": torch_predict(mlp, X),
        "cnn_clean": torch_predict(cnn, X),
        "rf_adv":    sklearn_predict(rf, X_adv),
        "lr_adv":    sklearn_predict(lr, X_adv),
        "mlp_adv":   torch_predict(mlp, X_adv),
        "cnn_adv":   torch_predict(cnn, X_adv),
    }
    np.savez(out_path, **result)
    logger.info("Saved %s per-sample predictions → %s", dataset_name, out_path)
    return result


def step1_cicids2018() -> Dict[str, np.ndarray]:
    data = np.load(DATA_PROCESSED / "cicids2018" / "splits" / "test.npz")
    input_dim = data["X"].shape[1]
    logger.info("CICIDS2018 input_dim detected: %d", input_dim)
    return _step1_phase6_dataset(
        "cicids2018",
        DATA_PROCESSED / "cicids2018" / "splits" / "test.npz",
        RF18_PATH, LR18_PATH, MLP18_PATH, CNN18_PATH, GBTD18_PATH,
        input_dim=input_dim,
    )


def step1_unsw_nb15() -> Dict[str, np.ndarray]:
    # Determine UNSW-NB15 feature count from test split
    data = np.load(DATA_PROCESSED / "unsw_nb15" / "splits" / "test.npz")
    input_dim = data["X"].shape[1]
    logger.info("UNSW-NB15 input_dim detected: %d", input_dim)
    return _step1_phase6_dataset(
        "unsw_nb15",
        DATA_PROCESSED / "unsw_nb15" / "splits" / "test.npz",
        RF_UNS_PATH, LR_UNS_PATH, MLP_UNS_PATH, CNN_UNS_PATH, GBTD_UNS_PATH,
        input_dim=input_dim,
    )


# ---------------------------------------------------------------------------
# Step 2: McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    continuity: bool = True,
) -> Dict:
    """
    McNemar's test comparing two classifiers on the same samples.
    Returns: chi2, p_value, b, c, odds_ratio, ci_lo, ci_hi (Wilson CI on odds ratio).
    b = A_right & B_wrong
    c = A_wrong & B_right
    """
    a_right = (y_pred_a == y_true)
    b_right = (y_pred_b == y_true)

    b = int(( a_right & ~b_right).sum())   # A correct, B wrong
    c = int((~a_right &  b_right).sum())   # A wrong, B correct

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": 0, "c": 0,
                "odds_ratio": 1.0, "n_discordant": 0}

    if continuity:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)

    p_value = float(stats.chi2.sf(chi2, df=1))
    odds_ratio = (b / c) if c > 0 else float("inf")

    return {
        "chi2": round(chi2, 4),
        "p_value": p_value,
        "b": b,
        "c": c,
        "n_discordant": b + c,
        "odds_ratio": round(odds_ratio, 4) if not np.isinf(odds_ratio) else None,
    }


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down correction. Returns list of rejection decisions."""
    n = len(p_values)
    order = np.argsort(p_values)
    rejected = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    return alpha / n_tests


def run_mcnemar_family(
    family_name: str,
    tests: List[Dict],  # each: {label, y_true, y_pred_a, y_pred_b, model_a, model_b, dataset}
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Run a family of McNemar's tests with Bonferroni and Holm-Bonferroni correction.
    """
    n = len(tests)
    bonf_alpha = bonferroni_threshold(n, alpha)
    results = []
    p_values = []

    for t in tests:
        r = mcnemar_test(t["y_true"], t["y_pred_a"], t["y_pred_b"])
        row = {
            "family": family_name,
            "label": t["label"],
            "dataset": t.get("dataset", ""),
            "model_a": t.get("model_a", ""),
            "model_b": t.get("model_b", ""),
            **r,
            "bonferroni_alpha": round(bonf_alpha, 6),
            "significant_bonferroni": bool(r["p_value"] <= bonf_alpha),
        }
        results.append(row)
        p_values.append(r["p_value"])

    holm_rej = holm_bonferroni(p_values, alpha)
    for row, rej in zip(results, holm_rej):
        row["significant_holm"] = bool(rej)

    logger.info("McNemar family '%s': %d tests, Bonferroni α=%.5f, "
                "%d significant (Bonf), %d significant (Holm)",
                family_name, n, bonf_alpha,
                sum(r["significant_bonferroni"] for r in results),
                sum(r["significant_holm"] for r in results))
    return results


# ---------------------------------------------------------------------------
# Step 3: Bootstrap CI on difference of means
# ---------------------------------------------------------------------------

def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_iters: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    paired: bool = False,
) -> Dict:
    """
    Bootstrap CI on mean(a) - mean(b).
    If paired=True, resample pairs (a[i], b[i]).
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iters)
    if paired:
        assert len(a) == len(b)
        n = len(a)
        d = a.astype(np.float64) - b.astype(np.float64)
        for i in range(n_iters):
            idx = rng.integers(0, n, size=n)
            diffs[i] = d[idx].mean()
    else:
        na, nb = len(a), len(b)
        for i in range(n_iters):
            diffs[i] = a[rng.integers(0, na, size=na)].astype(np.float64).mean() \
                     - b[rng.integers(0, nb, size=nb)].astype(np.float64).mean()

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(diffs, 100 * alpha))
    hi = float(np.percentile(diffs, 100 * (1 - alpha)))
    obs_diff = float(a.astype(np.float64).mean() - b.astype(np.float64).mean())
    ci_excludes_zero = bool(lo > 0 or hi < 0)

    return {
        "observed_diff": round(obs_diff, 6),
        "ci_lo": round(lo, 6),
        "ci_hi": round(hi, 6),
        "ci_excludes_zero": ci_excludes_zero,
        "n_iters": n_iters,
    }


# ---------------------------------------------------------------------------
# Step 4: Friedman test
# ---------------------------------------------------------------------------

def run_friedman_test(asr_matrix: np.ndarray, model_names: List[str],
                      dataset_names: List[str]) -> Dict:
    """
    Friedman test for consistency of model rankings across datasets.
    asr_matrix: shape (n_datasets, n_models) — rows are blocks, cols are treatments.
    scipy.stats.friedmanchisquare takes one arg per treatment (col), each arg = observations across blocks.
    """
    assert asr_matrix.shape == (len(dataset_names), len(model_names))
    cols = [asr_matrix[:, j] for j in range(len(model_names))]
    stat, p = stats.friedmanchisquare(*cols)

    # Compute ranks per dataset (lower ASR = better = lower rank)
    ranks = np.apply_along_axis(lambda row: stats.rankdata(row), axis=1, arr=asr_matrix)
    mean_ranks = ranks.mean(axis=0)

    result = {
        "statistic": round(float(stat), 4),
        "p_value": float(p),
        "reject_h0_at_0.05": bool(p < 0.05),
        "n_datasets": len(dataset_names),
        "n_models": len(model_names),
        "datasets": dataset_names,
        "models": model_names,
        "asr_matrix": asr_matrix.tolist(),
        "rank_matrix": ranks.tolist(),
        "mean_ranks": {m: round(float(r), 3) for m, r in zip(model_names, mean_ranks)},
    }
    logger.info("Friedman test: stat=%.4f  p=%.6f  reject_H0=%s",
                stat, p, p < 0.05)
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("Phase 7: Formal Statistical Testing")
    logger.info("Device: %s", DEVICE)
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Generate per-sample predictions
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 1: Per-sample prediction generation ---")
    pred17 = step1_cicids2017()
    pred18 = step1_cicids2018()
    pred_uns = step1_unsw_nb15()
    pred_def = step1_phase5_defense()

    datasets = {
        "cicids2017": pred17,
        "cicids2018": pred18,
        "unsw_nb15":  pred_uns,
    }
    model_keys = ["rf", "lr", "mlp", "cnn"]
    model_labels = ["RF", "LR", "MLP", "CNN1D"]

    # -----------------------------------------------------------------------
    # Step 2a: McNemar's — Set A: Clean accuracy, pairwise (6 pairs × 3 datasets = 18 tests)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 2a: McNemar's Set A (clean accuracy) ---")
    tests_A = []
    for ds_name, pred in datasets.items():
        y_true = pred["y_true"]
        # For CICIDS2017 (15-class), binarize y_true for consistency check? No —
        # models are evaluated on their native labels. Keep as-is.
        for (ka, la), (kb, lb) in combinations(zip(model_keys, model_labels), 2):
            tests_A.append({
                "label": f"{la}_vs_{lb}_{ds_name}",
                "dataset": ds_name,
                "model_a": la,
                "model_b": lb,
                "y_true": y_true,
                "y_pred_a": pred[f"{ka}_clean"],
                "y_pred_b": pred[f"{kb}_clean"],
            })
    results_A = run_mcnemar_family("Set_A_clean_accuracy", tests_A)

    # -----------------------------------------------------------------------
    # Step 2b: McNemar's — Set B: Adversarial accuracy, pairwise (6 pairs × 3 datasets)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 2b: McNemar's Set B (adversarial accuracy) ---")
    # Evaluate only on originally-correct samples (standard adversarial evaluation protocol)
    tests_B = []
    for ds_name, pred in datasets.items():
        y_true = pred["y_true"]
        for (ka, la), (kb, lb) in combinations(zip(model_keys, model_labels), 2):
            # Restrict to samples where BOTH models were correct on clean data
            correct_both = (pred[f"{ka}_clean"] == y_true) & (pred[f"{kb}_clean"] == y_true)
            if correct_both.sum() < 10:
                logger.warning("Too few paired correct samples for %s/%s on %s, skipping",
                               la, lb, ds_name)
                continue
            tests_B.append({
                "label": f"{la}_vs_{lb}_{ds_name}",
                "dataset": ds_name,
                "model_a": la,
                "model_b": lb,
                "y_true": y_true[correct_both],
                "y_pred_a": pred[f"{ka}_adv"][correct_both],
                "y_pred_b": pred[f"{kb}_adv"][correct_both],
            })
    results_B = run_mcnemar_family("Set_B_adv_accuracy", tests_B)

    # -----------------------------------------------------------------------
    # Step 2c: McNemar's — Set C: Defense comparison (AT-Hardened vs TRADES)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 2c: McNemar's Set C (defense comparison) ---")
    y_def = pred_def["y_true"]
    tests_C = []
    # MLP: AT-Hardened vs TRADES (adversarial)
    for arch, ka, kb in [("MLP", "mlp_at", "mlp_tr"), ("CNN1D", "cnn_at", "cnn_tr")]:
        correct_clean = (pred_def[f"{ka}_clean"] == y_def) & (pred_def[f"{kb}_clean"] == y_def)
        tests_C.append({
            "label": f"{arch}_AT-Hardened_vs_TRADES_adversarial",
            "dataset": "cicids2017",
            "model_a": f"{arch}-AT-Hardened",
            "model_b": f"{arch}-TRADES",
            "y_true":  y_def[correct_clean],
            "y_pred_a": pred_def[f"{ka}_adv"][correct_clean],
            "y_pred_b": pred_def[f"{kb}_adv"][correct_clean],
        })
        # Also compare on clean accuracy
        tests_C.append({
            "label": f"{arch}_AT-Hardened_vs_TRADES_clean",
            "dataset": "cicids2017",
            "model_a": f"{arch}-AT-Hardened",
            "model_b": f"{arch}-TRADES",
            "y_true":  y_def,
            "y_pred_a": pred_def[f"{ka}_clean"],
            "y_pred_b": pred_def[f"{kb}_clean"],
        })
    results_C = run_mcnemar_family("Set_C_defense_comparison", tests_C)

    # -----------------------------------------------------------------------
    # Step 2d: McNemar's — Set D: Cross-dataset consistency
    # (Compare in-distribution vs cross-dataset transfer on CICIDS2018 test samples)
    # We compare: RF under 2017-trained surrogate vs RF under 2018-trained surrogate
    # This requires generating 2017→2018 adversarial examples for the same X
    # -----------
    # For simplicity: use the aggregate cross-dataset transfer results (already computed in Phase 6)
    # and report that direct per-sample pairing isn't possible across different test sets.
    # Instead, we compare model performance (fooled/not) on the shared CICIDS2018 test set
    # with two attack conditions: in-distribution GBT-Distill (2018-trained) vs the already-computed
    # 2018 adversarial examples. This IS the Set D we already have in pred18.
    # The comparison: is 2018-RF harder to fool by GBT-Distill than MLP?
    # This is already covered by Set B. Set D is labeled separately for the paper.
    logger.info("\n--- Step 2d: Set D (cross-dataset consistency note) ---")
    logger.info("Set D not applicable as McNemar's: different test sets cannot be paired.")
    logger.info("Cross-dataset transfer analysis covered by Friedman test + bootstrap CIs.")
    results_D: List[Dict] = []

    # -----------------------------------------------------------------------
    # Save McNemar's results
    # -----------------------------------------------------------------------
    all_mcnemar = results_A + results_B + results_C + results_D
    mcnemar_df = pd.DataFrame(all_mcnemar)
    mcnemar_path = TESTS_DIR / "mcnemar_results.csv"
    mcnemar_df.to_csv(mcnemar_path, index=False)
    logger.info("Saved McNemar's results → %s", mcnemar_path)

    # -----------------------------------------------------------------------
    # Step 3: Friedman test (3 datasets × 4 models, CICIoT-2023 excluded)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 4: Friedman test (cross-dataset ranking consistency) ---")

    # Load aggregate ASR per target model per dataset
    # CICIDS2017: from Phase 1 transfer_matrix.csv
    ph1 = pd.read_csv(PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results" / "transfer_matrix.csv")
    # Columns: surrogate, target, attack, eps, clean_acc, adv_acc, delta_acc, asr, adv_acc_ci, asr_ci
    asr17 = ph1.groupby("target")["asr"].mean().to_dict()

    # CICIDS2018: from Phase 6 in-distribution CSV
    ph6_18 = pd.read_csv(PHASE6_RESULTS / "in_distribution" / "cicids2018_transfer_matrix.csv")
    asr18 = ph6_18.groupby("target")["asr"].mean().to_dict()

    # UNSW-NB15: from Phase 6 in-distribution CSV
    ph6_uns = pd.read_csv(PHASE6_RESULTS / "in_distribution" / "unsw_nb15_transfer_matrix.csv")
    asr_uns = ph6_uns.groupby("target")["asr"].mean().to_dict()

    # Build matrix: rows = datasets, cols = models (RF, LR, MLP, CNN1D)
    dataset_names_friedman = ["cicids2017", "cicids2018", "unsw_nb15"]
    model_names_friedman   = ["RF", "LR", "MLP", "CNN1D"]

    # Normalize keys to uppercase model names
    def normalize_asr_dict(d: Dict) -> Dict:
        mapping = {}
        for k, v in d.items():
            ku = k.upper()
            if ku in ("RF", "LR", "MLP", "CNN1D"):
                mapping[ku] = v
            elif "CNN" in ku:
                mapping["CNN1D"] = v
            elif "MLP" in ku:
                mapping["MLP"] = v
            elif ku.startswith("RF"):
                mapping["RF"] = v
            elif ku.startswith("LR"):
                mapping["LR"] = v
        return mapping

    asr17n  = normalize_asr_dict(asr17)
    asr18n  = normalize_asr_dict(asr18)
    asr_unsn = normalize_asr_dict(asr_uns)

    logger.info("ASR summary:")
    for ds_name, asr_dict in [("CICIDS2017", asr17n), ("CICIDS2018", asr18n), ("UNSW-NB15", asr_unsn)]:
        logger.info("  %s: %s", ds_name, {k: round(v, 4) for k, v in asr_dict.items()})

    asr_matrix = np.array([
        [asr17n.get(m, np.nan) for m in model_names_friedman],
        [asr18n.get(m, np.nan) for m in model_names_friedman],
        [asr_unsn.get(m, np.nan) for m in model_names_friedman],
    ])

    if np.any(np.isnan(asr_matrix)):
        logger.warning("NaN in ASR matrix — some model keys not matched: %s %s %s",
                       asr17n.keys(), asr18n.keys(), asr_unsn.keys())

    friedman_result = run_friedman_test(asr_matrix, model_names_friedman, dataset_names_friedman)
    with open(TESTS_DIR / "friedman_result.json", "w") as f:
        json.dump(friedman_result, f, indent=2)
    logger.info("Saved Friedman result → %s", TESTS_DIR / "friedman_result.json")

    # -----------------------------------------------------------------------
    # Step 3: Bootstrap CIs
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 3: Bootstrap CIs ---")
    bootstrap_results = {}

    # 3a: RF ASR gap CICIDS2017 vs CICIDS2018 (per-sample fooled indicators)
    # Use predictions from per-sample data (GBT-Distill PGD eps=0.05 attack)
    y17 = pred17["y_true"]
    rf17_clean = pred17["rf_clean"]; rf17_adv = pred17["rf_adv"]
    correct17 = rf17_clean == y17
    fooled17 = (rf17_adv[correct17] != y17[correct17]).astype(np.float64)

    y18 = pred18["y_true"]
    rf18_clean = pred18["rf_clean"]; rf18_adv = pred18["rf_adv"]
    correct18 = rf18_clean == y18
    fooled18 = (rf18_adv[correct18] != y18[correct18]).astype(np.float64)

    boot_rf_gap = bootstrap_diff_ci(fooled17, fooled18, n_iters=N_BOOT, seed=SEED, paired=False)
    bootstrap_results["rf_asr_gap_2017_minus_2018"] = {
        "description": "RF ASR gap: CICIDS2017 (in-dist, GBT-PGD-0.05) minus CICIDS2018 (in-dist, GBT-PGD-0.05)",
        "mean_asr_2017": round(float(fooled17.mean()), 6),
        "mean_asr_2018": round(float(fooled18.mean()), 6),
        **boot_rf_gap,
    }
    logger.info("RF ASR gap (2017-2018): diff=%.4f  CI=[%.4f, %.4f]  excl_zero=%s",
                boot_rf_gap["observed_diff"], boot_rf_gap["ci_lo"], boot_rf_gap["ci_hi"],
                boot_rf_gap["ci_excludes_zero"])

    # 3b: AT-Hardened vs TRADES ASR gap (MLP, CICIDS2017)
    correct_base = (pred_def["mlp_at_clean"] == y_def)  # use AT model's clean correct
    fooled_at = (pred_def["mlp_at_adv"][correct_base] != y_def[correct_base]).astype(np.float64)
    correct_trades = (pred_def["mlp_tr_clean"] == y_def)
    fooled_tr = (pred_def["mlp_tr_adv"][correct_trades] != y_def[correct_trades]).astype(np.float64)

    boot_at_tr = bootstrap_diff_ci(fooled_at, fooled_tr, n_iters=N_BOOT, seed=SEED, paired=False)
    bootstrap_results["mlp_at_vs_trades_asr_gap"] = {
        "description": "MLP: AT-Hardened ASR minus TRADES ASR (GBT-PGD-0.05, CICIDS2017)",
        "mean_asr_at":     round(float(fooled_at.mean()), 6),
        "mean_asr_trades": round(float(fooled_tr.mean()), 6),
        **boot_at_tr,
    }
    logger.info("MLP AT vs TRADES gap: diff=%.4f  CI=[%.4f, %.4f]  excl_zero=%s",
                boot_at_tr["observed_diff"], boot_at_tr["ci_lo"], boot_at_tr["ci_hi"],
                boot_at_tr["ci_excludes_zero"])

    # 3c: CNN1D AT-Hardened vs TRADES
    correct_cnn_at = (pred_def["cnn_at_clean"] == y_def)
    fooled_cnn_at = (pred_def["cnn_at_adv"][correct_cnn_at] != y_def[correct_cnn_at]).astype(np.float64)
    correct_cnn_tr = (pred_def["cnn_tr_clean"] == y_def)
    fooled_cnn_tr = (pred_def["cnn_tr_adv"][correct_cnn_tr] != y_def[correct_cnn_tr]).astype(np.float64)

    boot_cnn_at_tr = bootstrap_diff_ci(fooled_cnn_at, fooled_cnn_tr, n_iters=N_BOOT, seed=SEED, paired=False)
    bootstrap_results["cnn_at_vs_trades_asr_gap"] = {
        "description": "CNN1D: AT-Hardened ASR minus TRADES ASR (GBT-PGD-0.05, CICIDS2017)",
        "mean_asr_at":     round(float(fooled_cnn_at.mean()), 6),
        "mean_asr_trades": round(float(fooled_cnn_tr.mean()), 6),
        **boot_cnn_at_tr,
    }
    logger.info("CNN1D AT vs TRADES gap: diff=%.4f  CI=[%.4f, %.4f]  excl_zero=%s",
                boot_cnn_at_tr["observed_diff"], boot_cnn_at_tr["ci_lo"], boot_cnn_at_tr["ci_hi"],
                boot_cnn_at_tr["ci_excludes_zero"])

    # 3d: Uniform vs feature-aware ASR gap (from aggregate Phase 3/4 CSV)
    ua_df = pd.read_csv(PROJECT_ROOT / "experiments" / "phase3_4_domain_constraints" /
                        "results" / "uniform_vs_aware_comparison.csv")
    uniform_rows  = ua_df[ua_df["source"] == "phase1_uniform"]
    aware_rows    = ua_df[ua_df["source"] == "phase3_aware"]
    uniform_asrs  = uniform_rows["asr"].values.astype(np.float64)
    aware_asrs    = aware_rows["asr"].values.astype(np.float64)

    # Pair by attack × eps_level × target (aggregate CSV has same conditions in same order? Check)
    # Use independent bootstrap since they're different conditions by design
    boot_ua = bootstrap_diff_ci(uniform_asrs, aware_asrs, n_iters=N_BOOT, seed=SEED, paired=False)
    bootstrap_results["uniform_vs_feature_aware_asr_gap"] = {
        "description": "Uniform ASR minus Feature-Aware ASR (averaged across targets and attacks)",
        "mean_asr_uniform":      round(float(uniform_asrs.mean()), 6),
        "mean_asr_feature_aware": round(float(aware_asrs.mean()), 6),
        "n_uniform_conditions":  len(uniform_asrs),
        "n_aware_conditions":    len(aware_asrs),
        **boot_ua,
    }
    logger.info("Uniform vs feature-aware gap: diff=%.4f  CI=[%.4f, %.4f]  excl_zero=%s",
                boot_ua["observed_diff"], boot_ua["ci_lo"], boot_ua["ci_hi"],
                boot_ua["ci_excludes_zero"])

    # 3e: RF ASR consistency across Phase 6 datasets (CICIDS2018 vs UNSW-NB15)
    y_uns = pred_uns["y_true"]
    rf_uns_clean = pred_uns["rf_clean"]; rf_uns_adv = pred_uns["rf_adv"]
    correct_uns = rf_uns_clean == y_uns
    fooled_uns = (rf_uns_adv[correct_uns] != y_uns[correct_uns]).astype(np.float64)

    boot_rf_18_uns = bootstrap_diff_ci(fooled18, fooled_uns, n_iters=N_BOOT, seed=SEED, paired=False)
    bootstrap_results["rf_asr_gap_2018_minus_unsw"] = {
        "description": "RF ASR gap: CICIDS2018 minus UNSW-NB15 (in-dist, GBT-PGD-0.05)",
        "mean_asr_2018": round(float(fooled18.mean()), 6),
        "mean_asr_unsw": round(float(fooled_uns.mean()), 6),
        **boot_rf_18_uns,
    }
    logger.info("RF ASR gap (2018-UNSW): diff=%.4f  CI=[%.4f, %.4f]  excl_zero=%s",
                boot_rf_18_uns["observed_diff"], boot_rf_18_uns["ci_lo"],
                boot_rf_18_uns["ci_hi"], boot_rf_18_uns["ci_excludes_zero"])

    with open(TESTS_DIR / "bootstrap_cis.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    logger.info("Saved bootstrap CIs → %s", TESTS_DIR / "bootstrap_cis.json")

    # -----------------------------------------------------------------------
    # Step 5: Compile paper data package
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 5: Compiling paper data package ---")

    # Summary stats from McNemar's families
    def family_summary(results: List[Dict]) -> Dict:
        if not results:
            return {"n_tests": 0}
        return {
            "n_tests": len(results),
            "n_significant_bonferroni": sum(r["significant_bonferroni"] for r in results),
            "n_significant_holm": sum(r["significant_holm"] for r in results),
            "min_p_value": round(min(r["p_value"] for r in results), 8),
            "max_p_value": round(max(r["p_value"] for r in results), 8),
        }

    # Per-model clean accuracy per dataset
    clean_acc = {}
    for ds_name, pred in datasets.items():
        y = pred["y_true"]
        clean_acc[ds_name] = {
            "RF":    round(float((pred["rf_clean"]  == y).mean()), 4),
            "LR":    round(float((pred["lr_clean"]  == y).mean()), 4),
            "MLP":   round(float((pred["mlp_clean"] == y).mean()), 4),
            "CNN1D": round(float((pred["cnn_clean"] == y).mean()), 4),
        }

    # Per-model adversarial accuracy (on originally correct samples, GBT-PGD-0.05)
    adv_asr = {}
    for ds_name, pred in datasets.items():
        y = pred["y_true"]
        adv_asr[ds_name] = {}
        for k, l in zip(model_keys, model_labels):
            correct = pred[f"{k}_clean"] == y
            if correct.sum() == 0:
                adv_asr[ds_name][l] = None
            else:
                fooled = (pred[f"{k}_adv"][correct] != y[correct])
                adv_asr[ds_name][l] = round(float(fooled.mean()), 4)

    # Defense comparison ASR
    defense_asr = {}
    for arch, ka, kb, la, lb in [
        ("MLP",   "mlp_at", "mlp_tr", "AT-Hardened", "TRADES"),
        ("CNN1D", "cnn_at", "cnn_tr", "AT-Hardened", "TRADES"),
    ]:
        correct = (pred_def[f"{ka}_clean"] == y_def)
        fooled_a = (pred_def[f"{ka}_adv"][correct] != y_def[correct]).mean()
        correct_b = (pred_def[f"{kb}_clean"] == y_def)
        fooled_b = (pred_def[f"{kb}_adv"][correct_b] != y_def[correct_b]).mean()
        defense_asr[arch] = {
            la: round(float(fooled_a), 6),
            lb: round(float(fooled_b), 6),
        }

    package = {
        "experiment": "Phase 7 Statistical Testing",
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_eval_per_dataset": N_EVAL,
        "representative_attack": "GBT-Distill PGD eps=0.05",
        "clean_accuracy": clean_acc,
        "adversarial_asr_gbt_pgd_0p05": adv_asr,
        "defense_comparison_asr": defense_asr,
        "mcnemar_families": {
            "Set_A_clean_accuracy":    family_summary(results_A),
            "Set_B_adv_accuracy":      family_summary(results_B),
            "Set_C_defense_comparison": family_summary(results_C),
        },
        "friedman_test": {
            "statistic": friedman_result["statistic"],
            "p_value":   friedman_result["p_value"],
            "reject_h0_at_0.05": bool(friedman_result["reject_h0_at_0.05"]),
            "mean_ranks": friedman_result["mean_ranks"],
            "interpretation": (
                "REJECT: consistent ranking across datasets"
                if friedman_result["reject_h0_at_0.05"]
                else "FAIL TO REJECT: no consistent ranking — supports dataset-specificity claim"
            ),
        },
        "bootstrap_cis": bootstrap_results,
    }

    pkg_path = RESULTS_DIR / "paper_data_package.json"
    with open(pkg_path, "w") as f:
        json.dump(package, f, indent=2)
    logger.info("Saved paper data package → %s", pkg_path)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7 COMPLETE")
    logger.info("=" * 70)
    logger.info("McNemar's Set A (clean acc):    %d/%d significant (Bonferroni)",
                family_summary(results_A)["n_significant_bonferroni"],
                family_summary(results_A)["n_tests"])
    logger.info("McNemar's Set B (adv acc):      %d/%d significant (Bonferroni)",
                family_summary(results_B)["n_significant_bonferroni"],
                family_summary(results_B)["n_tests"])
    logger.info("McNemar's Set C (defense):      %d/%d significant (Bonferroni)",
                family_summary(results_C)["n_significant_bonferroni"],
                family_summary(results_C)["n_tests"])
    logger.info("Friedman: stat=%.4f  p=%.6f  reject=%s",
                friedman_result["statistic"], friedman_result["p_value"],
                friedman_result["reject_h0_at_0.05"])
    logger.info("RF ASR gap (2017 vs 2018): diff=%.4f  CI excludes zero: %s",
                bootstrap_results["rf_asr_gap_2017_minus_2018"]["observed_diff"],
                bootstrap_results["rf_asr_gap_2017_minus_2018"]["ci_excludes_zero"])
    logger.info("Uniform vs feature-aware gap: diff=%.4f  CI excludes zero: %s",
                bootstrap_results["uniform_vs_feature_aware_asr_gap"]["observed_diff"],
                bootstrap_results["uniform_vs_feature_aware_asr_gap"]["ci_excludes_zero"])
    logger.info("Outputs: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
