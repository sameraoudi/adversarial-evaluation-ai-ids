"""
Phase 1: Evaluate Transfer of Adversarial Examples to All 4 Target Models

For each adversarial array (surrogate x attack x eps):
  Evaluates on all 4 target models: RF, LR, MLP, CNN1D
  Computes: clean_acc, adv_acc, delta_acc, ASR, plus 1000-iteration bootstrap 95% CIs
  Saves results incrementally (crash recovery) and final full CSV.

CRITICAL: Target models are loaded read-only; no gradients are ever computed from them.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_npz
from src.utils.paths import get_splits_dir
from src.utils.seed import set_global_seed
from src.models.wrappers import (
    load_rf_wrapper, load_lr_wrapper, load_mlp_wrapper, load_cnn1d_wrapper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results" / "evaluate_transfer.log"
        ),
    ]
)
logger = logging.getLogger("phase1.evaluate_transfer")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_iters: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap percentile CI for the MEAN of `values`.
    Sampling WITH replacement, n_iters iterations.
    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lower, upper


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_clean_pred: np.ndarray,
    y_adv_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Compute clean_acc, adv_acc, delta_acc, ASR with bootstrap CIs.
    """
    n = len(y_true)
    clean_correct = (y_clean_pred == y_true).astype(np.float64)
    adv_correct = (y_adv_pred == y_true).astype(np.float64)

    clean_acc = float(clean_correct.mean())
    adv_acc = float(adv_correct.mean())
    delta_acc = clean_acc - adv_acc

    # ASR = fraction of originally-correct samples that become incorrect
    orig_correct_mask = clean_correct.astype(bool)
    if orig_correct_mask.sum() > 0:
        asr_values = (~adv_correct[orig_correct_mask].astype(bool)).astype(np.float64)
        asr = float(asr_values.mean())
    else:
        asr_values = np.zeros(1)
        asr = 0.0

    # Bootstrap CIs
    adv_acc_lo, adv_acc_hi = bootstrap_ci(adv_correct, n_iters=n_boot, ci=ci, seed=seed)
    asr_lo, asr_hi = bootstrap_ci(asr_values, n_iters=n_boot, ci=ci, seed=seed + 1)

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "delta_acc": delta_acc,
        "asr": asr,
        "adv_acc_ci_lo": adv_acc_lo,
        "adv_acc_ci_hi": adv_acc_hi,
        "asr_ci_lo": asr_lo,
        "asr_ci_hi": asr_hi,
        "n_samples": n,
        "n_orig_correct": int(orig_correct_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Load target models
# ---------------------------------------------------------------------------

def load_target_models(dataset_name: str) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Load all 4 target models. Each returns a predict(X) callable.
    Models are loaded read-only; no gradient computation.
    """
    logger.info("Loading target models for dataset=%s", dataset_name)
    rf = load_rf_wrapper(dataset_name)
    lr = load_lr_wrapper(dataset_name)
    mlp = load_mlp_wrapper(dataset_name)
    cnn1d = load_cnn1d_wrapper(dataset_name)
    return {
        "RF": rf.predict,
        "LR": lr.predict,
        "MLP": mlp.predict,
        "CNN1D": cnn1d.predict,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    seed = cfg["seed"]
    n_boot = cfg["bootstrap"]["n_iterations"]
    ci_level = cfg["bootstrap"]["confidence"]
    set_global_seed(seed)

    results_dir = Path(__file__).parent / "results"
    adv_dir = results_dir / "adversarial_examples"

    # Load clean test subset
    subset_path = results_dir / "test_subset.npz"
    if not subset_path.exists():
        raise FileNotFoundError(
            f"test_subset.npz not found at {subset_path}. Run generate_adversarial.py first."
        )
    subset = np.load(subset_path)
    X_clean = subset["X"]
    y_true = subset["y"]
    logger.info("Loaded clean test subset: %d samples", len(X_clean))

    # Load target models
    targets = load_target_models(dataset_name)
    logger.info("Target models loaded: %s", list(targets.keys()))

    # Pre-compute clean predictions for each target (once, not per attack)
    logger.info("Computing clean predictions on all target models...")
    clean_preds: Dict[str, np.ndarray] = {}
    for tname, pred_fn in targets.items():
        clean_preds[tname] = pred_fn(X_clean)
        acc = float(np.mean(clean_preds[tname] == y_true))
        logger.info("  %s clean accuracy: %.4f", tname, acc)

    # Enumerate all adversarial example files
    adv_files = sorted(adv_dir.glob("*.npz"))
    logger.info("Found %d adversarial example files", len(adv_files))

    # Incremental results file (crash recovery)
    incremental_path = results_dir / "transfer_results_incremental.jsonl"

    all_rows: List[Dict] = []

    # Load any previously computed results
    already_done = set()
    if incremental_path.exists():
        with incremental_path.open() as f_inc:
            for line in f_inc:
                row = json.loads(line.strip())
                all_rows.append(row)
                already_done.add((row["surrogate"], row["target"], row["attack"], row["eps"]))
        logger.info("Loaded %d previously computed rows from incremental file", len(all_rows))

    with incremental_path.open("a") as f_inc:
        for adv_file in tqdm(adv_files, desc="Evaluating adversarial examples"):
            # Parse filename: {surrogate}__{attack}__eps{eps}.npz
            stem = adv_file.stem
            parts = stem.split("__")
            if len(parts) != 3:
                logger.warning("Unexpected filename format: %s, skipping", adv_file.name)
                continue

            surrogate_name = parts[0]
            attack_type = parts[1]
            eps_str = parts[2].replace("eps", "")
            eps = float(eps_str)

            # Load adversarial examples
            X_adv = np.load(adv_file)["X_adv"]

            for target_name, pred_fn in targets.items():
                key = (surrogate_name, target_name, attack_type, eps)
                if key in already_done:
                    continue

                y_adv_pred = pred_fn(X_adv)
                y_clean_pred = clean_preds[target_name]

                metrics = compute_metrics_with_ci(
                    y_true=y_true,
                    y_clean_pred=y_clean_pred,
                    y_adv_pred=y_adv_pred,
                    n_boot=n_boot,
                    ci=ci_level,
                    seed=seed,
                )

                row = {
                    "surrogate": surrogate_name,
                    "target": target_name,
                    "attack": attack_type,
                    "eps": eps,
                    **metrics,
                }
                all_rows.append(row)
                f_inc.write(json.dumps(row) + "\n")
                f_inc.flush()

                logger.info(
                    "[%s -> %s] %s eps=%.3f | clean=%.3f adv=%.3f (CI:[%.3f,%.3f]) ASR=%.3f (CI:[%.3f,%.3f])",
                    surrogate_name, target_name, attack_type, eps,
                    metrics["clean_acc"], metrics["adv_acc"],
                    metrics["adv_acc_ci_lo"], metrics["adv_acc_ci_hi"],
                    metrics["asr"], metrics["asr_ci_lo"], metrics["asr_ci_hi"],
                )
                already_done.add(key)

    # Save final JSON
    final_json_path = results_dir / "transfer_results_full.json"
    with final_json_path.open("w") as f_json:
        json.dump(all_rows, f_json, indent=2)

    logger.info("Evaluation complete. %d total rows. Saved to %s", len(all_rows), final_json_path)
    return all_rows


if __name__ == "__main__":
    main()
