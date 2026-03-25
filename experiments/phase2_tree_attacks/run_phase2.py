"""
Phase 2 Runner: Tree-Aware Attacks for Random Forest

Addresses Reviewer #2 (R2): "gradient-based attacks are fundamentally unsuited
for tree-based models, RF robustness may be gradient incompatibility, not
structural robustness."

Pipeline:
  Step 1: Load RF model and 5K-sample evaluation subset (drawn from Phase 1's 20K subset).
  Step 2: Run three tree-aware attacks × 3 epsilons (with skip logic for crash recovery).
  Step 3: Evaluate each adversarial array with 1000-iteration bootstrap 95% CIs.
  Step 4: Load Phase 1 RF rows and build the combined comparison table.
  Step 5: Save rf_attack_comparison.csv, rf_attack_comparison_summary.json.

Usage:
    cd <project_root>
    python3.10 experiments/phase2_tree_attacks/run_phase2.py

Expected runtime (CPU):
  - Greedy Tree:       ~5-15 min  (O(n_trees × depth²) per sample, 5K samples)
  - FI Random Search:  ~10-15 min (200 candidates × 5K samples × 3ε, batch RF predict)
  - Boundary Attack:   ~20-30 min (n_find_initial/500 batches + 1 walk predict per sample)
  - Evaluation:        ~3-5  min  (bootstrap CIs for 9 attack files × 4 RF eval passes)
  Total:               ~40-65 min
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import set_global_seed

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(results_dir / "phase2_run.log"),
    ],
)
logger = logging.getLogger("phase2.runner")


# ---------------------------------------------------------------------------
# Bootstrap CI (same implementation as Phase 1 evaluate_transfer.py)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray, n_iters: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(
        np.percentile(boot_means, 100 * (1 - alpha))
    )


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_clean_pred: np.ndarray,
    y_adv_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    clean_correct = (y_clean_pred == y_true).astype(np.float64)
    adv_correct = (y_adv_pred == y_true).astype(np.float64)

    clean_acc = float(clean_correct.mean())
    adv_acc = float(adv_correct.mean())
    delta_acc = clean_acc - adv_acc

    orig_correct_mask = clean_correct.astype(bool)
    if orig_correct_mask.sum() > 0:
        asr_values = (~adv_correct[orig_correct_mask].astype(bool)).astype(np.float64)
        asr = float(asr_values.mean())
    else:
        asr_values = np.zeros(1)
        asr = 0.0

    adv_acc_lo, adv_acc_hi = bootstrap_ci(adv_correct, n_iters=n_boot, ci=ci, seed=seed)
    asr_lo, asr_hi = bootstrap_ci(asr_values, n_iters=n_boot, ci=ci, seed=seed + 1)

    return {
        "clean_acc":      clean_acc,
        "adv_acc":        adv_acc,
        "delta_acc":      delta_acc,
        "asr":            asr,
        "adv_acc_ci_lo":  adv_acc_lo,
        "adv_acc_ci_hi":  adv_acc_hi,
        "asr_ci_lo":      asr_lo,
        "asr_ci_hi":      asr_hi,
        "n_samples":      int(len(y_true)),
        "n_orig_correct": int(orig_correct_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------

def load_eval_subset(phase1_results_dir: Path, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Phase 1 20K evaluation subset and downsample to max_samples.
    Uses the same seed=42, ensuring reproducibility across phases.
    """
    subset_path = phase1_results_dir / "test_subset.npz"
    if not subset_path.exists():
        raise FileNotFoundError(
            f"Phase 1 test subset not found at {subset_path}. "
            "Run Phase 1 (generate_adversarial.py) first."
        )
    data = np.load(subset_path)
    X, y = data["X"], data["y"]
    logger.info("Loaded Phase 1 test subset: %d samples", len(X))

    if max_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        logger.info(
            "Downsampled to %d samples (seed=%d) for tree-aware attacks "
            "(documented: O(n_trees × depth²) per sample makes 20K infeasible)",
            max_samples, seed,
        )
    return X, y


def load_rf_model(dataset_name: str):
    rf_path = PROJECT_ROOT / "models" / dataset_name / "baselines" / "rf" / "rf_model.pkl"
    if not rf_path.exists():
        raise FileNotFoundError(f"RF model not found at {rf_path}")
    model = joblib.load(rf_path)
    logger.info(
        "Loaded RF: n_estimators=%d, n_features=%d, n_classes=%d",
        model.n_estimators, model.n_features_in_, model.n_classes_,
    )
    return model


# ---------------------------------------------------------------------------
# Constraint stats logging (same as Phase 1)
# ---------------------------------------------------------------------------

def log_constraint_stats(
    X_clean: np.ndarray, X_adv: np.ndarray, attack: str, eps: float
) -> Dict:
    delta = X_adv - X_clean
    linf = np.max(np.abs(delta), axis=1)
    l2 = np.linalg.norm(delta, ord=2, axis=1)
    clipped = np.clip(X_adv, 0.0, 1.0)
    max_violation = float(np.max(np.abs(X_adv - clipped)))
    stats = {
        "attack": attack,
        "eps": eps,
        "linf_mean": float(linf.mean()),
        "linf_max":  float(linf.max()),
        "l2_mean":   float(l2.mean()),
        "max_constraint_violation": max_violation,
        "n_samples": len(X_adv),
    }
    logger.info(
        "[%s/eps=%.3f] linf_mean=%.4f linf_max=%.4f l2_mean=%.4f max_violation=%.2e",
        attack, eps, stats["linf_mean"], stats["linf_max"],
        stats["l2_mean"], stats["max_constraint_violation"],
    )
    return stats


# ---------------------------------------------------------------------------
# Phase 1 RF results loader
# ---------------------------------------------------------------------------

SURROGATE_DISPLAY = {
    "lr_surrogate":         "LR-Surr",
    "gbt_distill_surrogate": "GBT-Distill",
    "cnn_v2_surrogate":     "CNN-v2",
}

SURROGATE_CATEGORY = {
    "lr_surrogate":         "Gradient transfer",
    "gbt_distill_surrogate": "Tree-approx transfer",
    "cnn_v2_surrogate":     "Cross-arch transfer",
}


def load_phase1_rf_rows(phase1_results_dir: Path) -> List[Dict]:
    """Load Phase 1 transfer results, filter to RF target, format for comparison table."""
    p = phase1_results_dir / "transfer_results_full.json"
    if not p.exists():
        logger.warning("Phase 1 results not found at %s; comparison table will show only Phase 2", p)
        return []
    with p.open() as f:
        rows = json.load(f)

    formatted = []
    for r in rows:
        if r["target"] != "RF":
            continue
        surr_raw = r["surrogate"]
        formatted.append({
            "attack_type": f"{SURROGATE_DISPLAY.get(surr_raw, surr_raw)} {r['attack'].upper()}",
            "category":    SURROGATE_CATEGORY.get(surr_raw, "Gradient transfer"),
            "eps":         float(r["eps"]),
            "clean_acc":   round(r["clean_acc"], 4),
            "adv_acc":     round(r["adv_acc"], 4),
            "delta_acc":   round(r["delta_acc"], 4),
            "asr":         round(r["asr"], 4),
            "asr_ci_lo":   round(r["asr_ci_lo"], 4),
            "asr_ci_hi":   round(r["asr_ci_hi"], 4),
            "n_samples":   r["n_samples"],
            "source":      "phase1",
        })
    logger.info("Loaded %d Phase 1 RF rows", len(formatted))
    return formatted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    seed = cfg["seed"]
    max_samples = cfg["max_samples"]
    eps_list: List[float] = cfg["attacks"]["eps_list"]
    n_boot = cfg["bootstrap"]["n_iterations"]
    ci_level = cfg["bootstrap"]["confidence"]
    set_global_seed(seed)

    adv_dir = results_dir / "adversarial_examples"
    adv_dir.mkdir(exist_ok=True)

    phase1_results_dir = PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results"

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 2: Tree-Aware Attacks for Random Forest")
    logger.info("=" * 70)

    # ---- Step 1: Load RF and eval subset ----
    logger.info("\n[Step 1/5] Loading RF model and evaluation subset...")
    rf_model = load_rf_model(dataset_name)
    X_clean, y_true = load_eval_subset(phase1_results_dir, max_samples, seed)
    n_features = X_clean.shape[1]
    n_classes = int(y_true.max()) + 1
    logger.info("Eval subset: %d samples, %d features, %d classes", len(X_clean), n_features, n_classes)

    # Save eval subset used for Phase 2 (for audit trail)
    np.savez_compressed(results_dir / "phase2_eval_subset.npz", X=X_clean, y=y_true)

    # Pre-compute clean RF predictions
    y_clean_pred = rf_model.predict(X_clean)
    clean_acc = float(np.mean(y_clean_pred == y_true))
    logger.info("RF clean accuracy on eval subset: %.4f", clean_acc)
    logger.info("[Step 1/5] DONE")

    # ---- Step 2: Run tree-aware attacks ----
    logger.info("\n[Step 2/5] Running tree-aware attacks...")
    from tree_attacks import (
        greedy_tree_attack,
        fi_random_search_attack,
        boundary_attack_rf,
    )

    attack_configs = []
    if cfg["attacks"]["greedy_tree"]["enabled"]:
        attack_configs.append(("greedy_tree", {}))
    if cfg["attacks"]["feature_importance"]["enabled"]:
        attack_configs.append(("fi_search", {
            "n_iterations":      cfg["attacks"]["feature_importance"]["n_iterations"],
            "n_perturb_features": cfg["attacks"]["feature_importance"]["n_perturb_features"],
        }))
    if cfg["attacks"]["boundary"]["enabled"]:
        attack_configs.append(("boundary", {
            "n_find_initial": cfg["attacks"]["boundary"]["n_find_initial"],
            "n_iterations":   cfg["attacks"]["boundary"]["n_iterations"],
        }))

    constraint_stats_all = []

    for attack_name, attack_kwargs in attack_configs:
        for eps in eps_list:
            key = f"{attack_name}__eps{eps:.4f}"
            save_path = adv_dir / f"{key}.npz"

            if save_path.exists():
                logger.info("Skipping (exists): %s", save_path)
                continue

            logger.info("\nRunning: attack=%s eps=%.4f ...", attack_name, eps)
            t_atk = time.time()

            if attack_name == "greedy_tree":
                X_adv = greedy_tree_attack(rf_model, X_clean, y_true, eps)
            elif attack_name == "fi_search":
                X_adv = fi_random_search_attack(
                    rf_model, X_clean, y_true, eps,
                    n_iterations=attack_kwargs["n_iterations"],
                    n_perturb_features=attack_kwargs["n_perturb_features"],
                    seed=seed,
                )
            elif attack_name == "boundary":
                X_adv = boundary_attack_rf(
                    rf_model, X_clean, y_true, eps,
                    n_find_initial=attack_kwargs["n_find_initial"],
                    n_iterations=attack_kwargs["n_iterations"],
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown attack: {attack_name}")

            # Enforce semantic constraints (clip to [0,1])
            X_adv = np.clip(X_adv, 0.0, 1.0)

            stats = log_constraint_stats(X_clean, X_adv, attack_name, eps)
            constraint_stats_all.append(stats)

            np.savez_compressed(save_path, X_adv=X_adv)
            logger.info("Saved %s (%.1f min)", save_path, (time.time() - t_atk) / 60)

    with (results_dir / "constraint_violation_stats.json").open("w") as f:
        json.dump(constraint_stats_all, f, indent=2)
    logger.info("[Step 2/5] DONE — adversarial examples generated")

    # ---- Step 3: Evaluate ----
    logger.info("\n[Step 3/5] Evaluating adversarial examples (1000-iter bootstrap CIs)...")

    # Load incremental results for crash recovery
    incremental_path = results_dir / "transfer_results_incremental.jsonl"
    phase2_rows: List[Dict] = []
    already_done = set()
    if incremental_path.exists():
        with incremental_path.open() as f_inc:
            for line in f_inc:
                row = json.loads(line.strip())
                phase2_rows.append(row)
                already_done.add((row["attack"], row["eps"]))
        logger.info("Loaded %d previously computed rows", len(phase2_rows))

    adv_files = sorted(adv_dir.glob("*.npz"))
    logger.info("Found %d adversarial example files to evaluate", len(adv_files))

    with incremental_path.open("a") as f_inc:
        for adv_file in adv_files:
            stem = adv_file.stem
            parts = stem.split("__")
            if len(parts) != 2:
                logger.warning("Unexpected filename: %s, skipping", adv_file.name)
                continue
            attack_name = parts[0]
            eps = float(parts[1].replace("eps", ""))

            key = (attack_name, eps)
            if key in already_done:
                continue

            X_adv = np.load(adv_file)["X_adv"]
            y_adv_pred = rf_model.predict(X_adv)

            metrics = compute_metrics_with_ci(
                y_true=y_true,
                y_clean_pred=y_clean_pred,
                y_adv_pred=y_adv_pred,
                n_boot=n_boot,
                ci=ci_level,
                seed=seed,
            )
            row = {"attack": attack_name, "eps": eps, **metrics}
            phase2_rows.append(row)
            f_inc.write(json.dumps(row) + "\n")
            f_inc.flush()

            logger.info(
                "[%s/eps=%.3f] clean=%.4f adv=%.4f ASR=%.4f CI:[%.4f,%.4f]",
                attack_name, eps,
                metrics["clean_acc"], metrics["adv_acc"],
                metrics["asr"], metrics["asr_ci_lo"], metrics["asr_ci_hi"],
            )
            already_done.add(key)

    with (results_dir / "transfer_results_phase2.json").open("w") as f:
        json.dump(phase2_rows, f, indent=2)
    logger.info("[Step 3/5] DONE — %d Phase 2 result rows", len(phase2_rows))

    # ---- Step 4: Build comparison table ----
    logger.info("\n[Step 4/5] Building RF attack comparison table...")
    phase1_rf_rows = load_phase1_rf_rows(phase1_results_dir)

    # Display names and categories for Phase 2 attacks
    PHASE2_DISPLAY = {
        "greedy_tree": ("Kantchelian Greedy", "Structure-aware"),
        "fi_search":   ("FI-Random Search",   "Importance-guided"),
        "boundary":    ("Boundary Attack",     "Decision-based"),
    }

    # Phase 1 rows already formatted; build Phase 2 rows
    p2_formatted = []
    for r in phase2_rows:
        atk = r["attack"]
        display_name, category = PHASE2_DISPLAY.get(atk, (atk, "Tree-aware"))
        p2_formatted.append({
            "attack_type": display_name,
            "category":    category,
            "eps":         float(r["eps"]),
            "clean_acc":   round(r["clean_acc"], 4),
            "adv_acc":     round(r["adv_acc"], 4),
            "delta_acc":   round(r["delta_acc"], 4),
            "asr":         round(r["asr"], 4),
            "asr_ci_lo":   round(r["asr_ci_lo"], 4),
            "asr_ci_hi":   round(r["asr_ci_hi"], 4),
            "n_samples":   r["n_samples"],
            "source":      "phase2",
        })

    # Combine and sort: Phase 1 first (ordered by surrogate, attack, eps), then Phase 2
    surr_order = ["LR-Surr", "GBT-Distill", "CNN-v2"]
    phase1_sort_key = lambda r: (
        next((i for i, s in enumerate(surr_order) if r["attack_type"].startswith(s)), 99),
        r["attack_type"],
        r["eps"],
    )
    phase1_rf_rows.sort(key=phase1_sort_key)

    phase2_order = ["Kantchelian Greedy", "FI-Random Search", "Boundary Attack"]
    phase2_sort_key = lambda r: (
        next((i for i, s in enumerate(phase2_order) if r["attack_type"] == s), 99),
        r["eps"],
    )
    p2_formatted.sort(key=phase2_sort_key)

    combined = phase1_rf_rows + p2_formatted
    logger.info("[Step 4/5] Comparison table has %d rows (%d Phase 1, %d Phase 2)",
                len(combined), len(phase1_rf_rows), len(p2_formatted))

    # ---- Step 5: Save outputs ----
    logger.info("\n[Step 5/5] Saving outputs...")

    # CSV
    csv_path = results_dir / "rf_attack_comparison.csv"
    fieldnames = [
        "attack_type", "category", "eps",
        "clean_acc", "adv_acc", "delta_acc", "asr",
        "asr_ci_lo", "asr_ci_hi", "n_samples", "source",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    logger.info("Saved CSV: %s", csv_path)

    # Summary JSON
    summary = _compute_summary(combined, phase2_rows, eps_list)
    summary_path = results_dir / "rf_attack_comparison_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", summary_path)

    # Print formatted table
    _print_table(combined)
    _print_summary(summary)

    total_min = (time.time() - t_start) / 60
    logger.info("=" * 70)
    logger.info("PHASE 2 COMPLETE in %.1f min", total_min)
    logger.info("Results in: %s", results_dir)
    logger.info("  rf_attack_comparison.csv          — combined comparison table")
    logger.info("  rf_attack_comparison_summary.json — key statistics")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Table printing and summary helpers
# ---------------------------------------------------------------------------

def _print_table(rows: List[Dict]) -> None:
    header = [
        ("Attack Type",        22),
        ("Category",           22),
        ("ε",                   5),
        ("Clean Acc",          10),
        ("Adv Acc",             9),
        ("Δ Acc",               8),
        ("ASR",                 7),
        ("95% CI (ASR)",       18),
        ("N Samples",          10),
    ]

    sep = " | ".join("-" * w for _, w in header)
    head = " | ".join(h.ljust(w) for h, w in header)

    print()
    print("=" * len(sep))
    print("RF ATTACK COMPARISON TABLE — Phase 1 (gradient transfer) vs Phase 2 (tree-aware)")
    print("=" * len(sep))
    print(head)
    print(sep)

    prev_source = None
    for row in rows:
        if prev_source is not None and row.get("source") != prev_source:
            print(sep)
        prev_source = row.get("source")

        ci_str = f"[{row['asr_ci_lo']:.4f}, {row['asr_ci_hi']:.4f}]"
        cols = [
            str(row["attack_type"]).ljust(22),
            str(row["category"]).ljust(22),
            f"{row['eps']:.2f}".ljust(5),
            f"{row['clean_acc']:.4f}".ljust(10),
            f"{row['adv_acc']:.4f}".ljust(9),
            f"{row['delta_acc']:.4f}".ljust(8),
            f"{row['asr']:.4f}".ljust(7),
            ci_str.ljust(18),
            str(row["n_samples"]).ljust(10),
        ]
        print(" | ".join(cols))

    print("=" * len(sep))
    print(f"Total rows: {len(rows)}")
    print()


def _compute_summary(
    combined: List[Dict], phase2_rows: List[Dict], eps_list: List[float]
) -> Dict:
    summary: Dict = {}

    # Phase 1 RF baseline: max ASR across all gradient attacks and all epsilons
    p1_rows = [r for r in combined if r.get("source") == "phase1"]
    if p1_rows:
        p1_max_asr = max(r["asr"] for r in p1_rows)
        p1_best = max(p1_rows, key=lambda r: r["asr"])
        summary["phase1_rf_max_asr"] = {
            "asr": p1_max_asr,
            "attack": p1_best["attack_type"],
            "eps": p1_best["eps"],
        }

    # Phase 2 per attack: max ASR over all epsilons
    PHASE2_ATTACKS = ["Kantchelian Greedy", "FI-Random Search", "Boundary Attack"]
    summary["phase2_per_attack"] = {}
    for atk_display in PHASE2_ATTACKS:
        atk_rows = [r for r in combined if r.get("source") == "phase2" and r["attack_type"] == atk_display]
        if atk_rows:
            max_row = max(atk_rows, key=lambda r: r["asr"])
            summary["phase2_per_attack"][atk_display] = {
                "max_asr":   max_row["asr"],
                "best_eps":  max_row["eps"],
                "asr_ci_lo": max_row["asr_ci_lo"],
                "asr_ci_hi": max_row["asr_ci_hi"],
                "mean_asr":  round(float(np.mean([r["asr"] for r in atk_rows])), 4),
            }

    # Overall best tree-aware attack
    p2_rows = [r for r in combined if r.get("source") == "phase2"]
    if p2_rows:
        p2_best = max(p2_rows, key=lambda r: r["asr"])
        p2_max_asr = p2_best["asr"]
        summary["phase2_best"] = {
            "attack": p2_best["attack_type"],
            "eps":    p2_best["eps"],
            "asr":    p2_max_asr,
        }

        # Key finding
        if p1_rows:
            delta = p2_max_asr - p1_max_asr
            if delta > 0.05:
                finding = (
                    "TREE-AWARE ATTACKS SIGNIFICANTLY EXCEED GRADIENT ATTACKS "
                    f"(Δ ASR = +{delta:.4f}). RF robustness was partly gradient incompatibility."
                )
            elif delta < -0.02:
                finding = (
                    "TREE-AWARE ATTACKS DO NOT EXCEED GRADIENT ATTACKS "
                    f"(Δ ASR = {delta:.4f}). RF has genuine structural robustness. "
                    "This STRENGTHENS the paper against R2's concern."
                )
            else:
                finding = (
                    f"TREE-AWARE AND GRADIENT ATTACKS ARE COMPARABLE (Δ ASR = {delta:+.4f}). "
                    "RF exhibits consistent robustness across attack paradigms."
                )
            summary["key_finding"] = finding
            summary["delta_asr_tree_vs_gradient"] = round(delta, 4)

    return summary


def _print_summary(summary: Dict) -> None:
    print("\n=== PHASE 2 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print()
    if "key_finding" in summary:
        print("*** KEY FINDING FOR REVIEWER #2 ***")
        print(summary["key_finding"])
        print()


if __name__ == "__main__":
    main()
