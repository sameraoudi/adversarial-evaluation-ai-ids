"""
Phase 3+4 Runner: Domain-Grounded Perturbation + Semantic Constraint Validation

Addresses:
  R5a: "Uniform L∞ ε is domain-ungrounded — 5% on Flow Duration ≠ 5% on Fwd PSH Flags."
  R5b: "Semantic constraints described but never empirically validated with violation stats."

Pipeline:
  Step 1: Load taxonomy, surrogate, targets, test subset.
  Step 2: Generate adversarial examples (FGSM+PGD × 3 eps levels) with feature-aware
          constraints. Log all constraint violations (Phase 4).
  Step 3: Evaluate all adversarial files on 4 targets with 1000-iter bootstrap CIs.
  Step 4: Load Phase 1 LR-Surr uniform-ε results as baseline for comparison.
  Step 5: Build Tables A (violations), B (ASR impact), C (uniform vs aware).
  Step 6: Save all outputs + generate two figures.

Usage:
    cd <project_root>
    python3.10 experiments/phase3_4_domain_constraints/run_phase3_4.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import set_global_seed
from src.models.wrappers import (
    load_rf_wrapper, load_lr_wrapper, load_mlp_wrapper, load_cnn1d_wrapper
)
from domain_constraints import (
    load_taxonomy, taxonomy_summary, build_epsilon_bounds,
    apply_aware_np, apply_aware_torch, SemanticConstraintLogger,
    EPSILON_MAP,
)

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
(results_dir / "figures").mkdir(exist_ok=True)
(results_dir / "adversarial_examples").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(results_dir / "phase3_4_run.log"),
    ],
)
logger = logging.getLogger("phase3_4.runner")


# ---------------------------------------------------------------------------
# Surrogate loading (Phase 1 checkpoints)
# ---------------------------------------------------------------------------

def load_lr_surrogate(dataset_name: str, device: torch.device) -> torch.nn.Module:
    """Load LR-Surr from Phase 1 checkpoint."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign"))
    from train_surrogates import LinearSurrogate
    from src.data.io import load_npz
    from src.utils.paths import get_splits_dir

    splits_dir = get_splits_dir(dataset_name)
    train = load_npz(splits_dir / "train.npz")
    input_dim = train["X"].shape[1]
    num_classes = int(train["y"].max()) + 1

    ckpt = (PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" /
            "results" / "surrogate_checkpoints" / "lr_surrogate_best.pt")
    model = LinearSurrogate(input_dim, num_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval().to(device)
    logger.info("Loaded LR-Surr: input_dim=%d, num_classes=%d, device=%s",
                input_dim, num_classes, device)
    return model


# ---------------------------------------------------------------------------
# Bootstrap CI (same implementation as Phases 1+2)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray, n_iters: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        boot[i] = values[rng.integers(0, n, size=n)].mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_clean_pred: np.ndarray,
    y_adv_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    clean_correct = (y_clean_pred == y_true).astype(np.float64)
    adv_correct   = (y_adv_pred   == y_true).astype(np.float64)
    clean_acc = float(clean_correct.mean())
    adv_acc   = float(adv_correct.mean())

    orig_mask = clean_correct.astype(bool)
    asr_vals  = (~adv_correct[orig_mask].astype(bool)).astype(np.float64) if orig_mask.any() \
                else np.zeros(1)
    asr = float(asr_vals.mean())

    acc_lo, acc_hi = bootstrap_ci(adv_correct, n_iters=n_boot, ci=ci, seed=seed)
    asr_lo, asr_hi = bootstrap_ci(asr_vals,    n_iters=n_boot, ci=ci, seed=seed + 1)

    return {
        "clean_acc": clean_acc,
        "adv_acc":   adv_acc,
        "delta_acc": clean_acc - adv_acc,
        "asr":       asr,
        "adv_acc_ci_lo": acc_lo, "adv_acc_ci_hi": acc_hi,
        "asr_ci_lo":     asr_lo, "asr_ci_hi":     asr_hi,
        "n_samples":      int(len(y_true)),
        "n_orig_correct": int(orig_mask.sum()),
    }


# ---------------------------------------------------------------------------
# FGSM / PGD with feature-aware constraints
# ---------------------------------------------------------------------------

def fgsm_aware(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    eps_lo: np.ndarray,
    eps_hi: np.ndarray,
    device: torch.device,
    batch_size: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FGSM with per-feature epsilon bounds.
    Returns (X_adv_aware, X_adv_uniform_for_logging).
    X_adv_uniform uses uniform eps = max(eps_hi) for violation logging.
    """
    model.eval()
    N, D = X.shape
    eps_uniform = float(eps_hi.max())

    X_adv_aware   = np.empty_like(X)
    X_adv_uniform = np.empty_like(X)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
        yb = torch.from_numpy(y[start:end].astype(np.int64)).to(device)

        xb_adv = xb.clone().requires_grad_(True)
        loss = F.cross_entropy(model(xb_adv), yb)
        model.zero_grad()
        loss.backward()
        grad_sign = xb_adv.grad.detach().sign()

        # Uniform version (for violation logging)
        x_unif = torch.clamp(xb + eps_uniform * grad_sign, 0.0, 1.0)
        X_adv_uniform[start:end] = x_unif.cpu().numpy()

        # Feature-aware version
        # delta_aware[i] = eps_hi[i] * sign(grad[i]) but clipped to [eps_lo[i], eps_hi[i]]
        lo_t = torch.from_numpy(eps_lo.astype(np.float32)).to(device)
        hi_t = torch.from_numpy(eps_hi.astype(np.float32)).to(device)
        delta = torch.clamp(eps_uniform * grad_sign, lo_t, hi_t)
        x_aware = torch.clamp(xb + delta, 0.0, 1.0)
        X_adv_aware[start:end] = x_aware.cpu().numpy()

    return X_adv_aware.astype(np.float32), X_adv_uniform.astype(np.float32)


def pgd_aware(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    eps_lo: np.ndarray,
    eps_hi: np.ndarray,
    num_steps: int,
    step_size: float,
    device: torch.device,
    seed: int = 42,
    batch_size: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PGD with per-feature epsilon bounds.
    Returns (X_adv_aware, X_adv_uniform_for_logging).
    """
    torch.manual_seed(seed)
    model.eval()
    N, D = X.shape
    eps_uniform = float(eps_hi.max())

    X_adv_aware   = np.empty_like(X)
    X_adv_uniform = np.empty_like(X)

    lo_t = torch.from_numpy(eps_lo.astype(np.float32)).to(device)
    hi_t = torch.from_numpy(eps_hi.astype(np.float32)).to(device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
        yb = torch.from_numpy(y[start:end].astype(np.int64)).to(device)

        # ---- feature-aware PGD ----
        delta_init = torch.empty_like(xb)
        for fi in range(D):
            delta_init[:, fi].uniform_(float(lo_t[fi]), max(float(hi_t[fi]), float(lo_t[fi]) + 1e-9))
        x_adv = torch.clamp(xb + delta_init, 0.0, 1.0)

        for _ in range(num_steps):
            x_adv = x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), yb)
            model.zero_grad()
            loss.backward()
            grad_sign = x_adv.grad.detach().sign()
            x_adv = x_adv.detach() + step_size * grad_sign
            x_adv = apply_aware_torch(x_adv, xb, eps_lo, eps_hi)

        X_adv_aware[start:end] = x_adv.detach().cpu().numpy()

        # ---- uniform PGD for logging (same num_steps, eps_uniform) ----
        delta_init_u = torch.empty_like(xb).uniform_(-eps_uniform, eps_uniform)
        x_adv_u = torch.clamp(xb + delta_init_u, 0.0, 1.0)
        for _ in range(num_steps):
            x_adv_u = x_adv_u.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv_u), yb)
            model.zero_grad()
            loss.backward()
            g = x_adv_u.grad.detach().sign()
            x_adv_u = x_adv_u.detach() + step_size * g
            x_adv_u = torch.clamp(x_adv_u - xb, -eps_uniform, eps_uniform) + xb
            x_adv_u = torch.clamp(x_adv_u, 0.0, 1.0)
        X_adv_uniform[start:end] = x_adv_u.detach().cpu().numpy()

    return X_adv_aware.astype(np.float32), X_adv_uniform.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 results loader (uniform baseline)
# ---------------------------------------------------------------------------

def load_phase1_lr_surr_rows(phase1_dir: Path, target_names: List[str]) -> List[Dict]:
    """Load Phase 1 LR-Surr transfer results for comparison."""
    p = phase1_dir / "transfer_results_full.json"
    if not p.exists():
        logger.warning("Phase 1 results not found at %s", p)
        return []
    with p.open() as f:
        rows = json.load(f)
    out = []
    for r in rows:
        if r["surrogate"] != "lr_surrogate":
            continue
        if r["target"] not in target_names:
            continue
        out.append({
            "condition":     f"Uniform ε={r['eps']:.2f}",
            "attack":        r["attack"].upper(),
            "eps_level":     f"uniform_{r['eps']}",
            "eps_value":     float(r["eps"]),
            "target":        r["target"],
            "clean_acc":     round(r["clean_acc"], 4),
            "adv_acc":       round(r["adv_acc"], 4),
            "delta_acc":     round(r["delta_acc"], 4),
            "asr":           round(r["asr"], 4),
            "asr_ci_lo":     round(r["asr_ci_lo"], 4),
            "asr_ci_hi":     round(r["asr_ci_hi"], 4),
            "n_samples":     r["n_samples"],
            "source":        "phase1_uniform",
        })
    logger.info("Loaded %d Phase 1 LR-Surr rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save_figures(taxonomy: List[Dict], table_c_rows: List[Dict], fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    # ---- Figure 1: Feature category distribution ----
    from collections import Counter
    cat_order = ["immutable", "backward_response", "flags",
                 "increase_only", "derived", "bidirectional"]
    cat_labels = {
        "immutable":         "Immutable\n(ε=0)",
        "backward_response": "Backward\nResponse\n(ε=0)",
        "flags":             "Flags\n(ε=0)",
        "increase_only":     "Increase\nOnly",
        "derived":           "Derived",
        "bidirectional":     "Bidirectional",
    }
    colors = {
        "immutable":         "#d62728",
        "backward_response": "#ff7f0e",
        "flags":             "#e377c2",
        "increase_only":     "#2ca02c",
        "derived":           "#17becf",
        "bidirectional":     "#1f77b4",
    }
    counts = Counter(f["category"] for f in taxonomy)
    vals = [counts.get(c, 0) for c in cat_order]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([cat_labels[c] for c in cat_order], vals,
                  color=[colors[c] for c in cat_order], edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Features", fontsize=11)
    ax.set_title("CICFlowMeter Feature Taxonomy (80 features)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) + 4)
    # Annotate non-perturbable vs perturbable
    non_pert = sum(counts.get(c, 0) for c in ["immutable", "backward_response", "flags"])
    pert = 80 - non_pert
    ax.axvline(x=2.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(1.0, max(vals) + 2, f"Non-perturbable\n(ε=0): {non_pert} features",
            ha="center", fontsize=9, color="darkred")
    ax.text(4.5, max(vals) + 2, f"Perturbable: {pert} features",
            ha="center", fontsize=9, color="darkgreen")
    plt.tight_layout()
    p = fig_dir / "feature_category_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", p)

    # ---- Figure 2: Uniform vs Feature-Aware ASR ----
    if not table_c_rows:
        return

    targets = ["RF", "LR", "MLP", "CNN1D"]
    attacks_plot = ["FGSM", "PGD"]
    eps_levels = ["low", "med", "high"]
    # uniform eps levels to compare with: 0.01→low, 0.03→med, 0.05→high
    eps_map = {"low": 0.01, "med": 0.03, "high": 0.05}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    bar_width = 0.12
    x_base = np.arange(len(targets))

    for ax_idx, attack in enumerate(attacks_plot):
        ax = axes[ax_idx]
        offsets = np.linspace(-(len(eps_levels) - 0.5) * bar_width,
                              (len(eps_levels) - 0.5) * bar_width, len(eps_levels) * 2)
        col_uniform = ["#aec7e8", "#c5b0d5", "#f7b6d2"]   # pale: uniform
        col_aware   = ["#1f77b4", "#9467bd", "#e377c2"]   # vivid: aware

        for li, eps_level in enumerate(eps_levels):
            unif_asrs, aware_asrs = [], []
            for tgt in targets:
                # uniform
                u_row = next((r for r in table_c_rows
                              if r["attack"] == attack
                              and r["target"] == tgt
                              and r["source"] == "phase1_uniform"
                              and abs(r["eps_value"] - eps_map[eps_level]) < 1e-9), None)
                unif_asrs.append(u_row["asr"] if u_row else 0.0)
                # aware
                a_row = next((r for r in table_c_rows
                              if r["attack"] == attack
                              and r["target"] == tgt
                              and r.get("eps_level") == eps_level
                              and r["source"] == "phase3_aware"), None)
                aware_asrs.append(a_row["asr"] if a_row else 0.0)

            off_u = -bar_width * (len(eps_levels) - li - 0.5) + li * 0.5 * bar_width
            off_a = off_u + bar_width
            ax.bar(x_base + off_u, unif_asrs, width=bar_width,
                   color=col_uniform[li], edgecolor="black", linewidth=0.4,
                   label=f"Uniform ε={eps_map[eps_level]:.2f}")
            ax.bar(x_base + off_a, aware_asrs, width=bar_width,
                   color=col_aware[li], edgecolor="black", linewidth=0.4,
                   label=f"Aware {eps_level}")

        ax.set_xticks(x_base)
        ax.set_xticklabels(targets, fontsize=10)
        ax.set_xlabel("Target Model", fontsize=10)
        ax.set_ylabel("ASR", fontsize=10)
        ax.set_title(f"LR-Surr {attack}: Uniform vs Feature-Aware", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 0.35)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle("Uniform vs Domain-Grounded Feature-Aware Perturbation", fontsize=12)
    plt.tight_layout()
    p = fig_dir / "uniform_vs_aware_asr.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    seed         = cfg["seed"]
    n_boot       = cfg["bootstrap"]["n_iterations"]
    ci_level     = cfg["bootstrap"]["confidence"]
    raw_eps      = cfg["attacks"]["raw_eps"]
    pgd_steps    = cfg["attacks"]["pgd"]["num_steps"]
    pgd_step_sz  = cfg["attacks"]["pgd"]["step_size"]
    eps_levels   = list(cfg["attacks"]["epsilon_levels"].keys())   # ["low","med","high"]

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("PHASE 3+4: Domain-Grounded Constraints + Semantic Validation")
    logger.info("Device: %s", device)
    logger.info("=" * 70)

    adv_dir  = results_dir / "adversarial_examples"
    phase1_dir = PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results"

    # ---- Step 1: Load taxonomy, surrogate, targets, data ----
    logger.info("\n[Step 1/6] Loading taxonomy, models, and data...")
    taxonomy = load_taxonomy()
    tx_summary = taxonomy_summary(taxonomy)
    logger.info("Taxonomy: %s", tx_summary)
    with (results_dir / "feature_taxonomy_summary.json").open("w") as f:
        json.dump(tx_summary, f, indent=2)

    # Save the taxonomy as JSON for downstream consumers
    tx_json = [{"index": feat["index"], "name": feat["name"],
                "category": feat["category"], "justification": feat["justification"]}
               for feat in taxonomy]
    with (results_dir / "feature_taxonomy_full.json").open("w") as f:
        json.dump(tx_json, f, indent=2)

    surrogate = load_lr_surrogate(dataset_name, device)

    logger.info("Loading target models...")
    rf_w    = load_rf_wrapper(dataset_name)
    lr_w    = load_lr_wrapper(dataset_name)
    mlp_w   = load_mlp_wrapper(dataset_name)
    cnn1d_w = load_cnn1d_wrapper(dataset_name)
    targets = {"RF": rf_w, "LR": lr_w, "MLP": mlp_w, "CNN1D": cnn1d_w}

    logger.info("Loading Phase 1 test subset (20K)...")
    subset_path = phase1_dir / "test_subset.npz"
    data = np.load(subset_path)
    X_clean, y_true = data["X"], data["y"]
    logger.info("Test subset: %d samples, %d features", *X_clean.shape)

    # Pre-compute clean predictions on all 4 targets (once)
    y_clean_preds: Dict[str, np.ndarray] = {}
    for tname, tw in targets.items():
        y_clean_preds[tname] = tw.predict(X_clean)
        clean_acc = float(np.mean(y_clean_preds[tname] == y_true))
        logger.info("%s clean acc: %.4f", tname, clean_acc)

    constraint_logger = SemanticConstraintLogger(taxonomy)
    logger.info("[Step 1/6] DONE")

    # ---- Step 2: Generate adversarial examples with feature-aware constraints ----
    logger.info("\n[Step 2/6] Generating feature-aware adversarial examples + logging violations...")
    t_gen_start = time.time()

    attacks = ["fgsm", "pgd"]

    for eps_level in eps_levels:
        eps_lo, eps_hi = build_epsilon_bounds(taxonomy, eps_level)
        eps_val = EPSILON_MAP[eps_level]
        logger.info("  eps_level=%s (lo[0]=%.3f hi[0]=%.3f, eps_uniform=%.3f)",
                    eps_level, eps_lo[0], eps_hi[0], eps_val)

        for attack in attacks:
            key = f"lr_surr__{attack}__{eps_level}"
            save_path_aware   = adv_dir / f"{key}__aware.npz"
            save_path_uniform = adv_dir / f"{key}__uniform.npz"

            if save_path_aware.exists() and save_path_uniform.exists():
                logger.info("  Skipping (exists): %s", key)
                # Still need violation stats — recompute from saved files
                X_adv_aware   = np.load(save_path_aware)["X_adv"]
                X_adv_uniform = np.load(save_path_uniform)["X_adv"]
                constraint_logger.log_raw_violations(X_clean, X_adv_uniform, attack, eps_level, eps_val)
                constraint_logger.log_aware_modification(X_adv_uniform, X_adv_aware, attack, eps_level)
                continue

            logger.info("  Generating: attack=%s eps_level=%s ...", attack, eps_level)
            t0 = time.time()

            if attack == "fgsm":
                X_adv_aware, X_adv_uniform = fgsm_aware(
                    surrogate, X_clean, y_true, eps_lo, eps_hi, device
                )
            else:  # pgd
                X_adv_aware, X_adv_uniform = pgd_aware(
                    surrogate, X_clean, y_true, eps_lo, eps_hi,
                    num_steps=pgd_steps, step_size=pgd_step_sz,
                    device=device, seed=seed,
                )

            logger.info("  Generated in %.1f min", (time.time() - t0) / 60)

            # Log constraint violations (Phase 4)
            constraint_logger.log_raw_violations(X_clean, X_adv_uniform, attack, eps_level, eps_val)
            constraint_logger.log_aware_modification(X_adv_uniform, X_adv_aware, attack, eps_level)

            np.savez_compressed(save_path_aware,   X_adv=X_adv_aware)
            np.savez_compressed(save_path_uniform, X_adv=X_adv_uniform)

    # Save violation records
    with (results_dir / "constraint_violation_records.json").open("w") as f:
        json.dump(constraint_logger.records, f, indent=2)

    logger.info("[Step 2/6] DONE — generation + violations logged in %.1f min",
                (time.time() - t_gen_start) / 60)

    # ---- Step 3: Evaluate feature-aware files ----
    logger.info("\n[Step 3/6] Evaluating adversarial examples with 1000-iter bootstrap CIs...")

    # Incremental crash recovery
    incr_path = results_dir / "transfer_results_incremental.jsonl"
    phase3_rows: List[Dict] = []
    already_done: set = set()
    if incr_path.exists():
        with incr_path.open() as f_inc:
            for line in f_inc:
                r = json.loads(line.strip())
                phase3_rows.append(r)
                already_done.add((r["attack"], r["eps_level"], r["target"], r["adv_type"]))
        logger.info("  Loaded %d previously computed rows", len(phase3_rows))

    # Also compute ASR impact while we evaluate
    asr_impact_rows: List[Dict] = []

    with incr_path.open("a") as f_inc:
        for eps_level in eps_levels:
            for attack in attacks:
                key = f"lr_surr__{attack}__{eps_level}"

                # Load both aware and uniform
                X_adv_aware   = np.load(adv_dir / f"{key}__aware.npz")["X_adv"]
                X_adv_uniform = np.load(adv_dir / f"{key}__uniform.npz")["X_adv"]

                for tname, tw in targets.items():
                    # --- Aware ---
                    k_aware = (attack, eps_level, tname, "aware")
                    if k_aware not in already_done:
                        y_adv_pred = tw.predict(X_adv_aware)
                        metrics = compute_metrics_with_ci(
                            y_true, y_clean_preds[tname], y_adv_pred,
                            n_boot=n_boot, ci=ci_level, seed=seed,
                        )
                        row = {
                            "surrogate": "lr_surrogate", "attack": attack.upper(),
                            "target": tname, "eps_level": eps_level,
                            "eps_value": EPSILON_MAP[eps_level],
                            "adv_type": "aware", **metrics,
                        }
                        phase3_rows.append(row)
                        f_inc.write(json.dumps(row) + "\n"); f_inc.flush()
                        already_done.add(k_aware)
                        logger.info(
                            "  [%s/%s/%s/aware] ASR=%.4f CI:[%.4f,%.4f]",
                            attack, eps_level, tname,
                            metrics["asr"], metrics["asr_ci_lo"], metrics["asr_ci_hi"],
                        )

                    # --- Uniform (for this eps level) ---
                    k_unif = (attack, eps_level, tname, "uniform")
                    if k_unif not in already_done:
                        y_adv_pred_u = tw.predict(X_adv_uniform)
                        metrics_u = compute_metrics_with_ci(
                            y_true, y_clean_preds[tname], y_adv_pred_u,
                            n_boot=n_boot, ci=ci_level, seed=seed,
                        )
                        row_u = {
                            "surrogate": "lr_surrogate", "attack": attack.upper(),
                            "target": tname, "eps_level": eps_level,
                            "eps_value": EPSILON_MAP[eps_level],
                            "adv_type": "uniform", **metrics_u,
                        }
                        phase3_rows.append(row_u)
                        f_inc.write(json.dumps(row_u) + "\n"); f_inc.flush()
                        already_done.add(k_unif)

                    # ASR impact
                    aware_asr  = next((r["asr"] for r in phase3_rows
                                       if r["attack"] == attack.upper() and
                                       r["eps_level"] == eps_level and
                                       r["target"] == tname and r["adv_type"] == "aware"), None)
                    unif_asr_local = next((r["asr"] for r in phase3_rows
                                          if r["attack"] == attack.upper() and
                                          r["eps_level"] == eps_level and
                                          r["target"] == tname and r["adv_type"] == "uniform"), None)
                    if aware_asr is not None and unif_asr_local is not None:
                        diff_row = {
                            "attack": attack.upper(), "eps_level": eps_level,
                            "target": tname,
                            "asr_uniform_local": round(unif_asr_local, 4),
                            "asr_aware":         round(aware_asr,      4),
                            "asr_delta":         round(unif_asr_local - aware_asr, 4),
                        }
                        asr_impact_rows.append(diff_row)

    with (results_dir / "transfer_results_phase3.json").open("w") as f:
        json.dump(phase3_rows, f, indent=2)

    logger.info("[Step 3/6] DONE — %d Phase 3 result rows", len(phase3_rows))

    # ---- Step 4: Load Phase 1 uniform baseline ----
    logger.info("\n[Step 4/6] Loading Phase 1 LR-Surr uniform baseline...")
    phase1_rows = load_phase1_lr_surr_rows(phase1_dir, list(targets.keys()))
    logger.info("[Step 4/6] DONE — %d Phase 1 baseline rows", len(phase1_rows))

    # ---- Step 5: Build Tables A, B, C ----
    logger.info("\n[Step 5/6] Building comparison tables...")

    # --- Table A: Constraint Violation Statistics ---
    viol_records = [r for r in constraint_logger.records if r["phase"] == "raw_uniform"]
    fieldnames_a = [
        "attack", "eps_level", "n_samples",
        "range_viol_samples", "range_viol_rate",
        "immut_viol_samples", "immut_viol_rate",
        "cat_viol_immutable", "cat_viol_backward", "cat_viol_flags",
        "dir_viol_samples", "dir_viol_rate",
        "any_viol_samples", "any_viol_rate",
    ]
    with (results_dir / "constraint_violations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_a, extrasaction="ignore")
        w.writeheader(); w.writerows(viol_records)
    logger.info("Saved constraint_violations.csv (%d rows)", len(viol_records))

    # --- Table B: ASR Impact ---
    fieldnames_b = [
        "attack", "eps_level", "target",
        "asr_uniform_local", "asr_aware", "asr_delta",
    ]
    with (results_dir / "asr_impact.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_b, extrasaction="ignore")
        w.writeheader(); w.writerows(asr_impact_rows)
    logger.info("Saved asr_impact.csv (%d rows)", len(asr_impact_rows))

    # --- Table C: Uniform (Phase 1) vs Feature-Aware (Phase 3) ---
    table_c_rows: List[Dict] = []

    # Phase 1 uniform rows
    for r in phase1_rows:
        table_c_rows.append({
            "condition": f"Uniform ε={r['eps_value']:.2f}",
            "attack":    r["attack"],
            "eps_level": r["eps_level"],
            "eps_value": r["eps_value"],
            "target":    r["target"],
            "clean_acc": round(r["clean_acc"], 4),
            "adv_acc":   round(r["adv_acc"],   4),
            "asr":       round(r["asr"],        4),
            "asr_ci_lo": round(r["asr_ci_lo"],  4),
            "asr_ci_hi": round(r["asr_ci_hi"],  4),
            "n_samples": r["n_samples"],
            "source":    "phase1_uniform",
        })

    # Phase 3 feature-aware rows
    for r in phase3_rows:
        if r["adv_type"] != "aware":
            continue
        table_c_rows.append({
            "condition": f"Feature-Aware {r['eps_level']}",
            "attack":    r["attack"],
            "eps_level": r["eps_level"],
            "eps_value": r["eps_value"],
            "target":    r["target"],
            "clean_acc": round(r["clean_acc"], 4),
            "adv_acc":   round(r["adv_acc"],   4),
            "asr":       round(r["asr"],        4),
            "asr_ci_lo": round(r["asr_ci_lo"],  4),
            "asr_ci_hi": round(r["asr_ci_hi"],  4),
            "n_samples": r["n_samples"],
            "source":    "phase3_aware",
        })

    fieldnames_c = [
        "condition", "attack", "eps_level", "eps_value", "target",
        "clean_acc", "adv_acc", "asr", "asr_ci_lo", "asr_ci_hi",
        "n_samples", "source",
    ]
    with (results_dir / "uniform_vs_aware_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_c, extrasaction="ignore")
        w.writeheader(); w.writerows(table_c_rows)
    logger.info("Saved uniform_vs_aware_comparison.csv (%d rows)", len(table_c_rows))

    logger.info("[Step 5/6] DONE")

    # ---- Step 6: Print tables + generate figures ----
    logger.info("\n[Step 6/6] Printing summary + generating figures...")
    _print_table_a(viol_records)
    _print_table_b(asr_impact_rows)
    _print_table_c(table_c_rows)
    _save_figures(taxonomy, table_c_rows, results_dir / "figures")

    total_min = (time.time() - t_gen_start) / 60
    logger.info("=" * 70)
    logger.info("PHASE 3+4 COMPLETE in %.1f min total (from generation start)", total_min)
    logger.info("Results in: %s", results_dir)
    logger.info("  feature_taxonomy_summary.json   — category counts")
    logger.info("  constraint_violations.csv       — Table A: violation rates")
    logger.info("  asr_impact.csv                  — Table B: ASR pre/post constraint")
    logger.info("  uniform_vs_aware_comparison.csv — Table C: key result")
    logger.info("  figures/                        — two PNG figures")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------

def _print_table_a(rows: List[Dict]):
    if not rows:
        return
    print("\n=== TABLE A: Constraint Violation Statistics (Uniform ε Attack) ===")
    fmt = "{:<6} {:<5} {:>9} {:>12} {:>12} {:>9} {:>12}"
    print(fmt.format("Attack", "ε-Lv", "N", "Immut Viol", "BwdResp Viol",
                     "Dir Viol", "Any Viol %"))
    print("-" * 75)
    for r in rows:
        print(fmt.format(
            r["attack"].upper(), r["eps_level"], r["n_samples"],
            f"{r['immut_viol_samples']}({100*r['immut_viol_rate']:.1f}%)",
            f"{r['cat_viol_backward']}({100*r['cat_viol_backward']/r['n_samples']:.1f}%)",
            f"{r['dir_viol_samples']}({100*r['dir_viol_rate']:.1f}%)",
            f"{100*r['any_viol_rate']:.1f}%",
        ))
    print()


def _print_table_b(rows: List[Dict]):
    if not rows:
        return
    print("\n=== TABLE B: ASR Impact of Feature-Aware Constraints ===")
    fmt = "{:<6} {:<5} {:<7} {:>12} {:>11} {:>8}"
    print(fmt.format("Attack", "ε-Lv", "Target", "ASR Uniform", "ASR Aware", "Δ ASR"))
    print("-" * 55)
    for r in rows:
        print(fmt.format(
            r["attack"], r["eps_level"], r["target"],
            f"{r['asr_uniform_local']:.4f}",
            f"{r['asr_aware']:.4f}",
            f"{r['asr_delta']:+.4f}",
        ))
    print()


def _print_table_c(rows: List[Dict]):
    if not rows:
        return
    print("\n=== TABLE C: Uniform ε vs Feature-Aware — Key Phase 3 Result ===")
    # Print FGSM med and PGD med as the primary comparison
    fmt = "{:<30} {:<7} {:<7} {:<9} {:>7} {:>16}"
    print(fmt.format("Condition", "Attack", "Target", "ε-Level",
                     "ASR", "95% CI"))
    print("-" * 80)
    # Show med-epsilon comparison sorted by target
    for attack in ["FGSM", "PGD"]:
        for tgt in ["RF", "LR", "MLP", "CNN1D"]:
            for src, cond_prefix in [("phase1_uniform", "Uniform ε=0.03"),
                                     ("phase3_aware",   "Feature-Aware med")]:
                row = next((r for r in rows
                            if r["attack"] == attack and r["target"] == tgt
                            and r["source"] == src
                            and ("0.03" in r["condition"] or r.get("eps_level") == "med")), None)
                if row:
                    ci = f"[{row['asr_ci_lo']:.4f},{row['asr_ci_hi']:.4f}]"
                    print(fmt.format(row["condition"], attack, tgt,
                                     row.get("eps_level", "med"),
                                     f"{row['asr']:.4f}", ci))
        print()
    # Compute and print summary delta
    print("--- Summary: ASR drop (Uniform med → Feature-Aware med) ---")
    for attack in ["FGSM", "PGD"]:
        for tgt in ["RF", "LR", "MLP", "CNN1D"]:
            u = next((r for r in rows if r["attack"] == attack and r["target"] == tgt
                      and r["source"] == "phase1_uniform" and "0.03" in r["condition"]), None)
            a = next((r for r in rows if r["attack"] == attack and r["target"] == tgt
                      and r["source"] == "phase3_aware" and r.get("eps_level") == "med"), None)
            if u and a:
                delta = u["asr"] - a["asr"]
                interp = (
                    "Aware substantially reduces attack"    if delta > 0.05 else
                    "Moderate constraint effect"            if delta > 0.01 else
                    "Attacks target perturbable features"
                )
                print(f"  {attack:<5} {tgt:<7}: Δ ASR = {delta:+.4f}  → {interp}")
    print()


if __name__ == "__main__":
    main()
