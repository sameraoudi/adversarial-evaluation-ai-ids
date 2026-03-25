"""
Phase 5 Runner: Expanded Adversarial Training

Addresses R7: "AT was overclaimed — single model, single attack, single ε.
               That's overfitting verification, not mitigation validation."

Pipeline:
  Step 1: Load models, data, surrogate, feature-aware bounds.
  Step 2: Adversarial training — MLP-Hardened + CNN1D-Hardened (surrogate-based).
  Step 3: TRADES training — MLP-TRADES + CNN1D-TRADES (self-generated adv).
  Step 4: Comprehensive generalization testing (Tests G1–G5).
  Step 5: Defense comparison table (AT vs TRADES vs Input Smoothing).
  Step 6: Figures + final outputs.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import set_global_seed
from src.data.io import load_npz
from src.utils.paths import get_splits_dir, get_models_root
from src.models.deep_mlp import MLP
from src.models.deep_cnn1d import CNN1D
from src.models.wrappers import load_mlp_wrapper, load_cnn1d_wrapper

from adv_train import (
    adversarial_train, trades_train,
    fgsm_batch, pgd_batch,
    smoothing_predict, hsja_batch,
)

results_dir   = Path(__file__).parent / "results"
models_dir    = results_dir / "models"
tc_dir        = results_dir / "training_curves"
gen_dir       = results_dir / "generalization"
def_dir       = results_dir / "defense_comparison"
fig_dir       = results_dir / "figures"

for d in [models_dir, tc_dir, gen_dir, def_dir, fig_dir]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(results_dir / "phase5_run.log"),
    ],
)
logger = logging.getLogger("phase5.runner")


# ---------------------------------------------------------------------------
# Utility: bootstrap CI (same as all prior phases)
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    N = len(values)
    boot = np.array([values[rng.integers(0, N, N)].mean() for _ in range(n)])
    a = (1 - ci) / 2
    return float(np.percentile(boot, 100 * a)), float(np.percentile(boot, 100 * (1 - a)))


def compute_metrics(y_true, y_clean_pred, y_adv_pred, n_boot=1000, seed=42):
    clean_ok = (y_clean_pred == y_true).astype(float)
    adv_ok   = (y_adv_pred   == y_true).astype(float)
    mask     = clean_ok.astype(bool)
    asr_vals = (~adv_ok[mask].astype(bool)).astype(float) if mask.any() else np.zeros(1)

    asr_lo, asr_hi = bootstrap_ci(asr_vals, n=n_boot, seed=seed)
    return {
        "clean_acc":     round(float(clean_ok.mean()), 4),
        "adv_acc":       round(float(adv_ok.mean()),   4),
        "asr":           round(float(asr_vals.mean()), 4),
        "asr_ci_lo":     round(asr_lo, 4),
        "asr_ci_hi":     round(asr_hi, 4),
        "n_samples":     int(len(y_true)),
        "n_orig_correct":int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model_arch(model_name: str, input_dim: int, num_classes: int, cfg_deep: dict):
    if model_name == "mlp":
        m = cfg_deep.get("mlp", {})
        return MLP(input_dim, num_classes,
                   hidden_dims=list(m.get("hidden_dims", [256, 128])),
                   dropout=float(m.get("dropout", 0.3)))
    else:
        c = cfg_deep.get("cnn1d", {})
        return CNN1D(input_dim, num_classes,
                     channels=list(c.get("channels", [32, 64])),
                     kernel_sizes=list(c.get("kernel_sizes", [3, 3])),
                     dropout=float(c.get("dropout", 0.3)))


def _load_pretrained(model_name: str, input_dim: int, num_classes: int,
                     cfg_deep: dict, device: torch.device) -> torch.nn.Module:
    """Load original (unhardened) checkpoint for fine-tuning."""
    m = _load_model_arch(model_name, input_dim, num_classes, cfg_deep)
    sub = "mlp" if model_name == "mlp" else "cnn1d"
    ckpt = get_models_root("cicids2017") / "deep" / sub / "deep_mlp_cnn_best.pt"
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m.to(device)


def _load_surrogate(device: torch.device) -> torch.nn.Module:
    """Load LR-Surr from Phase 1 checkpoint (frozen)."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign"))
    from train_surrogates import LinearSurrogate
    ckpt = (PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" /
            "results" / "surrogate_checkpoints" / "lr_surrogate_best.pt")
    splits_dir = get_splits_dir("cicids2017")
    d = load_npz(splits_dir / "train.npz")
    input_dim  = d["X"].shape[1]
    num_classes = int(d["y"].max()) + 1
    m = LinearSurrogate(input_dim, num_classes)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


def _load_all_surrogates(device: torch.device) -> Dict[str, torch.nn.Module]:
    """Load LR-Surr, GBT-Distill, CNN-v2 from Phase 1 checkpoints."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign"))
    from train_surrogates import LinearSurrogate, GBTDistillNet, CNNv2Surrogate
    import yaml as _yaml
    cfg_p1 = _yaml.safe_load(
        (PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "config.yaml").read_text()
    )
    splits_dir = get_splits_dir("cicids2017")
    d = load_npz(splits_dir / "train.npz")
    inp = d["X"].shape[1]; nc = int(d["y"].max()) + 1
    ckpt_dir = (PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" /
                "results" / "surrogate_checkpoints")

    surrs = {}

    lr = LinearSurrogate(inp, nc)
    lr.load_state_dict(torch.load(ckpt_dir / "lr_surrogate_best.pt", map_location="cpu", weights_only=True))
    surrs["LR-Surr"] = lr.eval().to(device)

    g = cfg_p1["surrogates"]["gbt_distill_surrogate"]
    gbt = GBTDistillNet(inp, nc, hidden_dims=g["hidden_dims"], dropout=g["dropout"])
    gbt.load_state_dict(torch.load(ckpt_dir / "gbt_distill_best.pt", map_location="cpu", weights_only=True))
    surrs["GBT-Distill"] = gbt.eval().to(device)

    c = cfg_p1["surrogates"]["cnn_v2_surrogate"]
    cnn = CNNv2Surrogate(inp, nc, channels=c["channels"], kernel_size=c["kernel_size"],
                         fc_dim=c["fc_dim"], dropout=c["dropout"])
    cnn.load_state_dict(torch.load(ckpt_dir / "cnn_v2_best.pt", map_location="cpu", weights_only=True))
    surrs["CNN-v2"] = cnn.eval().to(device)

    for s in surrs.values():
        for p in s.parameters():
            p.requires_grad_(False)

    logger.info("Loaded surrogates: %s", list(surrs.keys()))
    return surrs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(cfg, seed):
    """Load AT training subset + val subset + full eval subset."""
    splits_dir = get_splits_dir("cicids2017")

    tr = load_npz(splits_dir / "train.npz")
    X_tr, y_tr = tr["X"], tr["y"]

    # Stratified subsample for AT training
    n_at = cfg["at_train_samples"]
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(X_tr), size=n_at, replace=False)
    X_at, y_at = X_tr[idx], y_tr[idx]

    # Val subset
    vl  = load_npz(splits_dir / "val.npz")
    X_vl, y_vl = vl["X"], vl["y"]
    n_val = cfg.get("val_samples", 20000)
    idx_v = rng.choice(len(X_vl), size=n_val, replace=False)
    X_val, y_val = X_vl[idx_v], y_vl[idx_v]

    # Eval subset: Phase 1 test subset
    ev = np.load(PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" /
                 "results" / "test_subset.npz")
    X_eval, y_eval = ev["X"], ev["y"]
    n_eval = cfg["evaluation"]["n_eval"]
    if n_eval < len(X_eval):
        idx_e = rng.choice(len(X_eval), size=n_eval, replace=False)
        X_eval, y_eval = X_eval[idx_e], y_eval[idx_e]

    logger.info("AT train: %d, val: %d, eval: %d", len(X_at), len(X_val), len(X_eval))
    return X_at, y_at, X_val, y_val, X_eval, y_eval


def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Feature-aware epsilon bounds
# ---------------------------------------------------------------------------

def _load_feature_bounds(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Phase 3+4 taxonomy and build med-level epsilon bounds as device tensors."""
    tx_path = (PROJECT_ROOT / "experiments" / "phase3_4_domain_constraints" /
               "results" / "feature_taxonomy_full.json")
    with tx_path.open() as f:
        taxonomy = json.load(f)

    eps = 0.03   # "med" level
    ZERO  = {"immutable", "backward_response", "flags"}
    INCR  = {"increase_only"}

    eps_lo = np.full(80, -eps, dtype=np.float32)
    eps_hi = np.full(80,  eps, dtype=np.float32)
    for feat in taxonomy:
        idx, cat = feat["index"], feat["category"]
        if cat in ZERO:
            eps_lo[idx] = 0.0; eps_hi[idx] = 0.0
        elif cat in INCR:
            eps_lo[idx] = 0.0; eps_hi[idx] = eps

    return (torch.from_numpy(eps_lo).to(device),
            torch.from_numpy(eps_hi).to(device))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _predict_np(model, X_np, device, batch_size=2048, use_amp=True):
    """Run model inference on numpy array → numpy predictions."""
    model.eval()
    preds = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.from_numpy(X_np[i:i+batch_size].astype(np.float32)).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type=="cuda"):
                preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def _gen_adv_np(surrogate, X_np, y_np, attack, eps, pgd_steps, device,
                eps_lo_t, eps_hi_t, batch_size=2048, use_amp=True):
    """Generate adversarial examples (numpy in, numpy out) using surrogate."""
    surrogate.eval()
    pgd_ss = eps / max(pgd_steps // 2, 1)
    out = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.from_numpy(X_np[i:i+batch_size].astype(np.float32)).to(device)
        yb = torch.from_numpy(y_np[i:i+batch_size].astype(np.int64)).to(device)
        if attack == "fgsm":
            xa = fgsm_batch(surrogate, xb, yb, eps, eps_lo_t, eps_hi_t)
        else:
            xa = pgd_batch(surrogate, xb, yb, eps, pgd_ss, pgd_steps, eps_lo_t, eps_hi_t)
        out.append(xa.cpu().numpy())
    return np.concatenate(out)


def _gen_adv_uniform_np(surrogate, X_np, y_np, attack, eps, pgd_steps, device,
                        batch_size=2048):
    """Generate adversarial without feature-aware constraints (uniform epsilon)."""
    return _gen_adv_np(surrogate, X_np, y_np, attack, eps, pgd_steps, device,
                       None, None, batch_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    seed    = cfg["seed"]
    n_boot  = cfg["bootstrap"]["n_iterations"]
    ci_lvl  = cfg["bootstrap"]["confidence"]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("PHASE 5: Expanded Adversarial Training")
    logger.info("Device: %s", device)
    logger.info("=" * 70)

    # --- Load config for original deep models ---
    with (PROJECT_ROOT / "configs" / "training" / "deep.yaml").open() as f:
        cfg_deep = yaml.safe_load(f)

    splits_dir = get_splits_dir("cicids2017")
    _d = load_npz(splits_dir / "train.npz")
    input_dim   = _d["X"].shape[1]
    num_classes = int(_d["y"].max()) + 1

    # ---- Step 1: Load data, surrogate, feature bounds ----
    logger.info("\n[Step 1/6] Loading data, surrogate, feature-aware bounds...")
    X_at, y_at, X_val, y_val, X_eval, y_eval = _load_data(cfg, seed)
    surrogate = _load_surrogate(device)
    eps_lo_t, eps_hi_t = _load_feature_bounds(device)

    at_cfg   = cfg["adversarial_training"]
    tr_loader = _make_loader(X_at,  y_at,  at_cfg["batch_size"], shuffle=True)
    vl_loader = _make_loader(X_val, y_val, at_cfg["batch_size"], shuffle=False)

    # Pre-compute clean predictions for eval
    y_clean_preds_base: Dict[str, np.ndarray] = {}
    for mname in ["mlp", "cnn1d"]:
        m = _load_pretrained(mname, input_dim, num_classes, cfg_deep, device)
        y_clean_preds_base[mname] = _predict_np(m, X_eval, device)
        logger.info("%s baseline clean acc: %.4f", mname,
                    float((y_clean_preds_base[mname] == y_eval).mean()))
    logger.info("[Step 1/6] DONE")

    # ---- Step 2: Adversarial Training (MLP + CNN1D) ----
    logger.info("\n[Step 2/6] Adversarial training (surrogate-based)...")
    hardened_models: Dict[str, torch.nn.Module] = {}
    at_curves: Dict[str, List[Dict]] = {}

    for mname in ["mlp", "cnn1d"]:
        ckpt_path = models_dir / f"{mname}_hardened_best.pt"
        if ckpt_path.exists():
            logger.info("  Loading existing hardened %s from %s", mname, ckpt_path)
            m = _load_model_arch(mname, input_dim, num_classes, cfg_deep).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            hardened_models[mname] = m
            # Load training curves if exist
            tc_file = tc_dir / f"{mname}_hardened_training.csv"
            if tc_file.exists():
                with tc_file.open() as f:
                    reader = csv.DictReader(f)
                    at_curves[mname] = list(reader)
            continue

        model = _load_pretrained(mname, input_dim, num_classes, cfg_deep, device)
        curves, best_ckpt = adversarial_train(
            model, surrogate, tr_loader, vl_loader, at_cfg, device,
            models_dir, mname, eps_lo_t, eps_hi_t,
        )
        hardened_models[mname] = model
        at_curves[mname] = curves

        # Save training curves
        with (tc_dir / f"{mname}_hardened_training.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
            w.writeheader(); w.writerows(curves)
        logger.info("  Saved training curves for %s", mname)

    logger.info("[Step 2/6] DONE")

    # ---- Step 3: TRADES training ----
    logger.info("\n[Step 3/6] TRADES training...")
    trades_models: Dict[str, torch.nn.Module] = {}
    trades_curves: Dict[str, List[Dict]] = {}

    if cfg["trades"]["enabled"]:
        for mname in ["mlp", "cnn1d"]:
            ckpt_path = models_dir / f"{mname}_trades_best.pt"
            if ckpt_path.exists():
                logger.info("  Loading existing TRADES %s", mname)
                m = _load_model_arch(mname, input_dim, num_classes, cfg_deep).to(device)
                m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                trades_models[mname] = m
                tc_file = tc_dir / f"{mname}_trades_training.csv"
                if tc_file.exists():
                    with tc_file.open() as f:
                        trades_curves[mname] = list(csv.DictReader(f))
                continue

            model = _load_pretrained(mname, input_dim, num_classes, cfg_deep, device)
            curves, _ = trades_train(
                model, tr_loader, vl_loader, cfg["trades"], device,
                models_dir, mname, eps_lo_t, eps_hi_t,
            )
            trades_models[mname] = model
            trades_curves[mname] = curves

            with (tc_dir / f"{mname}_trades_training.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
                w.writeheader(); w.writerows(curves)
    else:
        logger.info("  TRADES disabled in config, skipping")

    logger.info("[Step 3/6] DONE")

    # ---- Step 4: Comprehensive Generalization Testing ----
    logger.info("\n[Step 4/6] Generalization testing (Tests G1–G5)...")
    all_surrogates = _load_all_surrogates(device)
    eval_cfg = cfg["evaluation"]
    pgd_eval_steps = eval_cfg["pgd_eval_steps"]

    all_gen_rows: List[Dict] = []
    incr_path = gen_dir / "generalization_incremental.jsonl"
    done_keys: set = set()
    if incr_path.exists():
        with incr_path.open() as f:
            for line in f:
                r = json.loads(line)
                all_gen_rows.append(r)
                done_keys.add((r["model"], r["hardened"], r["surrogate"],
                               r["attack"], r["eps"], r["constraint"]))
        logger.info("  Loaded %d cached generalization rows", len(all_gen_rows))

    # Pre-load baseline (unhardened) models for comparison
    baseline_models: Dict[str, torch.nn.Module] = {
        mname: _load_pretrained(mname, input_dim, num_classes, cfg_deep, device)
        for mname in ["mlp", "cnn1d"]
    }

    # All models to test
    model_variants = {
        ("mlp",    False, "Baseline"):    baseline_models["mlp"],
        ("mlp",    True,  "AT-Hardened"): hardened_models["mlp"],
        ("cnn1d",  False, "Baseline"):    baseline_models["cnn1d"],
        ("cnn1d",  True,  "AT-Hardened"): hardened_models["cnn1d"],
    }
    if trades_models:
        model_variants[("mlp",   True,  "TRADES")] = trades_models.get("mlp")
        model_variants[("cnn1d", True,  "TRADES")] = trades_models.get("cnn1d")

    eps_all = eval_cfg["eps_seen"] + eval_cfg["eps_unseen"]
    seen_eps = set(eval_cfg["eps_seen"])
    seen_attacks = {"fgsm", "pgd"}
    seen_surr    = {"LR-Surr"}

    with incr_path.open("a") as f_inc:
        for (mname, is_hardened, variant), target_model in model_variants.items():
            if target_model is None:
                continue
            target_model.eval()

            y_clean_base = _predict_np(target_model, X_eval, device)
            clean_acc = float((y_clean_base == y_eval).mean())

            for surr_name, surrogate_i in all_surrogates.items():
                for attack in ["fgsm", "pgd"]:
                    n_steps = pgd_eval_steps if attack == "pgd" else 1
                    for eps in eps_all:
                        for constraint in ["uniform", "feature_aware"]:
                            key = (mname, str(is_hardened), surr_name, attack, eps, constraint)
                            if key in done_keys:
                                continue

                            # Generate adversarial
                            if constraint == "feature_aware":
                                X_adv = _gen_adv_np(
                                    surrogate_i, X_eval, y_eval, attack, eps,
                                    n_steps, device, eps_lo_t, eps_hi_t
                                )
                            else:
                                X_adv = _gen_adv_uniform_np(
                                    surrogate_i, X_eval, y_eval, attack, eps,
                                    n_steps, device
                                )

                            y_adv_pred = _predict_np(target_model, X_adv, device)
                            m = compute_metrics(y_eval, y_clean_base, y_adv_pred,
                                                n_boot=n_boot, seed=seed)

                            # Seen-during-training flags
                            seen = (
                                is_hardened and
                                surr_name in seen_surr and
                                attack in seen_attacks and
                                eps in seen_eps and
                                constraint == "feature_aware"
                            )

                            row = {
                                "model": mname, "variant": variant,
                                "hardened": str(is_hardened),
                                "surrogate": surr_name,
                                "attack": attack.upper(), "eps": eps,
                                "constraint": constraint,
                                "seen_during_training": str(seen),
                                "clean_acc": clean_acc,
                                **m,
                            }
                            all_gen_rows.append(row)
                            f_inc.write(json.dumps(row) + "\n"); f_inc.flush()
                            done_keys.add(key)

                            logger.info(
                                "  [%s/%s/%s/%s/ε=%.2f/%s] ASR=%.4f (seen=%s)",
                                mname, variant, surr_name, attack.upper(),
                                eps, constraint, m["asr"], seen
                            )

    # Save generalization matrices
    for mname in ["mlp", "cnn1d"]:
        rows = [r for r in all_gen_rows if r["model"] == mname]
        if rows:
            fn = gen_dir / f"{mname}_generalization_matrix.csv"
            with fn.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    # ---- HSJA evaluation (G2: unseen attack) ----
    logger.info("  Running HSJA (decision-based, unseen attack type)...")
    hsja_rows: List[Dict] = []
    hsja_samples = eval_cfg["hsja_samples"]
    hsja_iters   = eval_cfg["hsja_iterations"]
    hsja_bs      = eval_cfg["hsja_binary_search_steps"]

    # Subsample for HSJA
    rng_hsja = np.random.default_rng(seed + 999)
    idx_hsja = rng_hsja.choice(len(X_eval), size=hsja_samples, replace=False)
    X_hsja = X_eval[idx_hsja]; y_hsja = y_eval[idx_hsja]

    for (mname, is_hardened, variant), target_model in model_variants.items():
        if target_model is None:
            continue
        key_hsja = (mname, variant, "HSJA")
        if any(r["attack"] == "HSJA" and r["model"] == mname and r["variant"] == variant
               for r in hsja_rows):
            continue

        target_model.eval()
        X_hsja_t = torch.from_numpy(X_hsja.astype(np.float32)).to(device)
        y_hsja_t = torch.from_numpy(y_hsja.astype(np.int64)).to(device)

        for eps in [0.03, 0.05]:
            X_adv_hsja = hsja_batch(
                target_model, X_hsja_t, y_hsja_t, eps,
                n_iterations=hsja_iters, binary_search_steps=hsja_bs,
                device=device, eps_lo=eps_lo_t, eps_hi=eps_hi_t, seed=seed
            )
            y_clean_h = _predict_np(target_model, X_hsja, device)
            y_adv_h   = X_adv_hsja.cpu().numpy()
            y_adv_pred_h = _predict_np(target_model,
                                        X_adv_hsja.cpu().numpy(), device)

            m_h = compute_metrics(y_hsja, y_clean_h, y_adv_pred_h, n_boot=n_boot, seed=seed)
            row_h = {
                "model": mname, "variant": variant, "hardened": str(is_hardened),
                "surrogate": "HSJA (target)", "attack": "HSJA", "eps": eps,
                "constraint": "feature_aware",
                "seen_during_training": "False",
                "clean_acc": float((y_clean_h == y_hsja).mean()),
                **m_h,
            }
            hsja_rows.append(row_h)
            all_gen_rows.append(row_h)
            logger.info("  [%s/%s/HSJA/ε=%.2f] ASR=%.4f",
                        mname, variant, eps, m_h["asr"])

    if hsja_rows:
        with (gen_dir / "hsja_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hsja_rows[0].keys()))
            w.writeheader(); w.writerows(hsja_rows)

    # Generalization summary
    _save_gen_summary(all_gen_rows, gen_dir, seen_eps, seen_attacks, seen_surr)
    logger.info("[Step 4/6] DONE — %d generalization rows", len(all_gen_rows))

    # ---- Step 5: Defense comparison ----
    logger.info("\n[Step 5/6] Defense comparison (AT vs TRADES vs Input Smoothing)...")
    def_rows: List[Dict] = []
    smooth_cfgs = eval_cfg["smoothing_noise_std"]
    smooth_n    = eval_cfg["smoothing_n_samples"]
    use_amp_def = device.type == "cuda"

    for mname in ["mlp", "cnn1d"]:
        baseline_m = baseline_models[mname]
        hardened_m = hardened_models[mname]
        trades_m   = trades_models.get(mname)

        for attack in ["fgsm", "pgd"]:
            for eps in [0.01, 0.03, 0.05]:
                n_steps = pgd_eval_steps if attack == "pgd" else 1
                X_adv = _gen_adv_np(surrogate, X_eval, y_eval, attack, eps,
                                    n_steps, device, eps_lo_t, eps_hi_t)

                for model_variant, model_i, defense_name in [
                    (mname, baseline_m, "No Defense"),
                    (mname, hardened_m, "AT-Hardened"),
                ] + ([(mname, trades_m, "TRADES")] if trades_m else []):
                    y_clean_d = _predict_np(model_i, X_eval, device)
                    y_adv_d   = _predict_np(model_i, X_adv,  device)
                    m_d = compute_metrics(y_eval, y_clean_d, y_adv_d,
                                          n_boot=n_boot, seed=seed)
                    def_rows.append({
                        "model": mname, "defense": defense_name,
                        "attack": attack.upper(), "eps": eps, **m_d,
                    })

                # Input smoothing on hardened model
                X_eval_t = torch.from_numpy(X_eval.astype(np.float32)).to(device)
                X_adv_t  = torch.from_numpy(X_adv.astype(np.float32)).to(device)

                for std in smooth_cfgs:
                    for ns in smooth_n:
                        # Smoothing on adversarial examples
                        logits_adv_s = smoothing_predict(hardened_m, X_adv_t, std, ns, device, use_amp_def)
                        y_adv_s = logits_adv_s.argmax(1).cpu().numpy()
                        y_clean_s = _predict_np(hardened_m, X_eval, device)
                        m_s = compute_metrics(y_eval, y_clean_s, y_adv_s, n_boot=n_boot, seed=seed)
                        def_rows.append({
                            "model": mname,
                            "defense": f"AT+Smooth(σ={std},n={ns})",
                            "attack": attack.upper(), "eps": eps, **m_s,
                        })

    with (def_dir / "defense_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(def_rows[0].keys()))
        w.writeheader(); w.writerows(def_rows)

    # Defense summary JSON
    def_summary = _compute_defense_summary(def_rows)
    with (def_dir / "defense_summary.json").open("w") as f:
        json.dump(def_summary, f, indent=2)

    logger.info("[Step 5/6] DONE — %d defense comparison rows", len(def_rows))

    # ---- Step 6: Figures + print tables ----
    logger.info("\n[Step 6/6] Generating figures and printing tables...")
    _print_gen_highlights(all_gen_rows)
    _print_defense_table(def_rows)
    _save_figures(all_gen_rows, def_rows, at_curves, fig_dir)

    logger.info("=" * 70)
    logger.info("PHASE 5 COMPLETE")
    logger.info("  models/             — hardened + TRADES checkpoints")
    logger.info("  training_curves/    — per-epoch metrics")
    logger.info("  generalization/     — full test matrix (Tests G1-G5)")
    logger.info("  defense_comparison/ — AT vs TRADES vs Smoothing")
    logger.info("  figures/            — 3 PNG figures")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _save_gen_summary(rows, out_dir, seen_eps, seen_attacks, seen_surr):
    """Compute key generalisation statistics and save JSON."""
    summary = {}
    for mname in ["mlp", "cnn1d"]:
        for variant in ["Baseline", "AT-Hardened", "TRADES"]:
            subset = [r for r in rows if r["model"] == mname and r["variant"] == variant]
            if not subset:
                continue
            seen   = [r for r in subset if r.get("seen_during_training") == "True"]
            unseen = [r for r in subset if r.get("seen_during_training") == "False"]
            k = f"{mname}_{variant}"
            summary[k] = {
                "avg_asr_seen":   round(float(np.mean([r["asr"] for r in seen])),   4) if seen   else None,
                "avg_asr_unseen": round(float(np.mean([r["asr"] for r in unseen])), 4) if unseen else None,
                "max_asr_seen":   round(float(np.max([r["asr"]  for r in seen])),   4) if seen   else None,
                "max_asr_unseen": round(float(np.max([r["asr"]  for r in unseen])), 4) if unseen else None,
                # G1: unseen eps (0.07, 0.10)
                "asr_unseen_eps": {
                    f"eps={r['eps']}": r["asr"]
                    for r in unseen
                    if r.get("attack","").upper() == "FGSM" and
                       r.get("constraint") == "feature_aware" and
                       r.get("surrogate") == "LR-Surr"
                },
                # G3: unseen surrogates (GBT-Distill, CNN-v2)
                "asr_unseen_surr": {
                    r["surrogate"]: r["asr"]
                    for r in subset
                    if r.get("attack","").upper() == "FGSM" and
                       r.get("eps") == 0.03 and
                       r.get("constraint") == "feature_aware" and
                       r["surrogate"] not in seen_surr
                },
            }
    with (out_dir / "generalization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def _compute_defense_summary(rows):
    s = {}
    for mname in ["mlp", "cnn1d"]:
        for atk in ["FGSM", "PGD"]:
            for eps in [0.03]:
                sub = [r for r in rows if r["model"] == mname
                       and r["attack"] == atk and r["eps"] == eps]
                if not sub:
                    continue
                k = f"{mname}_{atk}_eps{eps}"
                s[k] = {r["defense"]: r["asr"] for r in sub}
    return s


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_gen_highlights(rows):
    print("\n=== PHASE 5: GENERALISATION HIGHLIGHTS ===")
    print(f"{'Model':<10} {'Variant':<14} {'Surrogate':<13} {'Atk':<6} {'ε':>5}  "
          f"{'Constr':<13} {'ASR':>7}  {'Seen?':<5}")
    print("-" * 80)
    for r in rows:
        if r.get("attack","") in ("HSJA", "") or r.get("constraint") != "feature_aware":
            continue
        print(f"{r['model']:<10} {r['variant']:<14} {r['surrogate']:<13} "
              f"{r['attack']:<6} {r['eps']:>5.2f}  {r['constraint']:<13} "
              f"{r['asr']:>7.4f}  {r['seen_during_training'][:5]:<5}")
    print()


def _print_defense_table(rows):
    print("\n=== DEFENSE COMPARISON (FGSM ε=0.03) ===")
    fmt = "{:<10} {:<28} {:<6} {:>7}  {:>10}"
    print(fmt.format("Model", "Defense", "Atk", "ASR", "95% CI"))
    print("-" * 65)
    for r in rows:
        if r["attack"] == "FGSM" and r["eps"] == 0.03:
            ci = f"[{r['asr_ci_lo']:.4f},{r['asr_ci_hi']:.4f}]"
            print(fmt.format(r["model"], r["defense"][:28], r["attack"],
                             f"{r['asr']:.4f}", ci))
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figures(gen_rows, def_rows, at_curves, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    # ---- Figure 1: Training curves (clean + robust acc per epoch) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax_i, mname in enumerate(["mlp", "cnn1d"]):
        ax = axes[ax_i]
        curves = at_curves.get(mname, [])
        if not curves:
            continue
        epochs = [int(r.get("epoch", i+1)) for i, r in enumerate(curves)]
        clean  = [float(r.get("val_clean_acc", 0)) for r in curves]
        rob_p  = [float(r.get("val_robust_pgd", 0)) for r in curves]
        rob_f  = [float(r.get("val_robust_fgsm", 0)) for r in curves]
        ax.plot(epochs, clean, "b-o", ms=3, label="Clean acc")
        ax.plot(epochs, rob_f, "g-s", ms=3, label="Robust acc (FGSM)")
        ax.plot(epochs, rob_p, "r-^", ms=3, label="Robust acc (PGD)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{mname.upper()}-Hardened Training Curves")
        ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: Generalisation heatmap (ASR vs ε × attack for hardened MLP) ----
    for mname in ["mlp", "cnn1d"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax_i, variant in enumerate(["Baseline", "AT-Hardened"]):
            ax = axes[ax_i]
            eps_vals = sorted({r["eps"] for r in gen_rows
                               if r["model"] == mname and r.get("constraint") == "feature_aware"})
            attacks  = ["FGSM", "PGD"]
            data = np.zeros((len(attacks), len(eps_vals)))
            for ri, atk in enumerate(attacks):
                for ci, eps in enumerate(eps_vals):
                    rs = [r for r in gen_rows
                          if r["model"] == mname and r["variant"] == variant
                          and r["attack"].upper() == atk and r["eps"] == eps
                          and r["surrogate"] == "LR-Surr"
                          and r.get("constraint") == "feature_aware"]
                    data[ri, ci] = rs[0]["asr"] if rs else 0.0

            im = ax.imshow(data, vmin=0, vmax=0.5, cmap="Reds", aspect="auto")
            ax.set_xticks(range(len(eps_vals))); ax.set_xticklabels([f"ε={e}" for e in eps_vals], fontsize=8)
            ax.set_yticks(range(len(attacks))); ax.set_yticklabels(attacks)
            ax.set_title(f"{mname.upper()} {variant} ASR (LR-Surr, feature-aware)")
            for ri in range(len(attacks)):
                for ci in range(len(eps_vals)):
                    ax.text(ci, ri, f"{data[ri,ci]:.3f}", ha="center", va="center", fontsize=7)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(fig_dir / f"{mname}_generalization_heatmap.png", dpi=150)
        plt.close(fig)

    # ---- Figure 3: Defense comparison bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_i, mname in enumerate(["mlp", "cnn1d"]):
        ax = axes[ax_i]
        sub = [r for r in def_rows if r["model"] == mname
               and r["attack"] == "FGSM" and r["eps"] == 0.03]
        if not sub:
            continue
        names = [r["defense"][:24] for r in sub]
        asrs  = [r["asr"] for r in sub]
        colors = ["#d62728" if "No Defense" in n else
                  "#1f77b4" if "AT-Hardened" in n else
                  "#2ca02c" if "TRADES" in n else "#ff7f0e" for n in names]
        bars = ax.barh(range(len(names)), asrs, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("ASR (FGSM ε=0.03)"); ax.set_title(f"{mname.upper()} Defense Comparison")
        ax.set_xlim(0, min(0.6, max(asrs) + 0.05))
        for bar, v in zip(bars, asrs):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=7)
        ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fig_dir / "defense_comparison_bar.png", dpi=150)
    plt.close(fig)

    logger.info("Saved 3 figures to %s", fig_dir)


if __name__ == "__main__":
    main()
