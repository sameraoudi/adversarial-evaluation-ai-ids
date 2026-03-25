"""
Phase 1: Generate Adversarial Examples from Each New Surrogate

For each surrogate (LR, GBT-distill, CNN-v2) x attack (FGSM, PGD) x eps (0.01, 0.03, 0.05):
  1. Load the surrogate model checkpoint
  2. Load the 20k-sample stratified test subset (same seed=42 as original experiments)
  3. Generate adversarial examples using surrogate's gradients ONLY
  4. Apply semantic constraints (clip to [0,1], respect mutable mask)
  5. Save adversarial arrays to disk for later evaluation

CRITICAL: Gradients are NEVER computed from target models.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_npz
from src.utils.paths import get_splits_dir
from src.utils.seed import set_global_seed
from src.attacks.utils import FeatureConstraints, apply_feature_constraints, project_lp_ball
from train_surrogates import LinearSurrogate, GBTDistillNet, CNNv2Surrogate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "experiments" / "phase1_surrogate_redesign" / "results" / "generate_adversarial.log"
        ),
    ]
)
logger = logging.getLogger("phase1.generate_adv")


def load_test_subset(dataset_name: str, max_samples: int, seed: int = 42
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load stratified test subset.  Uses the same seed and random state as the
    original run_attack_suite() in orchestrator.py to ensure identical samples.
    """
    splits_dir = get_splits_dir(dataset_name)
    test = load_npz(splits_dir / "test.npz")
    X, y = test["X"], test["y"]

    if max_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        logger.info("Subsampled test set: %d samples (seed=%d)", len(X), seed)
    else:
        logger.info("Using full test set: %d samples", len(X))
    return X, y


def load_surrogate_model(name: str, surrogate_dir: Path, input_dim: int, num_classes: int,
                         cfg: dict) -> nn.Module:
    """Load a saved surrogate model checkpoint."""
    if name == "lr_surrogate":
        model = LinearSurrogate(input_dim, num_classes)
        ckpt = surrogate_dir / "lr_surrogate_best.pt"
    elif name == "gbt_distill_surrogate":
        g = cfg["surrogates"]["gbt_distill_surrogate"]
        model = GBTDistillNet(input_dim, num_classes, hidden_dims=g["hidden_dims"], dropout=g["dropout"])
        ckpt = surrogate_dir / "gbt_distill_best.pt"
    elif name == "cnn_v2_surrogate":
        c = cfg["surrogates"]["cnn_v2_surrogate"]
        model = CNNv2Surrogate(input_dim, num_classes, channels=c["channels"],
                               kernel_size=c["kernel_size"], fc_dim=c["fc_dim"], dropout=c["dropout"])
        ckpt = surrogate_dir / "cnn_v2_best.pt"
    else:
        raise ValueError(f"Unknown surrogate: {name}")

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded %s from %s", name, ckpt)
    return model


def fgsm_attack(model: nn.Module, X: np.ndarray, y: np.ndarray,
                eps: float, constraints: FeatureConstraints,
                device: torch.device) -> np.ndarray:
    """Single-step FGSM attack using surrogate gradients."""
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    X_t.requires_grad_(False)

    x_adv = X_t.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y_t)
    model.zero_grad()
    loss.backward()

    grad = x_adv.grad.detach()
    x_adv = (X_t + eps * torch.sign(grad)).detach()
    x_adv = project_lp_ball(x_adv, X_t, eps=eps, norm="linf")
    x_adv = apply_feature_constraints(x_adv, X_t, constraints)
    return x_adv.cpu().numpy()


def pgd_attack(model: nn.Module, X: np.ndarray, y: np.ndarray,
               eps: float, step_size: float, num_steps: int,
               constraints: FeatureConstraints, device: torch.device,
               seed: int = 42) -> np.ndarray:
    """Multi-step PGD attack using surrogate gradients."""
    torch.manual_seed(seed)
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)

    # Random start within epsilon ball
    delta = torch.empty_like(X_t).uniform_(-eps, eps)
    x_adv = project_lp_ball(X_t + delta, X_t, eps=eps, norm="linf")
    x_adv = apply_feature_constraints(x_adv, X_t, constraints)

    for _ in range(num_steps):
        x_adv = x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y_t)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = (x_adv.detach() + step_size * torch.sign(grad))
        x_adv = project_lp_ball(x_adv, X_t, eps=eps, norm="linf")
        x_adv = apply_feature_constraints(x_adv, X_t, constraints)

    return x_adv.cpu().numpy()


def log_constraint_stats(X_clean: np.ndarray, X_adv: np.ndarray,
                         surrogate_name: str, attack: str, eps: float) -> Dict:
    """Log and return perturbation statistics."""
    delta = X_adv - X_clean
    linf = np.max(np.abs(delta), axis=1)
    l2 = np.linalg.norm(delta, ord=2, axis=1)
    # Check all values within [0,1]
    clipped = np.clip(X_adv, 0.0, 1.0)
    max_violation = float(np.max(np.abs(X_adv - clipped)))
    stats = {
        "surrogate": surrogate_name,
        "attack": attack,
        "eps": eps,
        "linf_mean": float(linf.mean()),
        "linf_max": float(linf.max()),
        "l2_mean": float(l2.mean()),
        "max_constraint_violation": max_violation,
        "n_samples": len(X_adv),
    }
    logger.info("[%s/%s/eps=%.3f] linf_mean=%.4f linf_max=%.4f l2_mean=%.4f max_violation=%.2e",
                surrogate_name, attack, eps,
                stats["linf_mean"], stats["linf_max"], stats["l2_mean"], stats["max_constraint_violation"])
    return stats


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    seed = cfg["seed"]
    max_samples = cfg["max_samples"]
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    results_dir = Path(__file__).parent / "results"
    surrogate_dir = results_dir / "surrogate_checkpoints"
    adv_dir = results_dir / "adversarial_examples"
    adv_dir.mkdir(parents=True, exist_ok=True)

    # Load test subset (same seed as original orchestrator)
    X_clean, y_true = load_test_subset(dataset_name, max_samples, seed=seed)
    input_dim = X_clean.shape[1]
    num_classes = int(y_true.max()) + 1
    logger.info("Test subset: %d samples, %d features, %d classes",
                len(X_clean), input_dim, num_classes)

    # Save clean data reference (for later evaluation)
    np.savez_compressed(results_dir / "test_subset.npz", X=X_clean, y=y_true)
    logger.info("Saved clean test subset to %s/test_subset.npz", results_dir)

    surrogate_names = ["lr_surrogate", "gbt_distill_surrogate", "cnn_v2_surrogate"]
    attacks = cfg["attacks"]
    eps_list: List[float] = attacks["eps_list"]
    pgd_num_steps = attacks["pgd"]["num_steps"]
    pgd_step_size = attacks["pgd"]["step_size"]

    constraints = FeatureConstraints(clip_min=0.0, clip_max=1.0, mutable_mask=None)

    all_stats = []

    for surrogate_name in tqdm(surrogate_names, desc="Surrogates"):
        model = load_surrogate_model(surrogate_name, surrogate_dir, input_dim, num_classes, cfg)
        model = model.to(device)

        for attack_type in ["fgsm", "pgd"]:
            for eps in tqdm(eps_list, desc=f"{surrogate_name}/{attack_type}", leave=False):
                key = f"{surrogate_name}__{attack_type}__eps{eps:.4f}"
                save_path = adv_dir / f"{key}.npz"

                if save_path.exists():
                    logger.info("Skipping (exists): %s", save_path)
                    continue

                logger.info("Generating: surrogate=%s attack=%s eps=%.4f",
                            surrogate_name, attack_type, eps)

                if attack_type == "fgsm":
                    X_adv = fgsm_attack(model, X_clean, y_true, eps, constraints, device)
                else:  # pgd
                    X_adv = pgd_attack(model, X_clean, y_true, eps,
                                       step_size=pgd_step_size,
                                       num_steps=pgd_num_steps,
                                       constraints=constraints,
                                       device=device,
                                       seed=seed)

                # Enforce constraints post-generation (final check)
                X_adv = np.clip(X_adv, 0.0, 1.0)

                stats = log_constraint_stats(X_clean, X_adv, surrogate_name, attack_type, eps)
                all_stats.append(stats)

                np.savez_compressed(save_path, X_adv=X_adv)
                logger.info("Saved: %s", save_path)

    stats_path = results_dir / "constraint_violation_stats.json"
    with stats_path.open("w") as f_json:
        json.dump(all_stats, f_json, indent=2)
    logger.info("Saved constraint stats to %s", stats_path)
    logger.info("Adversarial generation complete. Total combinations: %d",
                len(surrogate_names) * 2 * len(eps_list))


if __name__ == "__main__":
    main()
