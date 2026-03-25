"""
Phase 3+4: Domain-Grounded Perturbation Constraints + Semantic Validation

Provides:
  - load_taxonomy()          — parse feature_taxonomy.yaml → list of dicts
  - build_epsilon_bounds()   — per-feature [lo, hi] epsilon arrays from taxonomy
  - apply_aware_np()         — numpy: apply feature-aware constraints to a delta
  - apply_aware_torch()      — torch: same, for use inside PGD iteration loop
  - SemanticConstraintLogger — tracks constraint violations and ASR impact
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger("phase3_4.domain_constraints")

EPSILON_MAP: Dict[str, float] = {"low": 0.01, "med": 0.03, "high": 0.05}

# Categories whose perturbation must be zeroed
ZERO_EPSILON_CATS = {"immutable", "backward_response", "flags"}
# Categories with positive-only perturbation
INCREASE_ONLY_CATS = {"increase_only"}
# Categories with full bidirectional perturbation
BIDIR_CATS = {"bidirectional", "derived"}


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_taxonomy(yaml_path: Optional[Path] = None) -> List[Dict]:
    """Load feature_taxonomy.yaml → sorted list of feature dicts."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "feature_taxonomy.yaml"
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    features = data["features"]
    features.sort(key=lambda x: x["index"])
    assert len(features) == 80, f"Expected 80 features, got {len(features)}"
    for i, feat in enumerate(features):
        assert feat["index"] == i, f"Index gap at {i}"
    return features


def taxonomy_summary(taxonomy: List[Dict]) -> Dict:
    """Count features per category."""
    from collections import Counter
    counts = Counter(f["category"] for f in taxonomy)
    zero_eps = sum(counts[c] for c in ZERO_EPSILON_CATS)
    perturbable = 80 - zero_eps
    return {
        "counts_by_category": dict(counts),
        "zero_epsilon_total": zero_eps,
        "perturbable_total": perturbable,
        "perturbable_fraction": round(perturbable / 80, 4),
    }


# ---------------------------------------------------------------------------
# Epsilon bound arrays
# ---------------------------------------------------------------------------

def build_epsilon_bounds(
    taxonomy: List[Dict], epsilon_level: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-feature lower and upper epsilon bound arrays.

    Returns:
        eps_lo : (80,) lower bound on delta[i]
        eps_hi : (80,) upper bound on delta[i]
    """
    eps = EPSILON_MAP[epsilon_level]
    eps_lo = np.full(80, -eps, dtype=np.float64)
    eps_hi = np.full(80, eps,  dtype=np.float64)

    for feat in taxonomy:
        idx = feat["index"]
        cat = feat["category"]
        if cat in ZERO_EPSILON_CATS:
            eps_lo[idx] = 0.0
            eps_hi[idx] = 0.0
        elif cat in INCREASE_ONLY_CATS:
            eps_lo[idx] = 0.0      # no decrease
            eps_hi[idx] = eps
        # bidirectional and derived: full [-eps, eps] (default)

    return eps_lo, eps_hi


# ---------------------------------------------------------------------------
# Constraint application — numpy
# ---------------------------------------------------------------------------

def apply_aware_np(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    eps_lo: np.ndarray,
    eps_hi: np.ndarray,
) -> np.ndarray:
    """
    Apply feature-aware per-feature epsilon bounds (numpy).

    delta = x_adv - x_orig
    delta clipped to [eps_lo[i], eps_hi[i]] per feature
    result clipped to [0, 1]
    """
    delta = x_adv - x_orig                                     # (N, 80)
    delta = np.clip(delta, eps_lo[None, :], eps_hi[None, :])   # broadcast
    return np.clip(x_orig + delta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Constraint application — torch (for use inside PGD)
# ---------------------------------------------------------------------------

def apply_aware_torch(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    eps_lo: np.ndarray,
    eps_hi: np.ndarray,
) -> torch.Tensor:
    """
    Apply feature-aware per-feature epsilon bounds (torch, for PGD iteration).
    """
    dev = x_adv.device
    lo_t = torch.from_numpy(eps_lo.astype(np.float32)).to(dev)   # (80,)
    hi_t = torch.from_numpy(eps_hi.astype(np.float32)).to(dev)   # (80,)
    delta = x_adv - x_orig
    delta = torch.clamp(delta, lo_t, hi_t)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Semantic Constraint Logger
# ---------------------------------------------------------------------------

class SemanticConstraintLogger:
    """
    Logs semantic constraint violations and their impact on ASR.

    Usage (Phase 4):
        logger = SemanticConstraintLogger(taxonomy)
        # After generating x_adv_raw (no constraints applied):
        logger.log_raw_violations(x_clean, x_adv_raw)
        # After applying feature-aware constraints:
        logger.log_constraint_delta(x_adv_raw, x_adv_aware, attack, eps_level)
    """

    def __init__(self, taxonomy: List[Dict]):
        self.taxonomy = taxonomy
        self.cat_map = {f["index"]: f["category"] for f in taxonomy}
        self.name_map = {f["index"]: f["name"] for f in taxonomy}
        self.records: List[Dict] = []

    def log_raw_violations(
        self,
        x_clean: np.ndarray,
        x_adv_raw: np.ndarray,
        attack: str,
        eps_level: str,
        eps_raw: float,
    ) -> Dict:
        """
        For x_adv_raw (generated with uniform eps, no semantic constraints):
        Count how many samples violate each constraint type.

        Returns violation_stats dict (also appended to self.records).
        """
        N, D = x_adv_raw.shape
        delta = x_adv_raw - x_clean   # (N, D)

        # 1. Range violations: any feature outside [0,1] before clipping
        below_zero = (x_adv_raw < 0.0)            # (N, D) bool
        above_one  = (x_adv_raw > 1.0)            # (N, D) bool
        range_viol_samples = int(np.any(below_zero | above_one, axis=1).sum())
        range_viol_features = int((np.any(below_zero | above_one, axis=0)).sum())

        # 2. Immutability / backward / flags — features that should have ε=0 but were changed
        zero_eps_mask = np.array([
            self.cat_map[i] in ZERO_EPSILON_CATS for i in range(D)
        ], dtype=bool)                             # (D,)
        zc = np.abs(delta[:, zero_eps_mask]) > 1e-9  # (N, n_zero_eps_feats) bool
        immut_viol_samples = int(np.any(zc, axis=1).sum())
        immut_viol_features = int(np.any(zc, axis=0).sum())

        # Per-category breakdown of how many samples had that category perturbed
        cat_viol: Dict[str, int] = {}
        for cat in ZERO_EPSILON_CATS:
            cat_indices = [i for i in range(D) if self.cat_map[i] == cat]
            if cat_indices:
                changed = np.any(np.abs(delta[:, cat_indices]) > 1e-9, axis=1)
                cat_viol[cat] = int(changed.sum())

        # 3. Direction violations: increase_only features perturbed negatively
        io_mask = np.array([
            self.cat_map[i] in INCREASE_ONLY_CATS for i in range(D)
        ], dtype=bool)
        neg_io = (delta[:, io_mask] < -1e-9)          # (N, n_io_feats) bool
        dir_viol_samples = int(np.any(neg_io, axis=1).sum())
        dir_viol_features = int(np.any(neg_io, axis=0).sum())

        stats = {
            "attack": attack,
            "eps_level": eps_level,
            "eps_raw": eps_raw,
            "n_samples": N,
            "phase": "raw_uniform",
            # Range
            "range_viol_samples": range_viol_samples,
            "range_viol_rate": round(range_viol_samples / N, 6),
            "range_viol_features": range_viol_features,
            # Immutability / backward / flags
            "immut_viol_samples": immut_viol_samples,
            "immut_viol_rate": round(immut_viol_samples / N, 6),
            "immut_viol_features": immut_viol_features,
            # Per-category immutability breakdown
            "cat_viol_immutable":    cat_viol.get("immutable", 0),
            "cat_viol_backward":     cat_viol.get("backward_response", 0),
            "cat_viol_flags":        cat_viol.get("flags", 0),
            # Direction
            "dir_viol_samples": dir_viol_samples,
            "dir_viol_rate": round(dir_viol_samples / N, 6),
            "dir_viol_features": dir_viol_features,
            # Summary: any violation
            "any_viol_samples": int(np.any(
                np.any(below_zero | above_one, axis=1) |
                np.any(np.abs(delta[:, zero_eps_mask]) > 1e-9, axis=1) |
                np.any(delta[:, io_mask] < -1e-9, axis=1)
            ).reshape(-1)[:1][0] if (
                np.any(below_zero | above_one, axis=1) |
                np.any(np.abs(delta[:, zero_eps_mask]) > 1e-9, axis=1) |
                np.any(delta[:, io_mask] < -1e-9, axis=1)
            ).any() else 0),
        }

        # Fix any_viol_samples — compute correctly
        any_viol = (
            np.any(below_zero | above_one, axis=1) |
            np.any(np.abs(delta[:, zero_eps_mask]) > 1e-9, axis=1) |
            np.any(delta[:, io_mask] < -1e-9, axis=1)
        )
        stats["any_viol_samples"] = int(any_viol.sum())
        stats["any_viol_rate"] = round(float(any_viol.mean()), 6)

        self.records.append(stats)
        logger.info(
            "[%s/%s] raw violations: range=%d/%.1f%%, immut=%d/%.1f%%, dir=%d/%.1f%%",
            attack, eps_level,
            range_viol_samples, 100 * range_viol_samples / N,
            immut_viol_samples, 100 * immut_viol_samples / N,
            dir_viol_samples,   100 * dir_viol_samples   / N,
        )
        return stats

    def log_aware_modification(
        self,
        x_adv_uniform: np.ndarray,
        x_adv_aware: np.ndarray,
        attack: str,
        eps_level: str,
    ) -> Dict:
        """
        Log how many samples were modified when switching from uniform to feature-aware.
        """
        N = x_adv_uniform.shape[0]
        diff = np.abs(x_adv_uniform - x_adv_aware)
        modified = np.any(diff > 1e-9, axis=1)
        linf_delta = diff.max(axis=1)

        stats = {
            "attack": attack,
            "eps_level": eps_level,
            "n_samples": N,
            "phase": "uniform_to_aware",
            "samples_modified": int(modified.sum()),
            "modification_rate": round(float(modified.mean()), 6),
            "linf_delta_mean": round(float(linf_delta.mean()), 6),
            "linf_delta_max":  round(float(linf_delta.max()),  6),
        }
        self.records.append(stats)
        logger.info(
            "[%s/%s] uniform→aware: modified %d/%.1f%% samples",
            attack, eps_level, stats["samples_modified"],
            100 * stats["modification_rate"],
        )
        return stats

    def compute_asr_impact(
        self,
        model_predict,
        x_clean: np.ndarray,
        x_adv_uniform: np.ndarray,
        x_adv_aware: np.ndarray,
        y_true: np.ndarray,
        model_name: str,
        attack: str,
        eps_level: str,
    ) -> Dict:
        """
        Measure ASR before (uniform) and after (feature-aware) constraint enforcement.
        """
        N = len(y_true)
        y_clean = model_predict(x_clean)
        correct_clean = (y_clean == y_true)

        y_unif = model_predict(x_adv_uniform)
        y_aware = model_predict(x_adv_aware)

        n_orig_correct = int(correct_clean.sum())
        if n_orig_correct == 0:
            return {}

        asr_unif  = float((y_unif[correct_clean]  != y_true[correct_clean]).mean())
        asr_aware = float((y_aware[correct_clean] != y_true[correct_clean]).mean())

        diff = np.abs(x_adv_uniform - x_adv_aware)
        modified = np.any(diff > 1e-9, axis=1)

        stats = {
            "model": model_name,
            "attack": attack,
            "eps_level": eps_level,
            "n_samples": N,
            "n_orig_correct": n_orig_correct,
            "asr_uniform":    round(asr_unif,  6),
            "asr_aware":      round(asr_aware, 6),
            "asr_delta":      round(asr_unif - asr_aware, 6),
            "samples_modified": int(modified.sum()),
            "modification_rate": round(float(modified.mean()), 6),
        }
        logger.info(
            "[%s/%s/%s] ASR uniform=%.4f aware=%.4f Δ=%.4f",
            model_name, attack, eps_level,
            asr_unif, asr_aware, asr_unif - asr_aware,
        )
        return stats
