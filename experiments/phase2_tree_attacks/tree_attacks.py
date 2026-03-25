"""
Phase 2: Tree-Aware Attack Implementations for Random Forest

Three attack methods that exploit RF structure directly (no neural network gradients):

  1. greedy_tree_attack      — Kantchelian-style single-split-flip evasion.
                               Traces natural decision path per tree; finds cheapest
                               L∞ perturbation to force each tree to a different-class
                               leaf; greedily applies cheapest flips until majority vote
                               changes. O(n_trees × depth²) per sample.

  2. fi_random_search_attack — Importance-weighted random search. Perturbs features
                               with probability proportional to RF feature importances.
                               Batch-evaluates 200 candidates at a time; returns first
                               prediction flip or best (lowest true-class confidence).

  3. boundary_attack_rf      — Decision-based boundary walk. Finds initial adversarial
                               starting point via random sampling, then iteratively moves
                               it toward the original sample while it stays adversarial.

CRITICAL: None of these methods use neural network gradients.
          The RF model is loaded read-only (no retraining, no modification).
          Semantic constraints (clip to [0,1], L∞ ball) are applied inside each attack
          and enforced again by the caller.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

logger = logging.getLogger("phase2.tree_attacks")

TREE_LEAF = -1  # sklearn's sentinel value for leaf nodes


# ---------------------------------------------------------------------------
# Shared helpers — tree structure
# ---------------------------------------------------------------------------

def extract_tree_structure(rf_model: RandomForestClassifier) -> List[dict]:
    """
    Extract decision structure from each tree in the ensemble.

    Critical optimisation: `children_left`, `children_right`, `feature`, and
    `threshold` are stored as *Python lists* so that the tight per-node traversal
    loops use fast list-indexing (~2.5x faster than numpy scalar access in
    CPython).  `value` stays as a numpy array because we only need it at leaf
    nodes (argmax call, not in a hot loop).
    """
    trees = []
    for estimator in rf_model.estimators_:
        t = estimator.tree_
        trees.append({
            "feature":        t.feature.tolist(),
            "threshold":      t.threshold.tolist(),
            "children_left":  t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "value":          t.value,                     # (n_nodes, 1, n_classes) numpy
            # Precomputed majority class per node (avoids repeated argmax in hot loop)
            "node_class":     np.argmax(t.value[:, 0, :], axis=1).tolist(),
        })
    return trees


def _tree_predict_single(tree: dict, x) -> int:
    """Return majority class at the leaf x reaches in `tree`.
    x may be a Python list or numpy array."""
    cl = tree["children_left"]
    cr = tree["children_right"]
    feat = tree["feature"]
    thr = tree["threshold"]
    nc = tree["node_class"]   # precomputed per-node majority class
    node = 0
    while cl[node] != TREE_LEAF:
        node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
    return nc[node]


def _follow_natural_class(tree: dict, start_node: int, x) -> int:
    """Starting at `start_node`, follow x's natural path and return leaf class."""
    cl = tree["children_left"]
    cr = tree["children_right"]
    feat = tree["feature"]
    thr = tree["threshold"]
    nc = tree["node_class"]
    node = start_node
    while cl[node] != TREE_LEAF:
        node = cl[node] if x[feat[node]] <= thr[node] else cr[node]
    return nc[node]


def _find_min_flip_cost(
    tree: dict, x: np.ndarray, y_true: int
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Single-split-flip approximation (Kantchelian-style):
    Trace x's natural path through `tree`. At each internal node with
    feature f and threshold t, compute the minimum L∞ cost to take the
    OTHER branch, then follow x's natural path in that forced subtree.

    Returns:
        (linf_cost, delta) for the cheapest split flip that leads to a
        leaf predicting != y_true, within the natural-path approximation.
        Returns (inf, None) if no such flip exists.

    Complexity: O(depth × avg_forced_depth) — typically O(depth²).
    """
    cl = tree["children_left"]
    cr = tree["children_right"]
    feat = tree["feature"]
    thr = tree["threshold"]
    # node_class used in _follow_natural_class (accessed via tree dict)
    n_features = len(x)

    best_cost = np.inf
    best_delta: Optional[np.ndarray] = None

    node = 0
    while cl[node] != TREE_LEAF:
        f = int(feat[node])
        t = float(thr[node])

        if x[f] <= t:
            # Natural: left.  Force: right (need x[f] > t).
            force_cost = t - x[f] + 1e-7
            forced_cls = _follow_natural_class(tree, int(cr[node]), x)
            if forced_cls != y_true and force_cost < best_cost:
                best_cost = force_cost
                d = np.zeros(n_features, dtype=np.float64)
                d[f] = force_cost   # must INCREASE x[f]
                best_delta = d
            node = int(cl[node])
        else:
            # Natural: right.  Force: left (need x[f] <= t).
            force_cost = x[f] - t
            forced_cls = _follow_natural_class(tree, int(cl[node]), x)
            if forced_cls != y_true and force_cost < best_cost:
                best_cost = force_cost
                d = np.zeros(n_features, dtype=np.float64)
                d[f] = -force_cost  # must DECREASE x[f]
                best_delta = d
            node = int(cr[node])

    return best_cost, best_delta


# ---------------------------------------------------------------------------
# Attack 1: Greedy Tree Attack (Kantchelian-style)
# ---------------------------------------------------------------------------

def greedy_tree_attack(
    rf_model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Greedy structure-aware evasion for sklearn RandomForest.

    For each sample x:
      1. Collect per-tree flip costs: for each tree voting y_true, compute the
         minimum L∞ perturbation (single-split-flip) to make it predict ≠ y_true.
      2. Sort by cost (ascending). Greedily apply cheapest flips until the
         majority vote (> n_trees/2 trees) changes or budget exhausted.
      3. Final adversarial is clipped to L∞(epsilon) ball and [0, 1].

    Complexity per sample: O(n_trees × depth²).
    Semantic constraints: L∞ ball + [0,1] clip enforced after each flip.

    Args:
        rf_model : Trained sklearn RandomForestClassifier (read-only).
        X        : Clean samples, shape (N, D), values in [0, 1].
        y        : True labels, shape (N,).
        epsilon  : L∞ perturbation budget.

    Returns:
        X_adv : Adversarial samples, shape (N, D).
    """
    trees = extract_tree_structure(rf_model)
    n_classes = rf_model.n_classes_
    X_adv = X.copy().astype(np.float64)

    for i in tqdm(range(len(X)), desc=f"Greedy-Tree eps={epsilon:.4f}"):
        x_np = X[i].astype(np.float64)
        x = x_np.tolist()  # Python list — 2.5x faster inner-loop indexing
        y_true = int(y[i])

        # Current votes for this sample
        votes = [_tree_predict_single(t, x) for t in trees]
        pred = int(np.argmax(np.bincount(votes, minlength=n_classes)))

        if pred != y_true:
            X_adv[i] = x
            continue

        # Collect flip costs from all trees that vote y_true, within budget
        flip_costs: List[Tuple[float, np.ndarray]] = []
        for j, tree in enumerate(trees):
            if votes[j] == y_true:
                cost, delta = _find_min_flip_cost(tree, x, y_true)
                if delta is not None and cost <= epsilon:
                    flip_costs.append((cost, delta))

        if not flip_costs:
            # No single-split flip possible within budget for any tree
            X_adv[i] = x_np
            continue

        flip_costs.sort(key=lambda t: t[0])

        # Cap at 50 flips and check majority vote every 5 to avoid O(n_flips × n_trees)
        # vote re-evaluation on hard samples.
        MAX_FLIPS = 50
        CHECK_EVERY = 5
        flip_costs = flip_costs[:MAX_FLIPS]

        x_adv = x_np.copy()
        flipped = False
        for batch_start in range(0, len(flip_costs), CHECK_EVERY):
            # Apply a batch of flips
            for cost, delta in flip_costs[batch_start: batch_start + CHECK_EVERY]:
                candidate = np.clip(x_adv + delta, x_np - epsilon, x_np + epsilon)
                candidate = np.clip(candidate, 0.0, 1.0)
                x_adv = candidate

            # Check majority vote once per batch
            x_adv_list = x_adv.tolist()
            new_votes = [_tree_predict_single(t, x_adv_list) for t in trees]
            new_pred = int(np.argmax(np.bincount(new_votes, minlength=n_classes)))
            if new_pred != y_true:
                flipped = True
                break

        X_adv[i] = x_adv

    return X_adv.astype(np.float32)


# ---------------------------------------------------------------------------
# Attack 2: Feature-Importance-Guided Random Search
# ---------------------------------------------------------------------------

def fi_random_search_attack(
    rf_model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    n_iterations: int = 1000,
    n_perturb_features: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Importance-weighted random search for RF evasion.

    Perturbs features with probability proportional to RF feature importances,
    concentrating the budget on features most used by the ensemble.

    For each sample:
      - Generate candidates in batches of 200 (vectorized RF.predict).
      - Return first candidate that changes prediction, or the candidate with
        lowest true-class predict_proba if no flip found.

    Args:
        rf_model          : Trained sklearn RandomForestClassifier.
        X                 : Clean samples, shape (N, D).
        y                 : True labels, shape (N,).
        epsilon           : L∞ perturbation budget.
        n_iterations      : Total candidates per sample.
        n_perturb_features: Features perturbed per candidate (importance-weighted sampling).
        seed              : RNG seed for reproducibility.

    Returns:
        X_adv : Adversarial samples, shape (N, D).
    """
    rng = np.random.default_rng(seed)
    importances = rf_model.feature_importances_
    # Normalise importances to probability distribution (add small floor for zero-importance features)
    imp_floor = 1e-6
    imp_probs = importances + imp_floor
    imp_probs = imp_probs / imp_probs.sum()

    n_samples, n_features = X.shape
    # Process all n_iterations candidates in a SINGLE batch per sample.
    # Avoids repeated RF.predict() calls (each call on (N,D) traverses all 200 trees × N paths).
    # n_iterations is set to ≤200 in config to keep runtime reasonable.
    BATCH = n_iterations  # one batch = all candidates at once

    X_adv = X.copy()

    # Precompute row indices for scatter (same for every sample)
    row_idx = np.repeat(np.arange(BATCH), n_perturb_features)

    for i in tqdm(range(n_samples), desc=f"FI-Search eps={epsilon:.4f}"):
        x = X[i]
        y_true = int(y[i])

        if rf_model.predict(x.reshape(1, -1))[0] != y_true:
            continue  # already misclassified

        # Fully vectorised candidate generation — no Python loop over candidates.
        # Importance-weighted feature selection via Efraimidis-Spirakis reservoir keys:
        # key_ij = U^(1/w_j)  →  top-k features per row give weighted sample w/o replacement.
        u = rng.uniform(1e-10, 1.0, (BATCH, n_features))
        keys = u ** (1.0 / imp_probs)          # (BATCH, n_features); higher imp → larger key
        col_idx = np.argpartition(-keys, n_perturb_features, axis=1)[:, :n_perturb_features]
        # (BATCH, n_perturb_features) — selected feature indices per candidate

        pert_vals = rng.uniform(-epsilon, epsilon, (BATCH, n_perturb_features))

        delta = np.zeros((BATCH, n_features))
        delta[row_idx, col_idx.ravel()] = pert_vals.ravel()

        candidates = np.clip(x + delta, x - epsilon, x + epsilon)
        candidates = np.clip(candidates, 0.0, 1.0)

        # Single RF.predict call for all candidates
        preds = rf_model.predict(candidates)
        flip_idx = np.where(preds != y_true)[0]

        if len(flip_idx) > 0:
            X_adv[i] = candidates[flip_idx[0]]
        else:
            # No flip found: return best-effort (candidate with lowest vote fraction for y_true)
            # Use predict_proba only once per sample, only on failure (rare after enough iterations)
            probas = rf_model.predict_proba(candidates)
            best_idx = int(np.argmin(probas[:, y_true]))
            X_adv[i] = candidates[best_idx]

    return X_adv.astype(np.float32)


# ---------------------------------------------------------------------------
# Attack 3: Boundary Attack (decision-based)
# ---------------------------------------------------------------------------

def boundary_attack_rf(
    rf_model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    n_find_initial: int = 2500,
    n_iterations: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Simplified boundary attack for RF (decision-based; uses only hard-label predict).

    Phase 1 — Find initial adversarial starting point:
        Batch random sampling (batch=500): first try large perturbations (±0.5 around x),
        then pure random in [0,1]^D if needed.  One RF.predict(500, D) call per batch.

    Phase 2 — Batched boundary walk (fully vectorised):
        Generate n_iterations candidates from x_adv_init toward x with linearly spaced
        alphas in [0.01, 0.99] plus small Gaussian noise.  Evaluate all in ONE
        RF.predict(n_iterations, D) call.  Return the adversarial candidate with the
        highest alpha (i.e. closest to the original x = minimum perturbation on the
        decision boundary).

        NOTE: This batched walk is equivalent in spirit to the iterative boundary walk
        but avoids n_iterations sequential RF.predict(1, D) calls (each ≈ 35 ms due to
        sklearn overhead) and instead uses one RF.predict(200, D) call (≈ 53 ms total).
        Speedup: ~135×; necessary to keep runtime feasible for 5K × 3ε samples on CPU.

    Final result is projected to L∞(epsilon) ball and [0,1].

    Args:
        rf_model        : Trained sklearn RandomForestClassifier.
        X               : Clean samples, shape (N, D).
        y               : True labels, shape (N,).
        epsilon         : L∞ budget for final projection.
        n_find_initial  : Max random trials to find adversarial starting point.
        n_iterations    : Number of walk candidates generated in one batch.
        seed            : RNG seed.

    Returns:
        X_adv : Adversarial samples, shape (N, D).
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    BATCH = 500

    X_adv = X.copy()

    for i in tqdm(range(n_samples), desc=f"Boundary eps={epsilon:.4f}"):
        x = X[i]
        y_true = int(y[i])

        if rf_model.predict(x.reshape(1, -1))[0] != y_true:
            continue

        # ---- Phase 1: find initial adversarial point ----
        x_adv_init: Optional[np.ndarray] = None
        half = n_find_initial // 2

        for start in range(0, n_find_initial, BATCH):
            sz = min(BATCH, n_find_initial - start)
            if start < half:
                # Large perturbations near x
                candidates = np.clip(
                    x + rng.uniform(-0.5, 0.5, (sz, n_features)), 0.0, 1.0
                )
            else:
                # Fully random in feature space
                candidates = rng.uniform(0.0, 1.0, (sz, n_features))

            preds = rf_model.predict(candidates)
            flip_idx = np.where(preds != y_true)[0]
            if len(flip_idx) > 0:
                x_adv_init = candidates[flip_idx[0]].copy()
                break

        if x_adv_init is None:
            # Could not find adversarial starting point; return best-effort (clean x)
            X_adv[i] = x
            continue

        # ---- Phase 2: batched walk toward original ----
        # Generate n_iterations candidates spanning the line from x_adv_init to x.
        # alphas close to 1 → candidate close to x (small perturbation from x).
        alphas = np.linspace(0.01, 0.99, n_iterations)          # (n_iterations,)
        noise = rng.normal(0.0, epsilon * 0.05, (n_iterations, n_features))
        walk_candidates = np.clip(
            (1.0 - alphas[:, None]) * x_adv_init + alphas[:, None] * x + noise,
            0.0, 1.0,
        )                                                         # (n_iterations, n_features)

        # One predict call for all walk candidates
        walk_preds = rf_model.predict(walk_candidates)
        adv_mask = walk_preds != y_true

        if adv_mask.any():
            # Pick adversarial candidate with highest alpha (closest to original x)
            best_walk_idx = int(np.where(adv_mask)[0][-1])
            x_adv = walk_candidates[best_walk_idx]
        else:
            # Walk failed to stay adversarial; keep x_adv_init
            x_adv = x_adv_init

        # Project to L∞ epsilon ball around original x
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
        x_adv = np.clip(x_adv, 0.0, 1.0)
        X_adv[i] = x_adv

    return X_adv.astype(np.float32)
