"""
Phase 1 Runner: Complete pipeline from surrogate training to transfer matrix.

Run this script to execute all Phase 1 experiments in order:
  Step 1: Verify target models are trained (RF, LR, MLP, CNN1D)
  Step 2: Train three new surrogate models
  Step 3: Generate adversarial examples (3 surrogates x 2 attacks x 3 epsilons)
  Step 4: Evaluate transfer to all 4 targets with bootstrap CIs
  Step 5: Build and print the 72-row transfer matrix

Usage:
    cd <project_root>
    python3.10 experiments/phase1_surrogate_redesign/run_phase1.py

Expected runtime (CPU-only):
  - Surrogate training: ~20-40 min (XGBoost for GBT-Distill is the bottleneck)
  - Adversarial generation: ~10-20 min
  - Transfer evaluation + bootstrap CIs: ~30-60 min
  Total: ~60-120 min on CPU
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.paths import get_models_root
from src.utils.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase1.runner")


def check_target_models(dataset_name: str) -> bool:
    """Verify all 4 target models exist."""
    models_root = get_models_root(dataset_name)
    required = [
        models_root / "baselines" / "rf" / "rf_model.pkl",
        models_root / "baselines" / "lr" / "lr_model.pkl",
        models_root / "deep" / "mlp" / "deep_mlp_cnn_best.pt",
        models_root / "deep" / "cnn1d" / "deep_mlp_cnn_best.pt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.error("Missing target model files:")
        for p in missing:
            logger.error("  %s", p)
        return False
    logger.info("All 4 target models found:")
    for p in required:
        logger.info("  %s (%.1f MB)", p.name, p.stat().st_size / 1e6)
    return True


def main():
    dataset_name = "cicids2017"
    set_global_seed(42)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 1: Surrogate Redesign Experiments")
    logger.info("=" * 70)
    logger.info("Expected runtime: 60-120 min (CPU-only)")

    # Step 1: Verify target models
    logger.info("\n[Step 1/5] Checking target models...")
    if not check_target_models(dataset_name):
        logger.error(
            "Target models missing. Run the following first:\n"
            "  python3.10 -m scripts.train_baselines_cicids2017\n"
            "  python3.10 -m scripts.train_deep_cicids2017"
        )
        sys.exit(1)
    logger.info("[Step 1/5] DONE — all target models present")

    # Step 2: Train surrogates
    logger.info("\n[Step 2/5] Training surrogate models...")
    t2 = time.time()
    surrogate_checkpoints = [
        results_dir / "surrogate_checkpoints" / "lr_surrogate_best.pt",
        results_dir / "surrogate_checkpoints" / "gbt_distill_best.pt",
        results_dir / "surrogate_checkpoints" / "cnn_v2_best.pt",
    ]
    all_surrogates_exist = all(p.exists() for p in surrogate_checkpoints)
    if all_surrogates_exist:
        logger.info("[Step 2/5] SKIPPED — all surrogate checkpoints already exist")
    else:
        from train_surrogates import main as train_main
        train_main()
        logger.info("[Step 2/5] DONE in %.1f min", (time.time() - t2) / 60)

    # Step 3: Generate adversarial examples
    logger.info("\n[Step 3/5] Generating adversarial examples...")
    t3 = time.time()
    # Check how many adv files already exist
    adv_dir = results_dir / "adversarial_examples"
    existing_adv = len(list(adv_dir.glob("*.npz"))) if adv_dir.exists() else 0
    expected_adv = 3 * 2 * 3  # 3 surrogates x 2 attacks x 3 epsilons = 18
    if existing_adv >= expected_adv:
        logger.info("[Step 3/5] SKIPPED — %d/%d adversarial example files exist",
                    existing_adv, expected_adv)
    else:
        logger.info("[Step 3/5] Found %d/%d files, generating remaining...",
                    existing_adv, expected_adv)
        from generate_adversarial import main as gen_main
        gen_main()
        logger.info("[Step 3/5] DONE in %.1f min", (time.time() - t3) / 60)

    # Step 4: Evaluate transfer
    logger.info("\n[Step 4/5] Evaluating adversarial transfer (with 1000-iter bootstrap CIs)...")
    t4 = time.time()
    full_results = results_dir / "transfer_results_full.json"
    if full_results.exists():
        import json
        with full_results.open() as f:
            existing_rows = json.load(f)
        if len(existing_rows) >= 72:
            logger.info("[Step 4/5] SKIPPED — %d rows already computed", len(existing_rows))
        else:
            logger.info("[Step 4/5] Found %d/72 rows, computing remaining...", len(existing_rows))
            from evaluate_transfer import main as eval_main
            eval_main()
            logger.info("[Step 4/5] DONE in %.1f min", (time.time() - t4) / 60)
    else:
        from evaluate_transfer import main as eval_main
        eval_main()
        logger.info("[Step 4/5] DONE in %.1f min", (time.time() - t4) / 60)

    # Step 5: Build transfer matrix
    logger.info("\n[Step 5/5] Building transfer matrix...")
    from build_transfer_matrix import main as matrix_main
    matrix_main()
    logger.info("[Step 5/5] DONE")

    total_min = (time.time() - t_start) / 60
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 COMPLETE in %.1f min", total_min)
    logger.info("Results in: %s", results_dir)
    logger.info("  transfer_matrix.csv   — 72-row transfer table")
    logger.info("  transfer_matrix_summary.json — per-target statistics")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
