"""
===============================================================================
Script Name   : run_attacks_cicids2017.py
Description   : Orchestrates the adversarial robustness evaluation on the CICIDS2017 dataset.  
                This script implements the "Black-Box Transfer" threat model:
                1. Loads the pre-trained Surrogate Model (MLP).
                2. Generates adversarial examples using:
                   - Gradient-based attacks: FGSM, PGD (L-inf norm).
                   - Black-box optimization attacks: ZOO, HopSkipJump (HSJA).
                3. Applies Semantic Constraints (non-negativity, monotonicity) to valid perturbations.
                4. "Transfers" these examples to the Target Models (RF, LR, MLP, CNN-1D).
                5. Calculates and logs Attack Success Rate (ASR) and Robust Accuracy.

Usage:
    Run from the project root:
    $ python scripts/run_attacks_cicids2017.py

Inputs:
    - configs/attacks/cicids2017_whitebox_transfer.yaml (Attack parameters)
    - models/cicids2017/surrogate/surrogate_mlp_best.pt (Source of gradients)
    - models/cicids2017/deep/* (Target Deep Models)
    - models/cicids2017/baselines/* (Target Classical Models)
    - data/processed/cicids2017/splits/test.npz (Test Data)

Outputs:
    - outputs/cicids2017/adv/runs/whitebox_transfer/fgsm_linf.csv
    - outputs/cicids2017/adv/runs/whitebox_transfer/pgd_linf.csv
    - outputs/cicids2017/adv/runs/whitebox_transfer/hsja_linf.csv
    - outputs/cicids2017/adv/runs/whitebox_transfer/zoo_linf.csv
    - logs/cicids2017/whitebox_transfer.log

Dependencies:
    - src.attacks.orchestrator
    - src.utils
    - pyyaml

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Nov-22

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import yaml

from src.attacks.orchestrator import AttackJob, run_whitebox_and_transfer_suite
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_global_seed


def load_attack_config() -> Dict[str, Any]:
    """
    Load attack configuration from configs/attacks/cicids2017_whitebox_transfer.yaml.

    The config file defines the suite of attacks to run.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "attacks" / "cicids2017_whitebox_transfer.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Attack config not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_jobs(cfg: Dict[str, Any]) -> List[AttackJob]:
    """
    Build a list of AttackJob instances from the YAML configuration.

    Parses entries for FGSM, PGD, ZOO, and HSJA. 
    Crucially, it handles 'max_samples' to limit the evaluation scope for 
    computationally expensive attacks (like ZOO/HSJA) compared to cheap ones (FGSM).

    Args:
        cfg: Dictionary loaded from the YAML file.

    Returns:
        List[AttackJob]: A list of configured attack jobs ready for the orchestrator.
    """
    jobs: List[AttackJob] = []
    attack_entries = cfg.get("attacks", [])
    
    # Default global sample limit (usually 20,000 for gradient attacks)
    global_max = int(cfg.get("max_samples", 20000))

    for entry in attack_entries:
        attack_type = str(entry.get("type", "fgsm")).lower()
        norm = str(entry.get("norm", "linf")).lower()
        eps_list = [float(e) for e in entry.get("eps_list", [0.01, 0.03, 0.05])]

        num_steps = int(entry.get("num_steps", 40))
        step_size = float(entry.get("step_size", 0.01))

        # IMPORTANT: Allow per-attack override.
        # e.g., ZOO/HSJA might run on only 1,000 samples due to time constraints.
        max_samples = int(entry.get("max_samples", global_max))

        jobs.append(
            AttackJob(
                attack_type=attack_type,
                norm=norm,
                eps_list=eps_list,
                num_steps=num_steps,
                step_size=step_size,
                max_samples=max_samples,
            )
        )

    return jobs


if __name__ == "__main__":
    # 1. Load Configuration
    cfg = load_attack_config()

    dataset_name = cfg.get("dataset_name", "cicids2017")
    run_name = cfg.get("run_name", "whitebox_transfer")
    seed = int(cfg.get("seed", 42))

    # 2. Setup Logging & Seeding
    configure_logging(dataset_name, run_name=run_name)
    set_global_seed(seed)

    # 3. Build Attack Jobs
    jobs = build_jobs(cfg)

    # 4. Execute Attack Suite
    # Iterates through every defined attack (FGSM -> PGD -> HSJA -> ZOO)
    # and runs the full generation-transfer-evaluation loop.
    for job in jobs:
        run_whitebox_and_transfer_suite(
            dataset_name=dataset_name,
            job=job,
            run_name=run_name,
            seed=seed,
        )
