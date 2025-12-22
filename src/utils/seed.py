"""
===============================================================================
Script Name   : seed.py
Description   : Manages global random seeds for reproducibility.
                Function: set_global_seed
                - Sets the seed for Python's built-in `random` module.
                - Sets the seed for `numpy`.
                - Sets the seed for `torch` (CPU and CUDA).
                - Optionally configures PyTorch cuDNN to be deterministic (slower but strictly reproducible).

Usage:
    Called at the beginning of training and attack scripts to ensure 
    consistent model initialization and data splitting.

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

# src/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # torch will come later for deep models
    torch = None


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Set global random seeds for Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed : int
        Seed value.
    deterministic_torch : bool
        If True and torch is available, set deterministic flags
        (slower but more reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
