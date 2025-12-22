"""
===============================================================================
Script Name   : prepare_cicids2018_friday.py
Description   : Executes the preprocessing pipeline specifically for the CICIDS2018 (Friday slice) dataset.
                This script is used to generate the "Out-of-Distribution" test set for 
                cross-dataset robustness evaluation.
                
                It handles:
                1. Loading the Friday-02-03-2018 traffic CSV.
                2. Cleaning and Label Encoding (converting specific 2018 labels to binary or mapped formats).
                3. Normalizing features using the same scaling logic as the source dataset.
                4. Saving processed splits for transferability testing.

Usage:
    Run from the project root:
    $ python scripts/prepare_cicids2018_friday.py

Inputs:
    data/raw/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv

Outputs:
    - data/processed/cicids2018_friday/splits/train.npz
    - data/processed/cicids2018_friday/splits/val.npz
    - data/processed/cicids2018_friday/splits/test.npz
    - data/artifacts/cicids2018_friday/scaler.pkl

Dependencies:
    - src.data.preprocess_cicids2018
    - src.utils.logging_utils

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

from src.data.preprocess_cicids2018 import (
    CICIDS2018Config,
    prepare_cicids2018_friday_slice,
)
from src.utils.logging_utils import configure_logging


if __name__ == "__main__":
    cfg = CICIDS2018Config(
        # Adjust if your relative path is slightly different
        raw_rel_path="data/raw/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        dataset_name="cicids2018_friday",
    )

    configure_logging(dataset_name=cfg.dataset_name, run_name="preprocess")
    prepare_cicids2018_friday_slice(cfg)
