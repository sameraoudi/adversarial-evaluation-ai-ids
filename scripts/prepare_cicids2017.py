"""
===============================================================================
Script Name   : prepare_cicids2017.py
Description   : Executes the complete preprocessing pipeline for the CICIDS2017 dataset.
                This includes:
                1. Loading raw CSV flows from 'data/raw/cicids2017'.
                2. Cleaning data (removing Infinity/NaN values).
                3. Normalizing features using Min-Max Scaling (0-1 range).
                4. Performing a stratified train (70%), validation (10%), and test (20%) split.
                5. Saving processed artifacts (scaler, label mappings) and split data (.npz).

Usage:
    python scripts/prepare_cicids2017.py

Inputs:
    data/raw/cicids2017/CSVs/MachineLearningCVE/*.csv (Raw Traffic Data)

Outputs:
    - data/processed/cicids2017/splits/train.npz
    - data/processed/cicids2017/splits/val.npz
    - data/processed/cicids2017/splits/test.npz
    - data/artifacts/cicids2017/scaler.pkl

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Nov-22

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.

Dependencies:
    - src.data.preprocess (pandas, numpy, scikit-learn)
===============================================================================
"""

from src.data.preprocess import prepare_dataset_cicids_style

if __name__ == "__main__":
    prepare_dataset_cicids_style("cicids2017")
