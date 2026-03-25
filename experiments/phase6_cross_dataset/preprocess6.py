"""
Phase 6: Dataset Preprocessing

Handles three new datasets with their idiosyncrasies:
  - CICIDS2018: same CICFlowMeter features as 2017 but abbreviated column names.
                Column mapping applied to produce 2017-canonical feature names.
                Binary labels (Benign=0, Attack=1). ~6.6M rows, capped at 1.5M.
  - UNSW-NB15:  Argus/Bro-IDS features (incompatible with CICIDS).
                Categorical columns (proto, service, state) label-encoded.
                Uses provided train/test split files directly. ~257K total.
  - CICIoT2023: IoT-specific features (incompatible with CICIDS).
                All numeric. Sampled from 63 files to ~1M rows.

Cross-dataset transfer alignment (CICIDS2017 ↔ CICIDS2018):
  - CICIDS2018 is preprocessed with 2017-canonical column names.
  - CICIDS2017's 2 unique features (Source Port, Fwd Header Length.1) are absent in 2018.
  - Both datasets share 78 features for cross-dataset transfer.
  - Each gets its OWN normalization statistics — no leakage.
"""
from __future__ import annotations

import glob
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger("phase6.preprocess")

# ---------------------------------------------------------------------------
# CICIDS2018 → CICIDS2017 column name mapping
# ---------------------------------------------------------------------------

# Maps abbreviated 2018 column names to 2017 canonical names.
# The 28 columns with identical names in both datasets do NOT appear here.
COL_MAP_2018_TO_2017 = {
    "ACK Flag Cnt":       "ACK Flag Count",
    "Bwd Blk Rate Avg":   "Bwd Avg Bulk Rate",
    "Bwd Byts/b Avg":     "Bwd Avg Bytes/Bulk",
    "Bwd Header Len":     "Bwd Header Length",
    "Bwd IAT Tot":        "Bwd IAT Total",
    "Bwd Pkt Len Max":    "Bwd Packet Length Max",
    "Bwd Pkt Len Mean":   "Bwd Packet Length Mean",
    "Bwd Pkt Len Min":    "Bwd Packet Length Min",
    "Bwd Pkt Len Std":    "Bwd Packet Length Std",
    "Bwd Pkts/b Avg":     "Bwd Avg Packets/Bulk",
    "Bwd Pkts/s":         "Bwd Packets/s",
    "Bwd Seg Size Avg":   "Avg Bwd Segment Size",
    "Dst Port":           "Destination Port",
    "ECE Flag Cnt":       "ECE Flag Count",
    "FIN Flag Cnt":       "FIN Flag Count",
    "Flow Byts/s":        "Flow Bytes/s",
    "Flow Pkts/s":        "Flow Packets/s",
    "Fwd Act Data Pkts":  "act_data_pkt_fwd",
    "Fwd Blk Rate Avg":   "Fwd Avg Bulk Rate",
    "Fwd Byts/b Avg":     "Fwd Avg Bytes/Bulk",
    "Fwd Header Len":     "Fwd Header Length",
    "Fwd IAT Tot":        "Fwd IAT Total",
    "Fwd Pkt Len Max":    "Fwd Packet Length Max",
    "Fwd Pkt Len Mean":   "Fwd Packet Length Mean",
    "Fwd Pkt Len Min":    "Fwd Packet Length Min",
    "Fwd Pkt Len Std":    "Fwd Packet Length Std",
    "Fwd Pkts/b Avg":     "Fwd Avg Packets/Bulk",
    "Fwd Pkts/s":         "Fwd Packets/s",
    "Fwd Seg Size Avg":   "Avg Fwd Segment Size",
    "Fwd Seg Size Min":   "min_seg_size_forward",
    "Init Bwd Win Byts":  "Init_Win_bytes_backward",
    "Init Fwd Win Byts":  "Init_Win_bytes_forward",
    "PSH Flag Cnt":       "PSH Flag Count",
    "Pkt Len Max":        "Max Packet Length",
    "Pkt Len Mean":       "Packet Length Mean",
    "Pkt Len Min":        "Min Packet Length",
    "Pkt Len Std":        "Packet Length Std",
    "Pkt Len Var":        "Packet Length Variance",
    "Pkt Size Avg":       "Average Packet Size",
    "RST Flag Cnt":       "RST Flag Count",
    "SYN Flag Cnt":       "SYN Flag Count",
    "Subflow Bwd Byts":   "Subflow Bwd Bytes",
    "Subflow Bwd Pkts":   "Subflow Bwd Packets",
    "Subflow Fwd Byts":   "Subflow Fwd Bytes",
    "Subflow Fwd Pkts":   "Subflow Fwd Packets",
    "Tot Bwd Pkts":       "Total Backward Packets",
    "Tot Fwd Pkts":       "Total Fwd Packets",
    "TotLen Bwd Pkts":    "Total Length of Bwd Packets",
    "TotLen Fwd Pkts":    "Total Length of Fwd Packets",
    "URG Flag Cnt":       "URG Flag Count",
}

# 2017-only features (absent from 2018 — not shared in cross-dataset transfer)
CICIDS2017_ONLY_FEATURES = {"Source Port", "Fwd Header Length.1"}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=seed
    )
    logger.info("Split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}


def _save_splits(
    out_dir: Path,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    scaler: MinMaxScaler,
    feature_names: List[str],
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (X, y) in splits.items():
        np.savez_compressed(out_dir / f"{split_name}.npz", X=X, y=y)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved splits + scaler + metadata to %s", out_dir)


def load_splits(out_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    splits = {}
    for name in ("train", "val", "test"):
        d = np.load(out_dir / f"{name}.npz")
        splits[name] = (d["X"], d["y"])
    return splits


# ---------------------------------------------------------------------------
# CICIDS2018 preprocessing
# ---------------------------------------------------------------------------

def preprocess_cicids2018(
    raw_dir: str,
    out_dir: Path,
    cap_rows: int = 1_500_000,
    seed: int = 42,
) -> Dict:
    """
    Load all 7 CICIDS2018 CSVs, rename columns to CICIDS2017 canonical names,
    binary-label (Benign=0, Attack=1), stratified split + MinMaxScaler.
    Saves splits + scaler to out_dir.
    Returns metadata dict.
    """
    if (out_dir / "train.npz").exists():
        logger.info("CICIDS2018 splits already exist at %s — skipping", out_dir)
        meta = json.loads((out_dir / "metadata.json").read_text())
        return meta

    logger.info("=== Preprocessing CICIDS2018 ===")
    csvs = sorted(glob.glob(str(Path(raw_dir) / "*.csv")))
    logger.info("Found %d CSV files", len(csvs))

    dfs = []
    for f in csvs:
        logger.info("  Loading %s ...", Path(f).name)
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Raw shape: %s", df.shape)

    # Drop header-contamination rows (rows where Label column literally says "Label")
    mask_header = df["Label"].astype(str).str.strip().str.lower() == "label"
    n_contam = mask_header.sum()
    if n_contam > 0:
        logger.info("Dropping %d header-contamination rows", n_contam)
        df = df[~mask_header].copy()

    # Drop Timestamp
    df = df.drop(columns=["Timestamp"], errors="ignore")

    # Rename columns to 2017 canonical names
    df = df.rename(columns=COL_MAP_2018_TO_2017)
    logger.info("Columns after renaming: %d", len(df.columns))

    # Extract label
    labels_raw = df["Label"].astype(str).str.strip()
    benign_mask = labels_raw.str.lower() == "benign"
    y = np.where(benign_mask, 0, 1).astype(np.int64)
    df = df.drop(columns=["Label"])

    # Convert features to numeric, drop all-NaN columns
    feature_cols = list(df.columns)
    df_num = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    all_nan_cols = df_num.columns[df_num.isna().all()].tolist()
    if all_nan_cols:
        logger.info("Dropping all-NaN cols: %s", all_nan_cols)
        df_num = df_num.drop(columns=all_nan_cols)

    # Drop inf/NaN rows
    finite_mask = np.isfinite(df_num.to_numpy(dtype=np.float64)).all(axis=1)
    label_ok = pd.Series(y).notna().to_numpy()
    valid = finite_mask & label_ok
    X = df_num.to_numpy(dtype=np.float32)[valid]
    y = y[valid]
    feature_names = list(df_num.columns)
    logger.info("After cleaning: %d rows, %d features", len(y), len(feature_names))

    # Class distribution
    n_benign = int((y == 0).sum())
    n_attack = int((y == 1).sum())
    logger.info("Class dist: Benign=%d (%.1f%%), Attack=%d (%.1f%%)",
                n_benign, 100*n_benign/len(y), n_attack, 100*n_attack/len(y))

    # Stratified subsample if too large
    if len(y) > cap_rows:
        logger.info("Subsampling to %d rows (from %d)", cap_rows, len(y))
        rng = np.random.default_rng(seed)
        # Proportional sampling per class
        idx_b = np.where(y == 0)[0]
        idx_a = np.where(y == 1)[0]
        frac = cap_rows / len(y)
        n_b = min(len(idx_b), int(frac * len(idx_b)))
        n_a = min(len(idx_a), cap_rows - n_b)
        sel_b = rng.choice(idx_b, size=n_b, replace=False)
        sel_a = rng.choice(idx_a, size=n_a, replace=False)
        sel = np.sort(np.concatenate([sel_b, sel_a]))
        X, y = X[sel], y[sel]
        logger.info("After subsample: %d rows", len(y))

    # Split
    splits_raw = _stratified_split(X, y, seed=seed)

    # Fit scaler on train only
    scaler = MinMaxScaler()
    scaler.fit(splits_raw["train"][0])
    splits = {
        k: (scaler.transform(v[0]).astype(np.float32), v[1])
        for k, v in splits_raw.items()
    }

    meta = {
        "dataset": "cicids2018",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_train": int(len(splits["train"][1])),
        "n_val": int(len(splits["val"][1])),
        "n_test": int(len(splits["test"][1])),
        "class_dist_full": {"benign": n_benign, "attack": n_attack},
        "label_mapping": {"0": "Benign", "1": "Attack"},
        "cross_transfer_compatible_with": "cicids2017",
        "note": "Columns remapped to CICIDS2017 canonical names. "
                "CICIDS2017-only features (Source Port, Fwd Header Length.1) absent.",
    }
    _save_splits(out_dir, splits, scaler, feature_names, meta)
    return meta


# ---------------------------------------------------------------------------
# UNSW-NB15 preprocessing
# ---------------------------------------------------------------------------

def preprocess_unsw_nb15(
    train_file: str,
    test_file: str,
    out_dir: Path,
    seed: int = 42,
) -> Dict:
    """
    Load UNSW-NB15 training-set.csv + testing-set.csv.
    Drop id, attack_cat. Label-encode categorical columns.
    Binary label (0=Normal, 1=Attack). MinMaxScaler on train.
    """
    if (out_dir / "train.npz").exists():
        logger.info("UNSW-NB15 splits already exist at %s — skipping", out_dir)
        return json.loads((out_dir / "metadata.json").read_text())

    logger.info("=== Preprocessing UNSW-NB15 ===")
    df_train = pd.read_csv(train_file, low_memory=False)
    df_test  = pd.read_csv(test_file,  low_memory=False)
    logger.info("Raw train: %s  test: %s", df_train.shape, df_test.shape)

    # Combine for consistent encoding, then split back
    df_train["_split"] = "train"
    df_test["_split"]  = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop unused columns
    drop_cols = ["id", "attack_cat", "_split"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Label column: UNSW uses 'label' (0=normal, 1=attack)
    if "label" in df.columns:
        label_col = "label"
    else:
        raise KeyError("Expected 'label' column in UNSW-NB15")

    y = df[label_col].to_numpy(dtype=np.int64)
    df = df.drop(columns=[label_col])

    # Encode categorical columns
    cat_cols = [c for c in ["proto", "service", "state"] if c in df.columns]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = {v: int(i) for i, v in enumerate(le.classes_)}
    logger.info("Encoded %d categorical columns: %s", len(cat_cols), cat_cols)

    # Convert remaining to numeric
    feature_cols = list(df.columns)
    df_num = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    all_nan_cols = df_num.columns[df_num.isna().all()].tolist()
    if all_nan_cols:
        logger.info("Dropping all-NaN cols: %s", all_nan_cols)
        df_num = df_num.drop(columns=all_nan_cols)
    feature_names = list(df_num.columns)

    # Drop invalid rows
    X_all = df_num.to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(X_all.astype(np.float64)).all(axis=1)
    X_all = X_all[finite_mask]
    y = y[finite_mask]
    logger.info("After cleaning: %d rows, %d features", len(y), len(feature_names))

    # Reconstruct train/test from original sizes (aligned to finite_mask)
    n_train_raw = len(df_train)
    # Rebuild split markers before filtering
    all_splits_marker = np.array(
        ["train"] * len(df_train) + ["test"] * len(df_test)
    )[finite_mask]
    train_mask = all_splits_marker == "train"
    test_mask  = all_splits_marker == "test"

    X_train_raw = X_all[train_mask]
    y_train_raw = y[train_mask]
    X_test_raw  = X_all[test_mask]
    y_test_raw  = y[test_mask]

    # Carve out val from train (10%)
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_raw, y_train_raw, test_size=0.111, stratify=y_train_raw,
        random_state=seed
    )  # 0.111 * 0.9 ≈ 0.1 of total

    # Scaler fit on train
    scaler = MinMaxScaler()
    scaler.fit(X_train_raw)

    splits = {
        "train": (scaler.transform(X_train_raw).astype(np.float32), y_train_raw),
        "val":   (scaler.transform(X_val_raw).astype(np.float32),   y_val_raw),
        "test":  (scaler.transform(X_test_raw).astype(np.float32),  y_test_raw),
    }

    n_normal = int((y == 0).sum())
    n_attack = int((y == 1).sum())
    meta = {
        "dataset": "unsw_nb15",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "categorical_encoders": label_encoders,
        "n_train": int(len(splits["train"][1])),
        "n_val":   int(len(splits["val"][1])),
        "n_test":  int(len(splits["test"][1])),
        "class_dist_full": {"normal": n_normal, "attack": n_attack},
        "label_mapping": {"0": "Normal", "1": "Attack"},
        "cross_transfer_compatible_with": "none",
        "note": "Argus/Bro-IDS features; NOT compatible with CICFlowMeter. "
                "Categorical cols encoded: " + str(cat_cols),
    }
    _save_splits(out_dir, splits, scaler, feature_names, meta)
    return meta


# ---------------------------------------------------------------------------
# CIC-IoT-2023 preprocessing
# ---------------------------------------------------------------------------

def preprocess_ciciot2023(
    raw_dir: str,
    out_dir: Path,
    sample_rows: int = 1_000_000,
    seed: int = 42,
) -> Dict:
    """
    Load 63 CIC-IoT-2023 Merged CSV files, stratified sample to sample_rows,
    binary label (BENIGN=0, Attack=1). MinMaxScaler on train.
    """
    if (out_dir / "train.npz").exists():
        logger.info("CICIoT2023 splits already exist at %s — skipping", out_dir)
        return json.loads((out_dir / "metadata.json").read_text())

    logger.info("=== Preprocessing CIC-IoT-2023 ===")
    csvs = sorted(glob.glob(str(Path(raw_dir) / "Merged*.csv")))
    logger.info("Found %d CSV files; will sample %d total rows", len(csvs), sample_rows)

    # Estimate rows per class proportionally by scanning one file
    probe = pd.read_csv(csvs[0], usecols=["Label"])
    probe_dist = probe["Label"].value_counts(normalize=True)
    logger.info("Probe label fractions (file 0): %s", probe_dist.to_dict())

    # Load with chunking: collect samples proportionally
    rng = np.random.default_rng(seed)
    rows_per_file = max(1, sample_rows // len(csvs))
    all_Xs, all_ys = [], []

    first_cols = None
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            logger.warning("Skipping %s: %s", Path(f).name, e)
            continue

        if first_cols is None:
            first_cols = list(df.columns)
        df.columns = [c.strip() for c in df.columns]

        if "Label" not in df.columns:
            logger.warning("No Label in %s — skipping", Path(f).name)
            continue

        # Sample from this file
        n = min(rows_per_file, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        df = df.iloc[idx].copy()

        labels = df["Label"].astype(str).str.strip()
        y_file = np.where(labels.str.upper() == "BENIGN", 0, 1).astype(np.int64)
        df = df.drop(columns=["Label"])

        # All features should be numeric already
        df_num = df.apply(pd.to_numeric, errors="coerce")
        X_file = df_num.to_numpy(dtype=np.float32)
        finite_mask = np.isfinite(X_file.astype(np.float64)).all(axis=1)
        X_file = X_file[finite_mask]
        y_file = y_file[finite_mask]
        all_Xs.append(X_file)
        all_ys.append(y_file)

    X = np.vstack(all_Xs)
    y = np.concatenate(all_ys)
    feature_names = [c.strip() for c in first_cols if c.strip() != "Label"]
    logger.info("After loading: %d rows, %d features", len(y), len(feature_names))

    n_benign = int((y == 0).sum())
    n_attack = int((y == 1).sum())
    logger.info("Class dist: Benign=%d (%.1f%%), Attack=%d (%.1f%%)",
                n_benign, 100*n_benign/len(y), n_attack, 100*n_attack/len(y))

    # Split
    splits_raw = _stratified_split(X, y, seed=seed)

    # Scaler
    scaler = MinMaxScaler()
    scaler.fit(splits_raw["train"][0])
    splits = {
        k: (scaler.transform(v[0]).astype(np.float32), v[1])
        for k, v in splits_raw.items()
    }

    meta = {
        "dataset": "ciciot2023",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_train": int(len(splits["train"][1])),
        "n_val":   int(len(splits["val"][1])),
        "n_test":  int(len(splits["test"][1])),
        "class_dist_full": {"benign": n_benign, "attack": n_attack},
        "label_mapping": {"0": "Benign", "1": "Attack"},
        "sample_rows": sample_rows,
        "cross_transfer_compatible_with": "none",
        "note": "IoT-specific features (CICFlowMeter-derived but different schema). "
                "NOT compatible with standard CICIDS CICFlowMeter.",
    }
    _save_splits(out_dir, splits, scaler, feature_names, meta)
    return meta


# ---------------------------------------------------------------------------
# Feature compatibility report
# ---------------------------------------------------------------------------

def build_feature_compatibility_report(
    meta2018: dict,
    meta_unsw: dict,
    meta_iot: dict,
    features_2017: List[str],
    out_path: Path,
) -> dict:
    """
    Describe feature compatibility between CICIDS2017 and the three new datasets.
    """
    feats18  = set(meta2018["feature_names"])
    feats17  = set(features_2017)
    common18 = sorted(feats17 & feats18)
    only17   = sorted(feats17 - feats18)
    only18   = sorted(feats18 - feats17)

    report = {
        "cicids2017_vs_cicids2018": {
            "cicids2017_n_features": len(feats17),
            "cicids2018_n_features": len(feats18),
            "common": len(common18),
            "only_in_2017": only17,
            "only_in_2018": only18,
            "cross_transfer_feasible": True,
            "cross_transfer_feature_count": len(common18),
            "note": "Same CICFlowMeter extractor, different column naming conventions. "
                    "CICIDS2018 columns remapped to 2017 canonical names. "
                    "Source Port and Fwd Header Length.1 are 2017-only.",
        },
        "cicids2017_vs_unsw_nb15": {
            "cicids2017_n_features": len(feats17),
            "unsw_nb15_n_features": meta_unsw["n_features"],
            "cross_transfer_feasible": False,
            "note": "Completely different feature extractors: CICFlowMeter (2017) vs "
                    "Argus/Bro-IDS (UNSW-NB15). No meaningful cross-transfer.",
        },
        "cicids2017_vs_ciciot2023": {
            "cicids2017_n_features": len(feats17),
            "ciciot2023_n_features": meta_iot["n_features"],
            "cross_transfer_feasible": False,
            "note": "Different feature schemas. CICIoT2023 uses a distinct IoT-oriented "
                    "feature set not directly compatible with standard CICFlowMeter.",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Feature compatibility report saved to %s", out_path)
    return report
