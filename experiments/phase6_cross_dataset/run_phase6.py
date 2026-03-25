"""
Phase 6: Cross-Dataset Evaluation
Addresses Reviewer R6: "Cross-dataset evaluation used meaningless baselines
 (CICIDS2017-trained models on CICIDS2018 without retraining = 4-8% accuracy)"

Objectives:
  (A) In-distribution robustness: train + attack + evaluate on each dataset independently.
  (B) Cross-dataset transfer: CICIDS2017 ↔ CICIDS2018 (compatible CICFlowMeter features).

Datasets:
  - CICIDS2018    (7 CSV files, ~6.6M rows → cap 1.5M; CICFlowMeter; compatible with 2017)
  - UNSW-NB15     (training-set + testing-set; Argus/Bro-IDS; independent evaluation only)
  - CIC-IoT-2023  (63 Merged CSV files, ~45M rows → sample 1M; IoT features; independent)

Pipeline (Steps 1–6):
  1. Preprocess all three new datasets
  2. Train target models (RF, LR, MLP, CNN1D) on each dataset
  3. Train surrogate models (LR-Surr, GBT-Distill, CNN-v2) on each dataset
  4. In-distribution adversarial evaluation (72-row matrix per dataset)
  5. Cross-dataset transfer (CICIDS2017 ↔ CICIDS2018)
  6. Save all results + feature compatibility report

Usage:
    cd <project_root>
    python3.10 experiments/phase6_cross_dataset/run_phase6.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import joblib

from preprocess6 import (
    preprocess_cicids2018,
    preprocess_unsw_nb15,
    preprocess_ciciot2023,
    build_feature_compatibility_report,
    load_splits,
    CICIDS2017_ONLY_FEATURES,
)
from models6 import (
    train_rf, train_lr, train_mlp, train_cnn1d,
    train_lr_surrogate, train_gbt_distill, train_cnn_v2,
    MLP, CNN1D, LinearSurrogate, GBTDistillNet, CNNv2Surrogate,
    sklearn_predict, torch_predict,
)
from attacks6 import (
    eval_transfer_matrix,
    eval_cross_dataset_transfer,
    stratified_eval_subset,
)

from src.utils.paths import get_models_root, get_splits_dir, get_artifacts_dir
from src.data.io import load_npz

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
log_path = RESULTS_DIR / "phase6_run.log"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ],
)
logger = logging.getLogger("phase6.runner")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Step 1: Preprocessing
# ---------------------------------------------------------------------------

def step1_preprocess(cfg: dict) -> dict:
    logger.info("\n[Step 1/6] Preprocessing datasets...")

    proc_dir = PROJECT_ROOT / "data" / "processed"

    meta2018 = preprocess_cicids2018(
        raw_dir=str(PROJECT_ROOT / cfg["datasets"]["cicids2018"]["raw_dir"]),
        out_dir=proc_dir / "cicids2018" / "splits",
        cap_rows=cfg["datasets"]["cicids2018"].get("cap_rows", 1_500_000),
        seed=cfg["seed"],
    )

    meta_unsw = preprocess_unsw_nb15(
        train_file=str(PROJECT_ROOT / cfg["datasets"]["unsw_nb15"]["train_file"]),
        test_file=str(PROJECT_ROOT / cfg["datasets"]["unsw_nb15"]["test_file"]),
        out_dir=proc_dir / "unsw_nb15" / "splits",
        seed=cfg["seed"],
    )

    meta_iot = preprocess_ciciot2023(
        raw_dir=str(PROJECT_ROOT / cfg["datasets"]["ciciot2023"]["raw_dir"]),
        out_dir=proc_dir / "ciciot2023" / "splits",
        sample_rows=cfg["datasets"]["ciciot2023"].get("sample_rows", 1_000_000),
        seed=cfg["seed"],
    )

    # Load CICIDS2017 feature list for compatibility report
    features_2017_path = get_artifacts_dir("cicids2017") / "features.csv"
    if features_2017_path.exists():
        features_2017 = pd.read_csv(features_2017_path)["feature"].tolist()
    else:
        # Fallback: read from processed splits
        d = np.load(get_splits_dir("cicids2017") / "train.npz")
        features_2017 = [f"feat_{i}" for i in range(d["X"].shape[1])]

    compat_report = build_feature_compatibility_report(
        meta2018=meta2018,
        meta_unsw=meta_unsw,
        meta_iot=meta_iot,
        features_2017=features_2017,
        out_path=RESULTS_DIR / "preprocessing" / "feature_compatibility_report.json",
    )

    # Save individual metadata
    for meta, name in [(meta2018, "cicids2018"), (meta_unsw, "unsw_nb15"),
                       (meta_iot, "ciciot2023")]:
        p = RESULTS_DIR / "preprocessing" / f"{name}_metadata.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2, default=str))

    logger.info("[Step 1/6] DONE")
    return {
        "meta2018": meta2018, "meta_unsw": meta_unsw, "meta_iot": meta_iot,
        "features_2017": features_2017, "compat_report": compat_report,
        "proc_dir": proc_dir,
    }


# ---------------------------------------------------------------------------
# Step 2: Train target models
# ---------------------------------------------------------------------------

def _load_dataset_splits(proc_dir: Path, name: str):
    splits = load_splits(proc_dir / name / "splits")
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]
    input_dim  = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    return X_train, y_train, X_val, y_val, X_test, y_test, input_dim, num_classes


def _train_targets_for_dataset(
    dataset_name: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    input_dim: int, num_classes: int,
    cfg: dict, device: torch.device,
    model_dir: Path,
) -> dict:
    logger.info("--- Training targets for %s (n_train=%d, dim=%d, classes=%d) ---",
                dataset_name, len(y_train), input_dim, num_classes)

    seed = cfg["seed"]

    # RF
    rf_info = train_rf(
        X_train, y_train, X_test, y_test,
        n_estimators=cfg["rf"]["n_estimators"],
        seed=seed, save_path=model_dir / "baselines" / "rf_model.pkl"
    )
    logger.info("[%s] RF  test_acc=%.4f", dataset_name, rf_info["test_acc"])

    # LR
    lr_info = train_lr(
        X_train, y_train, X_test, y_test,
        C=cfg["lr"]["C"], seed=seed,
        save_path=model_dir / "baselines" / "lr_model.pkl"
    )
    logger.info("[%s] LR  test_acc=%.4f", dataset_name, lr_info["test_acc"])

    # Check for <50% accuracy
    for name, info in [("RF", rf_info), ("LR", lr_info)]:
        if info["test_acc"] < 0.50:
            logger.warning("[%s] %s test_acc=%.4f < 0.50 — FLAGGED: unreliable baseline",
                           dataset_name, name, info["test_acc"])

    # MLP
    mlp_info = train_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_dim=input_dim, num_classes=num_classes,
        hidden_dims=cfg["mlp"]["hidden_dims"],
        dropout=cfg["mlp"]["dropout"],
        lr=cfg["mlp"]["lr"],
        weight_decay=cfg["mlp"].get("weight_decay", 1e-4),
        batch_size=cfg["mlp"]["batch_size"],
        max_epochs=cfg["mlp"]["max_epochs"],
        patience=cfg["mlp"]["early_stopping_patience"],
        seed=seed, device=device,
        save_path=model_dir / "deep" / "mlp_best.pt",
    )
    logger.info("[%s] MLP  test_acc=%.4f", dataset_name, mlp_info["test_acc"])

    # CNN1D
    cnn_info = train_cnn1d(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_dim=input_dim, num_classes=num_classes,
        channels=cfg["cnn1d"]["channels"],
        kernel_sizes=cfg["cnn1d"]["kernel_sizes"],
        dropout=cfg["cnn1d"]["dropout"],
        lr=cfg["cnn1d"]["lr"],
        weight_decay=cfg["cnn1d"].get("weight_decay", 1e-4),
        batch_size=cfg["cnn1d"]["batch_size"],
        max_epochs=cfg["cnn1d"]["max_epochs"],
        patience=cfg["cnn1d"]["early_stopping_patience"],
        seed=seed, device=device,
        save_path=model_dir / "deep" / "cnn1d_best.pt",
    )
    logger.info("[%s] CNN1D test_acc=%.4f", dataset_name, cnn_info["test_acc"])

    if mlp_info["test_acc"] < 0.50:
        logger.warning("[%s] MLP test_acc=%.4f < 0.50 — FLAGGED", dataset_name, mlp_info["test_acc"])
    if cnn_info["test_acc"] < 0.50:
        logger.warning("[%s] CNN1D test_acc=%.4f < 0.50 — FLAGGED", dataset_name, cnn_info["test_acc"])

    # Save clean accuracy summary
    clean_acc = {
        "dataset": dataset_name,
        "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "input_dim": input_dim, "num_classes": num_classes,
        "RF_test_acc":    rf_info["test_acc"],
        "LR_test_acc":    lr_info["test_acc"],
        "MLP_test_acc":   mlp_info["test_acc"],
        "CNN1D_test_acc": cnn_info["test_acc"],
        "flagged_below_50pct": {
            k: v for k, v in {
                "RF": rf_info["test_acc"], "LR": lr_info["test_acc"],
                "MLP": mlp_info["test_acc"], "CNN1D": cnn_info["test_acc"],
            }.items() if v < 0.5
        },
    }
    p = RESULTS_DIR / "clean_baselines" / f"{dataset_name}_clean_accuracy.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([clean_acc]).to_csv(p, index=False)
    logger.info("Clean accuracy saved to %s", p)

    return {
        "rf": rf_info["model"], "lr": lr_info["model"],
        "mlp": mlp_info["model"], "cnn1d": cnn_info["model"],
        "clean_acc": clean_acc,
    }


def step2_train_targets(cfg: dict, device: torch.device, proc_dir: Path) -> dict:
    logger.info("\n[Step 2/6] Training target models...")
    all_targets = {}

    for ds_name in ["cicids2018", "unsw_nb15", "ciciot2023"]:
        X_tr, y_tr, X_v, y_v, X_te, y_te, dim, nc = _load_dataset_splits(proc_dir, ds_name)
        model_dir = RESULTS_DIR / "models" / ds_name
        targets = _train_targets_for_dataset(
            ds_name, X_tr, y_tr, X_v, y_v, X_te, y_te, dim, nc, cfg, device, model_dir
        )
        all_targets[ds_name] = targets

    logger.info("[Step 2/6] DONE")
    return all_targets


# ---------------------------------------------------------------------------
# Step 3: Train surrogate models
# ---------------------------------------------------------------------------

def _train_surrogates_for_dataset(
    dataset_name: str,
    X_train, y_train, X_val, y_val,
    input_dim: int, num_classes: int,
    cfg: dict, device: torch.device,
    surr_dir: Path,
) -> dict:
    logger.info("--- Training surrogates for %s ---", dataset_name)
    scfg = cfg["surrogates"]
    seed = cfg["seed"]

    # LR-Surrogate
    lr_info = train_lr_surrogate(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim, num_classes=num_classes,
        lr=scfg["lr_surrogate"]["lr"],
        batch_size=scfg["lr_surrogate"]["batch_size"],
        epochs=scfg["lr_surrogate"]["epochs"],
        patience=scfg["lr_surrogate"]["early_stopping_patience"],
        seed=seed, device=device,
        save_path=surr_dir / "lr_surrogate_best.pt",
    )

    # GBT-Distill
    gbt_info = train_gbt_distill(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim, num_classes=num_classes,
        xgb_n_estimators=scfg["gbt_distill"]["xgb_n_estimators"],
        xgb_max_depth=scfg["gbt_distill"]["xgb_max_depth"],
        xgb_lr=scfg["gbt_distill"]["xgb_lr"],
        distill_lr=scfg["gbt_distill"]["distill_lr"],
        distill_batch=scfg["gbt_distill"]["distill_batch_size"],
        distill_epochs=scfg["gbt_distill"]["distill_epochs"],
        distill_patience=scfg["gbt_distill"]["distill_patience"],
        temperature=scfg["gbt_distill"]["temperature"],
        seed=seed, device=device,
        save_path=surr_dir / "gbt_distill_best.pt",
    )

    # CNN-v2
    cnnv2_info = train_cnn_v2(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim, num_classes=num_classes,
        lr=scfg["cnn_v2"]["lr"],
        batch_size=scfg["cnn_v2"]["batch_size"],
        epochs=scfg["cnn_v2"]["epochs"],
        patience=scfg["cnn_v2"]["early_stopping_patience"],
        seed=seed, device=device,
        save_path=surr_dir / "cnnv2_best.pt",
    )

    return {
        "LR-Surr":    lr_info["model"],
        "GBT-Distill": gbt_info["model"],
        "CNN-v2":     cnnv2_info["model"],
    }


def step3_train_surrogates(cfg: dict, device: torch.device, proc_dir: Path) -> dict:
    logger.info("\n[Step 3/6] Training surrogate models...")
    all_surrogates = {}

    for ds_name in ["cicids2018", "unsw_nb15", "ciciot2023"]:
        X_tr, y_tr, X_v, y_v, _, _, dim, nc = _load_dataset_splits(proc_dir, ds_name)
        surr_dir = RESULTS_DIR / "models" / ds_name / "surrogates"
        surrogates = _train_surrogates_for_dataset(
            ds_name, X_tr, y_tr, X_v, y_v, dim, nc, cfg, device, surr_dir
        )
        all_surrogates[ds_name] = surrogates

    logger.info("[Step 3/6] DONE")
    return all_surrogates


# ---------------------------------------------------------------------------
# Step 4: In-distribution adversarial evaluation
# ---------------------------------------------------------------------------

def _make_predict_fns(
    targets: dict, input_dim: int, num_classes: int, device: torch.device,
    cfg: dict, model_dir: Path,
) -> dict:
    """Build unified predict callables for all 4 target models."""
    def _sklearn(m):
        return lambda X: sklearn_predict(m, X)

    def _torch(m):
        return lambda X: torch_predict(m, X, device)

    return {
        "RF":    {"predict": _sklearn(targets["rf"])},
        "LR":    {"predict": _sklearn(targets["lr"])},
        "MLP":   {"predict": _torch(targets["mlp"])},
        "CNN1D": {"predict": _torch(targets["cnn1d"])},
    }


def step4_in_distribution(
    cfg: dict, device: torch.device, proc_dir: Path,
    all_targets: dict, all_surrogates: dict,
) -> dict:
    logger.info("\n[Step 4/6] In-distribution adversarial evaluation...")
    ecfg = cfg["eval"]
    results_all = {}

    for ds_name in ["cicids2018", "unsw_nb15", "ciciot2023"]:
        logger.info("--- In-distribution eval: %s ---", ds_name)
        X_tr, y_tr, X_v, y_v, X_te, y_te, dim, nc = _load_dataset_splits(proc_dir, ds_name)

        # Eval subset
        X_eval, y_eval = stratified_eval_subset(X_te, y_te, n=ecfg["n_eval"],
                                                  seed=cfg["seed"])
        logger.info("[%s] Eval subset: %d samples", ds_name, len(y_eval))

        targets = all_targets[ds_name]
        surrogates = all_surrogates[ds_name]

        model_dir = RESULTS_DIR / "models" / ds_name
        predict_fns = _make_predict_fns(targets, dim, nc, device, cfg, model_dir)

        # Clean accuracy on eval subset (sanity check)
        for tgt_name, tgt in predict_fns.items():
            y_pred = tgt["predict"](X_eval)
            acc = (y_pred == y_eval).mean()
            logger.info("  [%s] %s eval_subset clean_acc=%.4f", ds_name, tgt_name, acc)
            if acc < 0.5:
                logger.warning("  [%s] %s clean_acc < 0.5 — FLAGGED", ds_name, tgt_name)

        results = eval_transfer_matrix(
            dataset_name=ds_name,
            surrogates=surrogates,
            targets=predict_fns,
            X_eval=X_eval,
            y_eval=y_eval,
            eps_list=ecfg["epsilon_levels"],
            attacks=["fgsm", "pgd"],
            pgd_steps=ecfg["pgd_steps"],
            pgd_step_size_ratio=ecfg["pgd_step_size_ratio"],
            device=device,
            out_path=RESULTS_DIR / "in_distribution" / f"{ds_name}_transfer_matrix.csv",
            n_boot=ecfg["bootstrap_iters"],
        )
        results_all[ds_name] = results

    # Build cross-dataset summary (best ASR per target per dataset)
    summary_rows = []
    for ds_name, rows in results_all.items():
        df = pd.DataFrame(rows)
        for tgt in ["RF", "LR", "MLP", "CNN1D"]:
            sub = df[df["target"] == tgt]
            if len(sub) == 0:
                continue
            row = {
                "dataset": ds_name,
                "target": tgt,
                "best_asr": round(sub["asr"].max(), 4),
                "best_surrogate": sub.loc[sub["asr"].idxmax(), "surrogate"],
                "best_attack": sub.loc[sub["asr"].idxmax(), "attack"],
                "best_eps": sub.loc[sub["asr"].idxmax(), "eps"],
                "mean_asr": round(sub["asr"].mean(), 4),
            }
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    p = RESULTS_DIR / "in_distribution" / "in_distribution_summary.csv"
    summary_df.to_csv(p, index=False)
    logger.info("In-distribution summary saved to %s", p)

    logger.info("[Step 4/6] DONE")
    return results_all


# ---------------------------------------------------------------------------
# Step 5: Cross-dataset transfer (CICIDS2017 ↔ CICIDS2018)
# ---------------------------------------------------------------------------

def _load_cicids2017_models(device: torch.device) -> dict:
    """Load existing CICIDS2017 target and surrogate models."""
    models_root = get_models_root("cicids2017")
    splits_dir  = get_splits_dir("cicids2017")

    train_data = load_npz(splits_dir / "train.npz")
    input_dim  = train_data["X"].shape[1]
    num_classes = int(np.max(train_data["y"])) + 1

    targets = {}
    # RF
    rf_path = models_root / "baselines" / "rf" / "rf_model.pkl"
    if rf_path.exists():
        targets["RF"] = {"predict": lambda X, m=joblib.load(rf_path): sklearn_predict(m, X)}
    # LR
    lr_path = models_root / "baselines" / "lr" / "lr_model.pkl"
    if lr_path.exists():
        targets["LR"] = {"predict": lambda X, m=joblib.load(lr_path): sklearn_predict(m, X)}
    # MLP
    mlp_path = models_root / "deep" / "mlp" / "deep_mlp_cnn_best.pt"
    if mlp_path.exists():
        mlp = MLP(input_dim, num_classes, [256, 128], 0.3).to(device)
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp.eval()
        targets["MLP"] = {"predict": lambda X, m=mlp: torch_predict(m, X, device)}
    # CNN1D
    cnn_path = models_root / "deep" / "cnn1d" / "deep_mlp_cnn_best.pt"
    if cnn_path.exists():
        cnn = CNN1D(input_dim, num_classes, [32, 64], [3, 3], 0.3).to(device)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn.eval()
        targets["CNN1D"] = {"predict": lambda X, m=cnn: torch_predict(m, X, device)}

    # Surrogates (from Phase 1)
    surr_dir = Path(__file__).parent.parent / "phase1_surrogate_redesign" / "results" / "surrogates"
    surrogates = {}
    lr_surr_path = surr_dir / "lr_surrogate_best.pt"
    if lr_surr_path.exists():
        s = LinearSurrogate(input_dim, num_classes).to(device)
        s.load_state_dict(torch.load(lr_surr_path, map_location=device))
        s.eval()
        surrogates["LR-Surr"] = s

    gbt_path = surr_dir / "gbt_distill_surrogate_best.pt"
    if gbt_path.exists():
        s = GBTDistillNet(input_dim, num_classes).to(device)
        s.load_state_dict(torch.load(gbt_path, map_location=device))
        s.eval()
        surrogates["GBT-Distill"] = s

    cnnv2_path = surr_dir / "cnnv2_surrogate_best.pt"
    if cnnv2_path.exists():
        s = CNNv2Surrogate(input_dim, num_classes).to(device)
        s.load_state_dict(torch.load(cnnv2_path, map_location=device))
        s.eval()
        surrogates["CNN-v2"] = s

    logger.info("CICIDS2017: %d targets, %d surrogates loaded",
                len(targets), len(surrogates))
    return {"targets": targets, "surrogates": surrogates, "input_dim": input_dim}


def _get_common_feature_indices(
    features_2017: list,
    features_2018: list,
) -> Tuple:
    """
    Return (idx_in_2017, idx_in_2018) for the common feature subset.
    CICIDS2018 columns have been renamed to 2017 canonical names in preprocessing,
    so the intersection is straightforward.
    """
    common = sorted(set(features_2017) & set(features_2018) - CICIDS2017_ONLY_FEATURES)
    idx17 = np.array([features_2017.index(f) for f in common], dtype=int)
    idx18 = np.array([features_2018.index(f) for f in common], dtype=int)
    logger.info("Common feature count (2017 ↔ 2018): %d", len(common))
    return idx17, idx18, common


def step5_cross_dataset(
    cfg: dict, device: torch.device, proc_dir: Path,
    all_targets: dict, all_surrogates: dict,
    features_2017: list,
    meta2018: dict,
) -> None:
    logger.info("\n[Step 5/6] Cross-dataset transfer (CICIDS2017 ↔ CICIDS2018)...")

    ecfg = cfg["eval"]
    seed = cfg["seed"]
    cross_dir = RESULTS_DIR / "cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)

    # Load CICIDS2017 models
    d17 = _load_cicids2017_models(device)

    # Get common feature indices
    features_2018 = meta2018.get("feature_names", [])
    if not features_2018:
        features_2018 = json.loads(
            (proc_dir / "cicids2018" / "splits" / "feature_names.json").read_text()
        )

    idx17, idx18, common_features = _get_common_feature_indices(features_2017, features_2018)

    # Load CICIDS2017 eval subset (restricted to common features)
    test17 = load_npz(get_splits_dir("cicids2017") / "test.npz")
    X17_full, y17 = test17["X"], test17["y"]
    # CICIDS2017 has 15-class labels; binarize for cross-dataset use
    y17_binary = (y17 > 0).astype(np.int64)
    X17_eval, y17_eval = stratified_eval_subset(X17_full, y17_binary, n=ecfg["n_eval"], seed=seed)
    X17_common = X17_eval[:, idx17]

    # Load CICIDS2018 eval subset (restricted to common features)
    splits18 = load_splits(proc_dir / "cicids2018" / "splits")
    X18_full, y18 = splits18["test"]
    X18_eval, y18_eval = stratified_eval_subset(X18_full, y18, n=ecfg["n_eval"], seed=seed)
    X18_common = X18_eval[:, idx18]

    # --- Direction 1: CICIDS2017 surrogates → CICIDS2018 models ---
    # Surrogates trained on 2017 (full input_dim=80). They need the common 78-feature subset.
    # We create wrapper surrogates that select the common features by subslicing.
    # Simpler: retrain 2017 surrogates on common features? No — instead we rebuild them.
    # Pragmatic solution: retrain minimal surrogates on 2017 common-feature data for this step.

    # Train 2017 surrogates on common features (if not already done)
    # Binarize CICIDS2017 labels (0=benign, 1=attack) for compatibility with binary 2018 models
    surr_cross_dir = cross_dir / "surrogates_2017_common"
    X17_tr_full, y17_tr_raw = load_npz(get_splits_dir("cicids2017") / "train.npz")["X"], \
                              load_npz(get_splits_dir("cicids2017") / "train.npz")["y"]
    X17_val_full, y17_val_raw = load_npz(get_splits_dir("cicids2017") / "val.npz")["X"], \
                                load_npz(get_splits_dir("cicids2017") / "val.npz")["y"]
    # Binary: 0=benign, 1=any attack
    y17_tr  = (y17_tr_raw  > 0).astype(np.int64)
    y17_val = (y17_val_raw > 0).astype(np.int64)
    X17_tr_common  = X17_tr_full[:, idx17]
    X17_val_common = X17_val_full[:, idx17]
    dim_common = len(idx17)
    nc17 = 2  # binary

    logger.info("Training 2017 surrogates on %d common features for cross-transfer...", dim_common)
    surr_2017_common = _train_surrogates_for_dataset(
        "cicids2017_common", X17_tr_common, y17_tr, X17_val_common, y17_val,
        dim_common, nc17, cfg, device, surr_cross_dir
    )

    # 2018 predict functions (common features)
    # CICIDS2018 models were trained on 78 features — same as common (idx18 selects all/subset)
    # Actually: 2018 targets use all 78 features. Common = 78 features (all of 2018).
    # So idx18 = range(78) if all 2018 features are common.
    tgt18_predict = {
        "RF":    {"predict": lambda X: sklearn_predict(all_targets["cicids2018"]["rf"], X[:, idx18])},
        "LR":    {"predict": lambda X: sklearn_predict(all_targets["cicids2018"]["lr"], X[:, idx18])},
        "MLP":   {"predict": lambda X: torch_predict(all_targets["cicids2018"]["mlp"], X[:, idx18], device)},
        "CNN1D": {"predict": lambda X: torch_predict(all_targets["cicids2018"]["cnn1d"], X[:, idx18], device)},
    }

    logger.info("Direction 1: 2017-common-surrogates → 2018 data → 2018 models")
    eval_cross_dataset_transfer(
        source_name="cicids2017",
        target_name="cicids2018",
        source_surrogates=surr_2017_common,
        target_targets=tgt18_predict,
        X_target_eval=X18_eval,   # full 78 features
        y_target_eval=y18_eval,
        common_feature_indices_source=None,  # surrogates already on common features
        common_feature_indices_target=None,
        eps_list=ecfg["epsilon_levels"],
        attacks=["fgsm", "pgd"],
        pgd_steps=ecfg["pgd_steps"],
        pgd_step_size_ratio=ecfg["pgd_step_size_ratio"],
        device=device,
        out_path=cross_dir / "cicids2017_to_cicids2018.csv",
        n_boot=ecfg["bootstrap_iters"],
    )

    # --- Direction 2: CICIDS2018 surrogates → CICIDS2017 common data → 2017 models ---
    # 2018 surrogates trained on 78 features. Common = 78 features.
    # 2017 common data = X17_eval[:, idx17] (78 features matching 2018 feature order via common).
    # 2017 models were trained on 80 features — need common 78-feature versions.

    # Retrain 2017 targets on common features (if not already done)
    tgt_cross_dir = cross_dir / "targets_2017_common"

    logger.info("Training 2017 targets on %d common features for cross-transfer...", dim_common)

    # Subsample 2017 train for efficiency (200K)
    rng = np.random.default_rng(seed)
    n_sub = min(200_000, len(y17_tr))
    sub_idx = rng.choice(len(y17_tr), n_sub, replace=False)
    X17_tr_sub = X17_tr_common[sub_idx]
    y17_tr_sub  = y17_tr[sub_idx]  # already binary

    rf17c  = train_rf(X17_tr_sub, y17_tr_sub, X17_common, y17_eval, 100, seed,
                      tgt_cross_dir / "rf_model.pkl")
    lr17c  = train_lr(X17_tr_sub, y17_tr_sub, X17_common, y17_eval, 1.0, seed,
                      tgt_cross_dir / "lr_model.pkl")
    mlp17c = train_mlp(
        X17_tr_common, y17_tr, X17_val_common, y17_val, X17_common, y17_eval,
        input_dim=dim_common, num_classes=nc17,  # nc17=2 (binary)
        hidden_dims=[256, 128], dropout=0.3, lr=0.001, weight_decay=1e-4,
        batch_size=256, max_epochs=30, patience=5, seed=seed, device=device,
        save_path=tgt_cross_dir / "mlp_best.pt"
    )
    cnn17c = train_cnn1d(
        X17_tr_common, y17_tr, X17_val_common, y17_val, X17_common, y17_eval,
        input_dim=dim_common, num_classes=nc17,  # nc17=2 (binary)
        channels=[32, 64], kernel_sizes=[3, 3], dropout=0.3, lr=0.001,
        weight_decay=1e-4, batch_size=256, max_epochs=30, patience=5,
        seed=seed, device=device, save_path=tgt_cross_dir / "cnn1d_best.pt"
    )

    tgt17_common_predict = {
        "RF":    {"predict": lambda X, m=rf17c["model"]:  sklearn_predict(m, X)},
        "LR":    {"predict": lambda X, m=lr17c["model"]:  sklearn_predict(m, X)},
        "MLP":   {"predict": lambda X, m=mlp17c["model"]: torch_predict(m, X, device)},
        "CNN1D": {"predict": lambda X, m=cnn17c["model"]: torch_predict(m, X, device)},
    }

    logger.info("Direction 2: 2018-surrogates → 2017-common data → 2017-common models")
    eval_cross_dataset_transfer(
        source_name="cicids2018",
        target_name="cicids2017",
        source_surrogates=all_surrogates["cicids2018"],
        target_targets=tgt17_common_predict,
        X_target_eval=X17_common,
        y_target_eval=y17_eval,
        common_feature_indices_source=None,
        common_feature_indices_target=None,
        eps_list=ecfg["epsilon_levels"],
        attacks=["fgsm", "pgd"],
        pgd_steps=ecfg["pgd_steps"],
        pgd_step_size_ratio=ecfg["pgd_step_size_ratio"],
        device=device,
        out_path=cross_dir / "cicids2018_to_cicids2017.csv",
        n_boot=ecfg["bootstrap_iters"],
    )

    # Build transferability summary
    _build_cross_summary(cross_dir)
    logger.info("[Step 5/6] DONE")


def _build_cross_summary(cross_dir: Path) -> None:
    rows = []
    for f in cross_dir.glob("*.csv"):
        if "surrogate" in f.name or "target" in f.name:
            continue
        try:
            df = pd.read_csv(f)
            for tgt in df["target_model"].unique():
                sub = df[df["target_model"] == tgt]
                rows.append({
                    "file": f.name,
                    "target_model": tgt,
                    "max_asr": round(sub["asr"].max(), 4),
                    "mean_asr": round(sub["asr"].mean(), 4),
                    "best_surrogate": sub.loc[sub["asr"].idxmax(), "surrogate"],
                    "best_attack": sub.loc[sub["asr"].idxmax(), "attack"],
                    "best_eps": sub.loc[sub["asr"].idxmax(), "eps"],
                })
        except Exception as e:
            logger.warning("Skipping %s: %s", f.name, e)
    if rows:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(cross_dir / "cross_transfer_summary.csv", index=False)
        logger.info("Cross-transfer summary saved")


# ---------------------------------------------------------------------------
# Step 6: Final outputs
# ---------------------------------------------------------------------------

def step6_outputs(
    all_targets: dict, all_surrogates: dict, proc_dir: Path,
) -> None:
    logger.info("\n[Step 6/6] Generating final outputs...")

    # Compile combined in-distribution summary
    in_dist_dir = RESULTS_DIR / "in_distribution"
    dfs = []
    for f in sorted(in_dist_dir.glob("*_transfer_matrix.csv")):
        df = pd.read_csv(f)
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(in_dist_dir / "all_datasets_combined.csv", index=False)

        # Key finding: does RF robustness replicate?
        rf_rows = combined[combined["target"] == "RF"]
        mlp_rows = combined[combined["target"] == "MLP"]
        summary = {
            "RF_mean_asr_across_datasets": round(rf_rows["asr"].mean(), 4),
            "MLP_mean_asr_across_datasets": round(mlp_rows["asr"].mean(), 4),
            "RF_max_asr_across_datasets": round(rf_rows["asr"].max(), 4),
            "MLP_max_asr_across_datasets": round(mlp_rows["asr"].max(), 4),
            "datasets_evaluated": combined["dataset"].unique().tolist(),
            "finding": (
                "RF robustness pattern REPLICATES" if rf_rows["asr"].mean() < mlp_rows["asr"].mean()
                else "RF robustness pattern DOES NOT replicate — dataset-specific"
            ),
        }
        (RESULTS_DIR / "in_distribution" / "in_distribution_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info("Key finding: %s", summary["finding"])

    logger.info("[Step 6/6] DONE")
    logger.info("=" * 70)
    logger.info("PHASE 6 COMPLETE")
    logger.info("  preprocessing/    — metadata + feature compatibility report")
    logger.info("  clean_baselines/  — accuracy on each dataset's own test set")
    logger.info("  in_distribution/  — transfer matrices per dataset")
    logger.info("  cross_dataset/    — CICIDS2017 ↔ CICIDS2018 cross-transfer results")
    logger.info("  models/           — all trained model checkpoints")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import yaml
    cfg_path = Path(__file__).parent / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    logger.info("=" * 70)
    logger.info("PHASE 6: Cross-Dataset Evaluation")
    logger.info("Device: %s", device)
    logger.info("=" * 70)

    t_start = time.time()
    proc_dir = PROJECT_ROOT / "data" / "processed"

    step1_out    = step1_preprocess(cfg)
    all_targets  = step2_train_targets(cfg, device, proc_dir)
    all_surr     = step3_train_surrogates(cfg, device, proc_dir)
    step4_in_distribution(cfg, device, proc_dir, all_targets, all_surr)
    step5_cross_dataset(
        cfg, device, proc_dir, all_targets, all_surr,
        features_2017=step1_out["features_2017"],
        meta2018=step1_out["meta2018"],
    )
    step6_outputs(all_targets, all_surr, proc_dir)

    elapsed = time.time() - t_start
    logger.info("Total runtime: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
