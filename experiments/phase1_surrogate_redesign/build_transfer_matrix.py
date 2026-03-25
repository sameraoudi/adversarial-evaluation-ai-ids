"""
Phase 1: Build and Print the 72-Row Transfer Matrix

Reads transfer_results_full.json and produces:
  1. Formatted 72-row table (printed to console)
  2. results/transfer_matrix.csv
  3. results/transfer_matrix_summary.json (per-target best/worst surrogate stats)

Table structure (72 rows: 3 surrogates x 4 targets x 2 attacks x 3 epsilons):
  Surrogate | Target | Attack | ε | Clean Acc | Adv Acc | Δ Acc | ASR | 95% CI (Adv Acc) | 95% CI (ASR)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SURROGATE_DISPLAY = {
    "lr_surrogate": "LR-Surr",
    "gbt_distill_surrogate": "GBT-Distill",
    "cnn_v2_surrogate": "CNN-v2",
}


def load_results(results_dir: Path) -> List[Dict]:
    json_path = results_dir / "transfer_results_full.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Results not found at {json_path}. Run evaluate_transfer.py first.")
    with json_path.open() as f:
        rows = json.load(f)
    return rows


def build_matrix(rows: List[Dict]) -> List[Dict]:
    """
    Sort rows into the canonical 72-row order:
      surrogates x targets x attacks x epsilons
    and compute display-friendly fields.
    """
    surrogate_order = ["lr_surrogate", "gbt_distill_surrogate", "cnn_v2_surrogate"]
    target_order = ["RF", "LR", "MLP", "CNN1D"]
    attack_order = ["fgsm", "pgd"]
    eps_order = [0.01, 0.03, 0.05]

    # Index rows by (surrogate, target, attack, eps)
    indexed = {}
    for r in rows:
        key = (r["surrogate"], r["target"], r["attack"], round(float(r["eps"]), 4))
        indexed[key] = r

    matrix = []
    for surr in surrogate_order:
        for tgt in target_order:
            for atk in attack_order:
                for eps in eps_order:
                    key = (surr, tgt, atk, round(eps, 4))
                    if key not in indexed:
                        # Missing row — fill with NaN
                        row = {
                            "surrogate_raw": surr,
                            "surrogate": SURROGATE_DISPLAY.get(surr, surr),
                            "target": tgt,
                            "attack": atk.upper(),
                            "eps": eps,
                            "clean_acc": float("nan"),
                            "adv_acc": float("nan"),
                            "delta_acc": float("nan"),
                            "asr": float("nan"),
                            "adv_acc_ci": "[nan, nan]",
                            "asr_ci": "[nan, nan]",
                        }
                    else:
                        r = indexed[key]
                        row = {
                            "surrogate_raw": surr,
                            "surrogate": SURROGATE_DISPLAY.get(surr, surr),
                            "target": tgt,
                            "attack": atk.upper(),
                            "eps": eps,
                            "clean_acc": round(r["clean_acc"], 4),
                            "adv_acc": round(r["adv_acc"], 4),
                            "delta_acc": round(r["delta_acc"], 4),
                            "asr": round(r["asr"], 4),
                            "adv_acc_ci": f"[{r['adv_acc_ci_lo']:.4f}, {r['adv_acc_ci_hi']:.4f}]",
                            "asr_ci": f"[{r['asr_ci_lo']:.4f}, {r['asr_ci_hi']:.4f}]",
                        }
                    matrix.append(row)
    return matrix


def print_table(matrix: List[Dict]) -> None:
    """Print formatted table to console."""
    # Column widths
    header = [
        ("Surrogate",   11),
        ("Target",       6),
        ("Attack",       6),
        ("ε",            5),
        ("Clean Acc",   10),
        ("Adv Acc",      9),
        ("Δ Acc",        8),
        ("ASR",          7),
        ("95% CI (Adv Acc)",  18),
        ("95% CI (ASR)",      14),
    ]

    def fmt(val, width):
        return str(val).ljust(width)

    sep = " | ".join("-" * w for _, w in header)
    head = " | ".join(h.ljust(w) for h, w in header)

    print()
    print("=" * len(sep))
    print("PHASE 1 TRANSFER MATRIX — 72 rows (3 surrogates × 4 targets × 2 attacks × 3 epsilons)")
    print("=" * len(sep))
    print(head)
    print(sep)

    prev_block = None
    for i, row in enumerate(matrix):
        block = (row["surrogate"], row["target"])
        if prev_block and block != prev_block:
            print(sep)
        prev_block = block

        def fmtf(v, w):
            if isinstance(v, float) and (v != v):  # nan
                return "N/A".ljust(w)
            if isinstance(v, float):
                return f"{v:.4f}".ljust(w)
            return str(v).ljust(w)

        cols = [
            fmt(row["surrogate"], 11),
            fmt(row["target"], 6),
            fmt(row["attack"], 6),
            fmtf(row["eps"], 5),
            fmtf(row["clean_acc"], 10),
            fmtf(row["adv_acc"], 9),
            fmtf(row["delta_acc"], 8),
            fmtf(row["asr"], 7),
            fmt(row["adv_acc_ci"], 18),
            fmt(row["asr_ci"], 14),
        ]
        print(" | ".join(cols))

    print("=" * len(sep))
    print(f"Total rows: {len(matrix)}")
    print()


def save_csv(matrix: List[Dict], path: Path) -> None:
    fieldnames = [
        "surrogate", "target", "attack", "eps",
        "clean_acc", "adv_acc", "delta_acc", "asr",
        "adv_acc_ci", "asr_ci",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matrix)
    print(f"Saved CSV: {path}")


def compute_summary(matrix: List[Dict]) -> Dict:
    """
    Summary statistics:
    - Per target: which surrogate produces highest/lowest ASR?
    - Overall best-performing attack setup
    - Any surprising results (e.g., non-MLP surrogate still beats MLP target hard)
    """
    target_order = ["RF", "LR", "MLP", "CNN1D"]
    surrogate_order = ["LR-Surr", "GBT-Distill", "CNN-v2"]

    summary = {"per_target": {}, "overall_best": None, "surprises": []}

    # Per target: average ASR per surrogate
    for tgt in target_order:
        tgt_rows = [r for r in matrix if r["target"] == tgt and not (isinstance(r["asr"], float) and r["asr"] != r["asr"])]
        per_surr = {}
        for surr in surrogate_order:
            surr_rows = [r for r in tgt_rows if r["surrogate"] == surr]
            if surr_rows:
                asr_vals = [r["asr"] for r in surr_rows]
                per_surr[surr] = {
                    "mean_asr": round(float(np.mean(asr_vals)), 4),
                    "max_asr": round(float(np.max(asr_vals)), 4),
                    "min_asr": round(float(np.min(asr_vals)), 4),
                }
        if per_surr:
            best_surr = max(per_surr, key=lambda s: per_surr[s]["mean_asr"])
            worst_surr = min(per_surr, key=lambda s: per_surr[s]["mean_asr"])
            summary["per_target"][tgt] = {
                "per_surrogate_asr": per_surr,
                "highest_transfer_surrogate": best_surr,
                "lowest_transfer_surrogate": worst_surr,
            }

    # Overall best setup (highest single ASR)
    valid = [r for r in matrix if isinstance(r["asr"], float) and r["asr"] == r["asr"]]
    if valid:
        best = max(valid, key=lambda r: r["asr"])
        summary["overall_best"] = {
            "surrogate": best["surrogate"],
            "target": best["target"],
            "attack": best["attack"],
            "eps": best["eps"],
            "asr": best["asr"],
        }

    # Surprises: non-MLP surrogate with ASR > 0.5 against MLP target
    for r in valid:
        if r["target"] == "MLP" and r["surrogate"] != "LR-Surr" and r["asr"] > 0.5:
            summary["surprises"].append({
                "note": "Non-LR surrogate achieves high transfer to MLP target",
                "row": {k: r[k] for k in ["surrogate", "target", "attack", "eps", "asr"]},
            })
        # Cross-model transfer: CNN-v2 surrogate attacking RF with high ASR
        if r["target"] == "RF" and r["surrogate"] == "CNN-v2" and r["asr"] > 0.3:
            summary["surprises"].append({
                "note": "CNN-v2 surrogate achieves notable transfer to RF target",
                "row": {k: r[k] for k in ["surrogate", "target", "attack", "eps", "asr"]},
            })

    return summary


def main():
    results_dir = Path(__file__).parent / "results"
    rows = load_results(results_dir)
    matrix = build_matrix(rows)

    print_table(matrix)
    save_csv(matrix, results_dir / "transfer_matrix.csv")

    summary = compute_summary(matrix)
    summary_path = results_dir / "transfer_matrix_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")

    # Print comparison note
    print("\n=== BIAS ANALYSIS ===")
    print("Old surrogate (MLP, hidden=[256,128], ReLU) was architecturally identical")
    print("to target MLP (hidden=[256,128], ReLU), making that evaluation near-white-box.")
    print("Phase 1 uses three distinct surrogate architectures:")
    print("  1. LR-Surr:       Linear(input_dim, num_classes) — single layer")
    print("  2. GBT-Distill:   512->256 GELU, distilled from XGBoost")
    print("  3. CNN-v2:        3×conv(16/32/64,k=5)+LeakyReLU+BN+GAP->FC(64)")
    print("Compare MLP-target ASR values across surrogates to quantify the original bias.")

    return matrix, summary


if __name__ == "__main__":
    main()
