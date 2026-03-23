# Adversarial Robustness of AI-Based Intrusion Detection Systems: A Multi-Dataset, Multi-Paradigm Evaluation Framework

A modular, reproducible, research-grade framework for evaluating the adversarial robustness of AI-based intrusion detection systems across multiple datasets, surrogate architectures, attack paradigms, and defense strategies.

**Supplementary Code for:**
> Aoudi, S., & Al-Aqrabi, H. (2025). *Adversarial Robustness of AI-Based Intrusion Detection Systems: A Multi-Dataset, Multi-Paradigm Evaluation Framework.* Submitted to Computers & Security (Elsevier).

---

## Overview

This repository implements the complete experimental pipeline described in the paper, covering:

- **Multi-surrogate adversarial evaluation** — Three architecturally distinct surrogates (LR-Surr, GBT-Distill, CNN-v2) attacking four target IDS models (RF, LR, MLP, CNN1D)
- **Nine adversarial attack paradigms** — Gradient-transfer (FGSM, PGD), tree-structure-aware (Kantchelian greedy), importance-guided random search, decision-based (Boundary Attack, HSJA), and score-based (ZOO)
- **Domain-grounded feature perturbation taxonomy** — Six-category classification of 80 CICFlowMeter features with per-feature perturbation constraints
- **Defense evaluation** — Adversarial training (AT), TRADES, and input smoothing with five generalization tests
- **Cross-dataset evaluation** — Independent training and adversarial analysis on four benchmark datasets with cross-dataset transfer experiments
- **Statistical validation** — Bootstrap confidence intervals, McNemar's pairwise tests, and Friedman ranking analysis

## Key Findings

| Finding | Evidence |
|---------|----------|
| Surrogate–target architectural alignment governs transfer success | LR-Surr→MLP: 55.6% ASR; LR-Surr→CNN1D: 17.2% ASR |
| RF robustness is attack-paradigm-independent on CICIDS2017 | 9 attack types converge at ~19–20% ASR |
| 100% of uniform-ε adversarial samples violate domain constraints | 41% of features are non-perturbable by a network attacker |
| Adversarial training creates a defense paradigm trade-off | AT eliminates gradient attacks but increases HSJA ASR by +22pp |
| TRADES provides universal defense | <1% ASR across all attack paradigms |
| Robustness rankings are dataset-dependent | RF most robust on CICIDS2017, most vulnerable on CICIDS2018 |
| Cross-dataset adversarial transfer is asymmetric | 2017→2018: ~0% ASR; 2018→2017: ~12% ASR |

## Datasets

| Dataset | Year | Samples | Features | Extractor | Classes | Benign % |
|---------|------|---------|----------|-----------|---------|----------|
| CICIDS2017 | 2017 | 2.8M | 80 | CICFlowMeter | 15 | 82.5% |
| CICIDS2018 | 2018 | 1.5M | 78 | CICFlowMeter | Binary | 72.5% |
| UNSW-NB15 | 2015 | 258K | 42 | Argus/Bro-IDS | Binary | 38.5% |
| CIC-IoT-2023 | 2023 | 1.0M | 39 | CICFlowMeter-v2 | Binary | 2.3% |

All datasets are publicly available. See [Data Setup](#data-setup) for download instructions.

## Models

### Target Models (trained per dataset)
| Model | Architecture | Framework |
|-------|-------------|-----------|
| RF | 100–200 trees, class-balanced | scikit-learn |
| LR | Linear(d, C), LBFGS | scikit-learn |
| MLP | d→256→128→C, BN+ReLU+Dropout | PyTorch |
| CNN1D | Conv(32,64)→GAP→C, BN+ReLU | PyTorch |

### Surrogate Models (architecturally distinct from all targets)
| Surrogate | Architecture | Design Rationale |
|-----------|-------------|-----------------|
| LR-Surr | Linear(d, C) | Captures linear decision boundaries only |
| GBT-Distill | d→512→256→C, GELU (distilled from XGBoost) | Smooth approximation of tree-based boundaries |
| CNN-v2 | Conv(16,32,64, k=5)→GAP→C, LeakyReLU+BN | Wide kernels, different depth/activation from CNN1D |

## Attack Suite

| Category | Attack | Target Access | Gradient Source |
|----------|--------|--------------|-----------------|
| Gradient transfer | FGSM | None (black-box) | Surrogate |
| Gradient transfer | PGD (T=40) | None (black-box) | Surrogate |
| Structure-aware | Kantchelian greedy | RF internals | None (tree traversal) |
| Importance-guided | FI-Random Search | RF predictions | None (feature importance) |
| Decision-based | Boundary Attack | Hard labels only | None |
| Decision-based | HSJA | Hard labels only | None |
| Score-based | ZOO | Confidence scores | Finite differences |

## Defense Methods

| Defense | Training Signal | Attack Coverage |
|---------|----------------|-----------------|
| AT-Hardened | Mixed clean + PGD from surrogate | Gradient attacks (FGSM/PGD) |
| TRADES (β=1.0) | KL-divergence regularization using model's own gradients | All paradigms |
| Input Smoothing | Gaussian noise (σ∈{0.005, 0.01, 0.02}) at inference | Partial |

## Repository Structure

```
adversarial-evaluation-ai-ids/
├── configs/
│   ├── attacks/                          # Attack configurations (YAML)
│   ├── datasets/                         # Dataset paths and split ratios
│   └── training/                         # Model hyperparameters
│
├── data/
│   ├── raw/
│   │   ├── cicids2017/CSVs/MachineLearningCVE/
│   │   ├── cicids2018/
│   │   ├── unsw_nb15/
│   │   └── ciciot2023/
│   └── processed/                        # Generated splits (train/val/test .npz)
│
├── src/
│   ├── data/                             # Preprocessing, splits, constraints
│   ├── models/                           # RF, LR, MLP, CNN1D, surrogates, wrappers
│   ├── attacks/                          # FGSM, PGD, ZOO, HSJA, Kantchelian, FI-Search
│   ├── evaluation/                       # Metrics, bootstrap CIs, McNemar, Friedman
│   └── utils/                            # Config, logging, paths, seeding
│
├── scripts/                              # Entry-point scripts for each pipeline stage
│
├── experiments/
│   ├── phase1_surrogate_redesign/        # 3-surrogate × 4-target transfer matrices
│   ├── phase2_tree_attacks/              # 9-paradigm RF robustness analysis
│   ├── phase3_4_domain_constraints/      # Feature taxonomy + constraint validation
│   ├── phase5_adversarial_training/      # AT, TRADES, smoothing + generalization
│   ├── phase6_cross_dataset/             # 4-dataset evaluation + cross-transfer
│   ├── phase7_statistics/                # McNemar, bootstrap, Friedman tests
│   └── phase8_paper/                     # Manuscript source files
│
├── models/                               # Saved model checkpoints per dataset
│   ├── cicids2017/{baselines,deep,surrogates}/
│   ├── cicids2018/{baselines,deep,surrogates}/
│   ├── unsw_nb15/{baselines,deep,surrogates}/
│   └── ciciot2023/{baselines,deep,surrogates}/
│
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended; CPU fallback supported)
- 16 GB+ RAM (32 GB recommended for CICIDS2017 full preprocessing)

## Installation

```bash
# Clone the repository
git clone https://github.com/sameraoudi/adversarial-evaluation-ai-ids.git
cd adversarial-evaluation-ai-ids

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# For CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data Setup

1. **CICIDS2017** — Download the MachineLearningCVE CSVs from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html). Place in `data/raw/cicids2017/CSVs/MachineLearningCVE/`.

2. **CICIDS2018** — Download Wednesday, Thursday, and Friday CSVs from [CIC](https://www.unb.ca/cic/datasets/ids-2018.html). Place in `data/raw/cicids2018/`.

3. **UNSW-NB15** — Download from [UNSW Sydney](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Place in `data/raw/unsw_nb15/`.

4. **CIC-IoT-2023** — Download from [CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html). Place in `data/raw/ciciot2023/`.

## Reproducing the Paper Results

Execute the following steps in order. Each phase has crash recovery and will skip completed steps on re-run.

### Step 1: Preprocessing
```bash
python -m scripts.prepare_cicids2017
python -m scripts.prepare_cicids2018
python -m scripts.prepare_unsw_nb15
python -m scripts.prepare_ciciot2023
```

### Step 2: Train Target Models (all datasets)
```bash
python -m scripts.train_baselines    # RF, LR
python -m scripts.train_deep         # MLP, CNN1D
```

### Step 3: Phase 1 — Multi-Surrogate Transfer Analysis
```bash
python experiments/phase1_surrogate_redesign/run_phase1.py
python experiments/phase1_surrogate_redesign/build_transfer_matrix.py
```

### Step 4: Phase 2 — Tree-Aware RF Attacks
```bash
python experiments/phase2_tree_attacks/run_phase2.py
```

### Step 5: Phases 3+4 — Domain Constraints & Validation
```bash
python experiments/phase3_4_domain_constraints/run_phase3_4.py
```

### Step 6: Phase 5 — Adversarial Training & Defense Comparison
```bash
python experiments/phase5_adversarial_training/run_phase5.py
```

### Step 7: Phase 6 — Cross-Dataset Evaluation
```bash
python experiments/phase6_cross_dataset/run_phase6.py
```

### Step 8: Phase 7 — Statistical Testing
```bash
python experiments/phase7_statistics/run_phase7.py
```

All results are saved to the respective `experiments/phaseN/results/` directories as CSV, JSON, and log files.

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Clean Accuracy | Accuracy on unperturbed test data |
| Robust Accuracy | Accuracy on adversarial examples (originally correct samples only) |
| ASR | Fraction of originally correct predictions flipped by the attack |
| Δ Accuracy | Clean accuracy − robust accuracy |
| Bootstrap 95% CI | Percentile-based confidence intervals (1,000 iterations, seed 42) |
| McNemar's Test | Pairwise per-sample classifier comparison with Bonferroni correction |
| Friedman Test | Cross-dataset ranking consistency (k=4 models, b=3 datasets) |

## Feature Perturbation Taxonomy

| Category | Count | Perturbable? | Description |
|----------|-------|-------------|-------------|
| Bidirectional | 24 | Yes (both directions) | Aggregate flow statistics |
| Increase-only | 14 | Yes (positive only) | Monotone counters (packets, bytes) |
| Flags | 10 | Restricted | TCP control bits |
| Backward-response | 22 | **No** | Server-side features |
| Derived | 9 | **No** | Computed from other features |
| Immutable | 1 | **No** | Fixed by definition |
| **Total** | **80** | **47 perturbable / 33 non-perturbable (41%)** | |

Full per-feature assignments are provided in `experiments/phase3_4_domain_constraints/results/feature_taxonomy_full.json`.

## Threat Model

The evaluation considers two operationally distinct threat scenarios:

- **Transfer-based black-box**: Attacker trains a surrogate on proxy data; generates FGSM/PGD perturbations; transfers to the target IDS. No target access required.
- **Query-based black-box**: Attacker queries the target IDS and observes decisions. HSJA uses binary search on the decision boundary. ZOO approximates gradients via finite differences.

All attacks operate under L∞ perturbation constraints with ε ∈ {0.01, 0.03, 0.05} on min-max normalized features. Feature-aware variants additionally enforce the domain taxonomy.

## Reproducibility

This project ensures reproducibility through:
- Fixed random seeds (42) across all data splits, model initialization, and attack generation
- Deterministic PyTorch operations where supported
- YAML-based experiment configuration
- Crash recovery with checkpoint-based resumption
- Machine-readable CSV/JSON output for all results
- Modular separation of data, models, attacks, and evaluation

## Hardware

Experiments were conducted on:
- **GPU**: NVIDIA L40 (12 GB VRAM)
- **CPU**: Intel i7, 32 GB RAM
- **Software**: Python 3.10, PyTorch 2.x, scikit-learn 1.3, XGBoost 2.0

Total experimental runtime: approximately 15 hours across all phases.

## Citing This Work

If you use this repository in academic research, please cite:

```bibtex
@article{aoudi2025adversarial,
  title={Adversarial Robustness of AI-Based Intrusion Detection Systems: 
         A Multi-Dataset, Multi-Paradigm Evaluation Framework},
  author={Aoudi, Samer and Al-Aqrabi, Hussain},
  journal={Computers \& Security},
  year={2025},
  note={Under review}
}
```

## License

MIT License

## Issues / Support

If you encounter errors running the pipeline:
1. Open an Issue on GitHub
2. Attach the relevant log file from the `results/` directory
3. Specify your OS, Python version, GPU (if applicable), and steps to reproduce
