# Adversarial-Robustness-Evaluation-of-AI-IDS
A modular, reproducible, research-grade framework for evaluating adversarial robustness of AI-based intrusion detection systems using CICIDS2017 and CICIDS2018 datasets.

**Supplementary Code for the Paper:**
*"Evaluating Adversarial Robustness of AI Intrusion Detection Systems Using Automated Traffic Generation"*

**Author:** Dr. Samer Aoudi  
**Target Journal:** Discover Artificial Intelligence

**Overview**
This repository implements a full experimental pipeline for adversarial robustness evaluation of machine-learning and deep-learning–based Intrusion Detection Systems (IDS).
It supports:
- Clean training of classical and deep IDS models
- Surrogate-based adversarial example generation
- White-box, score-based black-box, and decision-based black-box attacks
- Transferability testing across heterogeneous IDS models
- Cross-dataset robustness evaluation (CICIDS2017 → CICIDS2018)
- Semantic & protocol-aware feature constraints
- Reproducible, CPU-friendly experiment orchestration

This framework is designed for academic research, adversarial ML evaluations, and security testing of AI-based IDS.

**Key Features**
Complete Adversarial Attack Suite
- FGSM (white-box gradient)
- PGD (iterative white-box)
- ZOO (score-based black-box)
- HSJA (decision-based black-box)
- Transfer attacks from surrogate → target IDS models

**Multiple Target IDS Models**
All trained on CICIDS2017:
- Random Forest (RF)
- Logistic Regression (LR)
- Deep MLP
- 1D CNN (CNN1D)

Surrogate MLP is used for white-box gradient-based attack generation.

**Cross-Dataset Evaluation**
Attack examples generated on CICIDS2017 are evaluated on:
- CICIDS2017 (in-distribution)
- CICIDS2018 Friday Slice (out-of-distribution)

This supports reviewer-requested robustness generalization studies.

**Reproducible Experiment Management**
Deterministic seeds
Modular architecture
YAML-based configuration
Versioned outputs
Logging for every stage of the experiment

Clean separation between data, models, attack modules, and evaluation

**Semantic & Protocol-Aware Constraints**
All adversarial examples respect:
- Feature min/max bounds
- Non-negativity
- Monotonicity where required
- Protocol-valid transformations
- Batched clipping & enforcement utilities

**Repository Structure**

```txt
Adversarial-Robustness-Evaluation-AI-IDS/
|
|-- configs/
|   |-- attacks/
|   |   `-- cicids2017_whitebox_transfer.yaml
|   `-- train/
|       `-- cicids2017_baselines.yaml
|
|-- data/
|   |-- raw/
|   |   |-- cicids2017/
|   |   `-- cicids2018/
|   `-- processed/
|       |-- cicids2017/
|       `-- cicids2018_friday/
|
|-- scripts/
|   |-- prepare_cicids2017.py
|   |-- prepare_cicids2018_friday.py
|   |-- train_baselines_cicids2017.py
|   |-- train_deep_cicids2017.py
|   |-- run_attacks_cicids2017.py
|   `-- run_attacks_cicids2018_friday.py
|
|-- src/
│   ├── data/              # preprocessing, splits, constraints
│   ├── models/            # RF, LR, MLP, CNN1D, surrogate wrappers
│   ├── attacks/           # FGSM, PGD, ZOO, HSJA, transfer evaluation
│   ├── utils/             # config, logging, paths, seeding
|   `-- evaluation/       # metrics, confusion matrices, plotting
|
|-- artifacts/
|   `-- <dataset>/
|       |-- scaler.pkl
|       |-- label_mapping.json
|       `-- constraints.json
|
|-- outputs/
|   `-- <dataset>/
|       |-- adv/
|       |   `-- runs/
|       |       `-- <attack_name>/*.csv
|       |-- plots/
|       `-- tables/
|
`-- README.md
```
**Prerequisites**
1. Python 3.10
2. CICIDS2017 Dataset (CSV)
3. CICIDS2018 Dataset (Friday Slice)

**Installation**
1. Install Python (3.10 recommended) 
Download from https://python.org if needed.

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

   Ensure PyTorch CPU version is installed:  
   pip install torch --index-url https://download.pytorch.org/whl/cpu

5. Data Setup  
   -Download the MachineLearningCVE version of CICIDS2017.  
   -Place the CSV files in: data/raw/cicids2017/CSVs/MachineLearningCVE/  
   -(Optional) For cross-dataset testing, place CICIDS2018 CSVs in: data/raw/cicids2018/

**Reproducibility Workflow**  
To reproduce the results presented in the paper, follow these steps in order:

**Step 1 Dataset Preparation (Preprocessing)**  
**CICIDS2017**  
python -m scripts.prepare_cicids2017

**CICIDS2018 Friday Slice**  
python -m scripts.prepare_cicids2018_friday


These scripts:
- Merge, clean, and validate CSV data
- Apply scaling and constraints
- Produce reproducible train/val/test splits
- Save artifacts (scaler, label map, constraints)

**Training IDS Models**  
**Classical Models (RF, LR)**  
python -m scripts.train_baselines_cicids2017

**Deep Models (MLP, CNN1D, Surrogate)**  
python -m scripts.train_deep_cicids2017

All models are saved automatically under:  
artifacts/cicids2017/models/

**Running Adversarial Attacks**  
**On CICIDS2017 (in-distribution)**  
python -m scripts.run_attacks_cicids2017

**On CICIDS2018 (cross-dataset)**  
python -m scripts.run_attacks_cicids2018_friday

Outputs (CSV tables) will appear under:  
outputs/<dataset>/adv/runs/<run_name>/

**Evaluation Metrics**  
The framework computes:
- Clean accuracy
- Adversarial accuracy
- Δ accuracy (clean – adv)
- Attack success rate (ASR)
- Per-class degradation
- Robust accuracy vs. ε curves
- ASR vs. ε curves
- Confusion matrices
- Cross-dataset generalization metrics

All metrics follow best-practice definitions widely used in adversarial ML literature.

**Experiments Included**
**✔ FGSM / PGD white-box attacks
✔ ZOO score-based black-box attacks
✔ HSJA decision-based black-box attacks
✔ Transferability across:**
- RF
- LR
- MLP
- CNN1D
**✔ Cross-dataset evaluation:**
- CICIDS2017 → CICIDS2018 Friday slice

**Threat Model Summary**
- The attacker has full white-box access to the surrogate model.
- The attacker has no internal access to the target IDS models.
- Perturbations must satisfy semantic & protocol constraints.
- The attacker attempts to achieve:
    - Misclassification
    - Evasion
    - Cross-model & cross-dataset transferability

**Reproducibility**
This project ensures reproducibility through:
- Fixed random seeds
- Deterministic PyTorch operations where possible
- Versioned outputs
- Machine-readable CSV logs for each attack
- Configurable YAML experiment files
- Separated code modules with self-contained components

**Citing This Work**
If you use this repository in academic research, please cite:
Aoudi, S. (2025). Adversarial Robustness Evaluation of AI-Based Intrusion Detection Systems Using Automated Traffic Generation. 
Higher Colleges of Technology / UAE.

**Contributing**
Pull requests are welcome!
For major changes, please open an issue first to discuss:
- New attack types
- New datasets
- Faster preprocessing pipelines
- GPU extensions
- CI/CD integration for reproducible experiments
All code should follow:
- PEP-8
- Modular design principles
- Clear documentation and docstrings

**Issues / Support**
If you encounter errors running the pipeline:
- Open an Issue on GitHub
- Attach logs from logs/<dataset>/<run_name>.log
- Specify OS, Python version, and steps to reproduce

**License**
MIT License

Thank you!
Dr. Samer Aoudi
