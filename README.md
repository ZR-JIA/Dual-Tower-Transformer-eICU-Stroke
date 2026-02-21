# Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower Transformer Framework

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

> **⚠️ IMPORTANT: Medical AI Disclaimer**
> This model is for **Academic Research Use Only**. It is **NOT** an FDA-approved medical device and must **NOT** be used for clinical diagnosis, treatment decisions, or patient care. Commercial use is strictly prohibited. See [LICENSE](LICENSE) for details.

---

## Abstract

Stroke mortality prediction in Intensive Care Units (ICU) relies heavily on heterogeneous tabular data, where tree-based ensembles typically dominate. However, these models lack the flexibility for end-to-end multimodal integration. This study proposes a novel **Dual-Tower Transformer (DT-Transformer)** engineered to bridge the performance gap between deep learning architectures and gradient boosting benchmarks on the multi-center eICU database. We introduce a decoupled architecture that processes categorical demographics and numerical vitals through separate pathways, utilizing a **Self-Attention mechanism** to capture global feature interactions. Furthermore, an **Adaptive Runtime Safeguard** is integrated to ensure inference stability against physiological outliers. Results demonstrate that our method successfully resolves the convergence difficulty observed in standard Transformers (AUPRC 0.5279), achieving a higher mean AUPRC of **0.6171** (std 0.006). While the tuned XGBoost baseline retains the performance ceiling (0.6467), our framework offers a competitive, fully differentiable alternative suitable for future integration with unstructured clinical notes.

---

## Key Features

- **Dual-Tower Architecture** — Decoupled processing of categorical demographics (Left Tower: learnable embeddings) and numerical vitals (Right Tower: configurable MLP or Transformer encoder)
- **NumericalFeatureTokenizer** — Converts scalar physiological features into dense token embeddings for self-attention, enabling high-order feature interaction capture
- **Adaptive Runtime Safeguard** — Inference-time stability mechanism against physiological outliers
- **Hierarchical Configuration System** — YAML-based config with `_base/` global defaults and per-model overrides, supporting `${variable}` resolution
- **Unified Training Engine** — Single entry point supporting 7 model types: `DualTower (Transformer)`, `DualTower (MLP)`, `Transformer`, `MLP`, `NN`, `XGBoost`, `RandomForest`
- **Multi-Seed Reproducibility** — Academic-grade multi-seed experiment runner with aggregated mean ± std reporting
- **Optuna Hyperparameter Optimization** — Integrated Bayesian search with configurable search spaces
- **Comprehensive Evaluation** — AUROC, AUPRC, Brier score, ECE, bootstrap confidence intervals, Youden threshold selection, and isotonic calibration
- **Publication-Ready Exports** — Model versioning, leaderboard tracking, SHAP/LIME explainability, and publication-style visualization

---

## Adaptive Runtime Safeguard

Unlike traditional offline data cleaning, the **Adaptive Runtime Safeguard** operates **online within the data loader**, acting as a protection mechanism against physiological outliers before tensors are transferred to the GPU. This ensures a "zero-crash" inference process critical for clinical deployment reliability.

The safeguard consists of two complementary mechanisms (see paper TABLE I):

### 1. Categorical Clamping

| Feature | Constraint | Purpose |
|---------|-----------|---------|
| Age | x ∈ [0, 149] | Ensure biological plausibility |
| Gender | x < 10 (vocabulary size) | Prevent embedding index-out-of-bounds |
| Ethnicity | x < 50 (vocabulary size) | Prevent embedding index-out-of-bounds |

**Implementation**: `src/train/architectures/models/dualtower.py` → `DualTower.forward()` uses `torch.clamp` on categorical indices.

### 2. Numerical Sanitization

| Mechanism | Constraint | Purpose |
|-----------|-----------|---------|
| Negative rectification | x = max(0, x) | Rectify negative sensor errors |
| Percentile capping | x ≤ 99th percentile | Cap extreme statistical outliers |

**Implementation**: `src/train/data_pipeline/preprocessor.py` → `_fit_clip_outliers()` / `_transform_clip_outliers()`

Configuration via `src/train/config/_base/01_data.yaml`:

```yaml
data:
  features:
    clip:
      enabled: true       # Toggle the safeguard
      p_low: 0.001        # Lower percentile bound
      p_high: 0.999       # Upper percentile bound
```

---

## Project Structure

```
Dual_Tower_CCAI/
├── README.md                           # This file
├── LICENSE                             # CC BY-NC 4.0 + Medical AI Disclaimer
├── data/
│   └── train_ready/                    # Place preprocessed CSV splits here
│       └── DATA_README.md              # Data format instructions
├── outputs/                            # All experiment outputs (git-ignored)
│   ├── experiments/                    # Per-run experiment artifacts
│   ├── final/                          # Promoted best models
│   └── optuna/                         # Hyperparameter search results
└── src/
    └── train/
        ├── Makefile                    # Convenience commands (install, train, test)
        ├── requirements.txt            # Python dependencies
        ├── run_unified_train.py        # Main training entry point
        ├── run_multiple_seeds.py       # Multi-seed experiment runner
        ├── architectures/
        │   ├── model_factory.py        # Factory pattern for model instantiation
        │   └── models/
        │       ├── base.py             # NumericalFeatureTokenizer
        │       ├── dualtower.py        # ★ DualTower model (Transformer / MLP right tower)
        │       ├── mlp.py              # MLP baseline
        │       └── transformer.py      # Vanilla Transformer baseline
        ├── config/
        │   ├── _base/                  # Shared base configurations
        │   │   ├── 00_global.yaml      # Paths, logging, reproducibility, resources
        │   │   ├── 01_data.yaml        # Data splits, preprocessing, loader settings
        │   │   └── 02_train.yaml       # Training loop, early stopping, checkpointing
        │   └── models/                 # Per-model configuration overrides
        │       ├── dualtower.yaml      # DT-Transformer (proposed method)
        │       ├── dualtower_mlp.yaml  # DT-MLP (ablation study)
        │       ├── transformer.yaml    # Vanilla Transformer baseline
        │       ├── mlp.yaml            # MLP baseline
        │       ├── nn.yaml             # Simple neural network
        │       ├── xgboost.yaml        # XGBoost baseline
        │       └── random_forest.yaml  # RandomForest baseline
        ├── data_pipeline/
        │   ├── loader.py               # DataModule, DataLoader factory, trainer builder
        │   └── preprocessor.py         # Feature standardization, imputation, clipping
        ├── engine/
        │   ├── trainers.py             # NeuralTrainer & TreeTrainer
        │   ├── evaluator.py            # UnifiedEvaluator (metrics, CI, plots)
        │   ├── losses.py               # BCE, Focal, Label Smoothing
        │   ├── metrics.py              # AUROC, AUPRC, Brier, ECE, etc.
        │   ├── optimizers.py           # AdamW, SGD, etc.
        │   ├── schedulers.py           # Cosine, StepLR, ReduceOnPlateau, warmup
        │   ├── calibration.py          # Isotonic / Platt calibration
        │   ├── callbacks.py            # Early stopping, checkpointing
        │   └── utils_cast.py           # Mixed precision utilities
        ├── optuna/
        │   ├── run_optuna.py           # Optuna optimization entry point
        │   └── optuna_modules/
        │       ├── objective.py        # Trial objective function
        │       └── spaces.py           # Per-model search spaces
        ├── tests/
        │   └── smoke_test_cli.py       # CLI smoke tests
        └── utils/
            ├── config_manager.py       # Hierarchical YAML config with variable resolution
            ├── validator.py            # Data & config validation
            ├── exporter.py             # Model export, versioning, leaderboard
            ├── logger.py               # Structured logging (JSONL + console)
            ├── seed.py                 # Reproducibility (seed all RNGs)
            ├── inference.py            # Batch prediction & threshold selection
            ├── explainers.py           # SHAP & LIME wrappers
            ├── visualization.py        # Publication-style plots
            ├── statistical_tests.py    # DeLong test, McNemar, etc.
            ├── calibration.py          # CalibratorWrapper
            ├── registry.py             # ModelRegistry
            └── system_check.py         # Environment & dependency validation
```

---

## Data Preprocessing

> **This repository covers the model training pipeline only.**
> The data preprocessing pipeline (eICU raw data → train-ready CSV splits) is maintained in a separate repository:
>
> **🔗 [DATA PROJECT — Link to be added upon publication]**
>
> Please refer to that repository for:
> - Raw eICU data extraction and cohort selection
> - Feature engineering and missing value handling
> - Patient-stratified train/val/test splitting
>
> The preprocessed output files should be placed in `data/train_ready/`:
> ```
> data/train_ready/
> ├── train_patient_stratified.csv
> ├── val_patient_stratified.csv
> └── test_patient_stratified.csv
> ```
>
> **Format**: CSV with `patient_id` as the first column. The target column is `mortality` (binary: 0/1).

---

## Installation

All commands should be run from the `src/train/` directory:

```bash
cd src/train
```

### Option 1: Using Make (Recommended)

```bash
# Install all dependencies (CPU)
make install

# Verify installation
make validate
```

### Option 2: Using pip directly

```bash
pip install -r requirements.txt
```

### Option 3: GPU Support

```bash
# NVIDIA CUDA 11.8
make install-gpu

# Apple Silicon (MPS) — included by default
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### System Requirements

- **Python** >= 3.10
- **PyTorch** >= 2.0
- **RAM** >= 16 GB recommended
- **GPU** optional (CUDA 11.8+ or Apple MPS)

---

## Usage

> **Important**: All training commands must be executed from the `src/train/` directory.
> All output paths in the codebase are relative to this working directory (e.g., `../../outputs/` resolves to the project-root `outputs/` folder).

### Quick Start

```bash
cd src/train

# Run a quick smoke test
make test

# Train the proposed DT-Transformer
python run_unified_train.py --model dualtower

# Train with a specific seed
python run_unified_train.py --model dualtower --seed 42
```

### Training All Models

```bash
# DualTower Transformer (proposed method)
python run_unified_train.py --model dualtower

# DualTower MLP (ablation: right tower without self-attention)
python run_unified_train.py --model dualtower_mlp

# Baselines
python run_unified_train.py --model transformer
python run_unified_train.py --model mlp
python run_unified_train.py --model nn
python run_unified_train.py --model xgboost
python run_unified_train.py --model random_forest
```

### Multi-Seed Experiments (for Paper Reporting)

```bash
# Run DT-Transformer with 5 academic seeds → reports mean ± std
python run_multiple_seeds.py --model dualtower --seeds 5

# Run all 7 models × 5 seeds
python run_multiple_seeds.py --all --seeds 5

# Use custom seeds
python run_multiple_seeds.py --model dualtower --seeds 42,2024,777,9999,12345
```

### Hyperparameter Optimization

```bash
# Quick Optuna search (5 trials)
make optuna

# Full search (50 trials)
cd optuna && python run_optuna.py --model dualtower --n_trials 50
```

### Makefile Shortcuts

```bash
make help              # Show all available commands
make install           # Install dependencies
make test              # Run smoke tests
make validate          # System health check
make train             # Example MLP training
make train-xgb         # XGBoost training
make seeds             # Multi-seed experiment (3 seeds)
make clean             # Clean __pycache__ and .DS_Store
make clean-all         # Deep clean (cache + all outputs)
```

---

## Configuration

The project uses a **hierarchical YAML configuration system**:

```
config/
├── _base/
│   ├── 00_global.yaml    # Inherited by ALL models
│   ├── 01_data.yaml      # Data loading & preprocessing
│   └── 02_train.yaml     # Training loop defaults
└── models/
    └── dualtower.yaml     # Model-specific overrides
```

Base configs are automatically merged with model-specific configs via `ConfigManager`. Variable references like `${common.paths.project_root}` are resolved at load time.

To customize a run, either:
1. Edit the relevant YAML file in `config/models/`
2. Or pass a custom config: `python run_unified_train.py --model dualtower --config path/to/custom.yaml`

---

## Results

### Main Results (Table III from paper)

Results averaged over 5 random seeds. RF and XGBoost are deterministic (single run).

| Model | AUROC | AUPRC | F1 |
|-------|-------|-------|----|
| XGBoost | 0.8908 | 0.6467 | 0.5204 |
| RF | 0.8806 | 0.6236 | 0.5081 |
| **DT-Transformer** | **0.8848 ± 0.0034** | **0.6171 ± 0.0058** | **0.5401 ± 0.0172** |
| NN | 0.8582 ± 0.0018 | 0.5394 ± 0.0054 | 0.4954 ± 0.0183 |
| Standard Transformer | 0.8457 ± 0.0129 | 0.5279 ± 0.0195 | 0.4826 ± 0.0130 |
| Standard MLP | 0.8534 ± 0.0058 | 0.5170 ± 0.0081 | 0.4998 ± 0.0113 |

### Ablation Study (Table IV from paper)

| Architecture | AUPRC | AUROC | Relative Improvement |
|-------------|-------|-------|---------------------|
| DT-MLP | 0.5394 ± 0.0054 | 0.8645 ± 0.0027 | — |
| DT-Transformer | 0.6171 ± 0.0058 | 0.8848 ± 0.0034 | +14.41% |

> Full results with confidence intervals are generated automatically by `run_multiple_seeds.py` and saved to `outputs/multi_seed_results/`.

---

## Code Availability

This repository will be made **publicly available** upon official publication of the paper, in compliance with the Data Use Agreement (DUA) Section 9. During the review period, the repository is set to private.

---

## Paper

> **Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower Transformer Framework**
>
> Zhengrong Jia\* (Asia AI Education and Future Technology Association, Shenzhen, China)
> Kwong-Cheong Wong\* (School of Governance and Policy Science, The Chinese University of Hong Kong, Hong Kong SAR, China)
>
> \*Corresponding authors
>
> 📄 **[Paper Link — To be added upon publication]**

If you find this work useful, please cite:

```bibtex
@inproceedings{Jia2026DualTower,
  title     = {Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower Transformer Framework},
  author    = {Jia, Zhengrong and Wong, Kwong-Cheong},
  booktitle = {Proceedings of the CCAI Conference},
  year      = {2026},
  note      = {Under Review}
}
```

---

## Acknowledgment

We would like to express our gratitude to the eICU Collaborative Research Database team for making the multi-center critical care data publicly available, which was essential for this study. We also thank the Macau University of Science and Technology for providing access to the academic databases and literature resources that supported this research.

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). See the [LICENSE](LICENSE) file for full terms, including the Medical AI Disclaimer.
