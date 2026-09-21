# Regionwise Bayesian Model Averaging (Regionwise BMA)

Code for the paper **"Uncertainty-Aware Regionwise Bayesian Model Averaging for Intelligent Decision Systems: Diagnosing and Repairing Covariance-Induced Weight Collapse"** (submitted to Knowledge-Based Systems).

## Overview

This repository implements **Regionwise BMA**, a closed-form framework for spatially adaptive Bayesian Model Averaging that addresses **Covariance-Estimation-Induced Weight Collapse (CEIWC)** — a statistical failure mode where localized covariance estimation suppresses minority-region expert weights.

### Key Components

1. **γ-τ Diagnostic Criterion** — Closed-form criterion for uncertainty-aware region splitting, approximating Bayesian Product Partition Models without MCMC
2. **Discrete-Event Gate** — Hierarchical Bayesian (truncated Beta) weight floor mechanism with provable risk degradation bounds
3. **Supermodular Subset Selection** — (1−1/e)-optimal greedy model pruning within regions
4. **Conditional PAC-Bayes Bounds** — Region-decomposed generalization guarantees

## Repository Structure

```
├── regionwise_bma_experiments.py   # Main implementation
├── exp_run.py                      # Full experiment pipeline (46 datasets)
├── exp_crossmodal.py               # Cross-modal high-dimensional stress test
├── exp_moe_ceiwc.py                # MoE router CEIWC verification
├── gen_phase_diagram.py            # Phase transition figure generation
├── gen_sensitivity_plot.py         # ε_min sensitivity analysis plot
├── figures/                        # Generated figures and results
│   ├── fig_phase_diagram.png
│   ├── fig_sensitivity.png
│   ├── fig3_collapse_gate.png
│   └── *.json                      # Experiment results
└── data/                           # Datasets (see below)
```

## Installation

### Requirements

```bash
python >= 3.8
numpy
scipy
scikit-learn
pandas
matplotlib
```

### Install Dependencies

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

## Quick Start

### 1. Load Data

Datasets are from the [KEEL Imbalanced Classification](https://sci2s.ugr.es/keel/imbalanced.php) repository. Place them in the `data/` directory in the standard KEEL format.

### 2. Run Main Experiments

```bash
python exp_run.py --datasets 46 --output figures/
```

This runs:
- Global BMA baseline
- Regionwise BMA (no gate)
- Regionwise BMA + Gate (ε_min = 0.15)
- Stratified analysis (Favorable vs. Challenging regime)
- Ablation study on representative datasets

### 3. Cross-Modal Stress Test

```bash
python exp_crossmodal.py --dim 200 --n_min 1000 --output figures/
```

Simulates high-dimensional embedding spaces to test CEIWC universality and sample-complexity boundaries.

### 4. Generate Figures

```bash
python gen_phase_diagram.py
python gen_sensitivity_plot.py
```

## Key Results

| Experiment | Metric | Value |
|------------|--------|-------|
| Creditcard fraud (Challenging regime) | ΔAUC (RW+Gate vs Global) | +6.3% |
| HTRU_2 (Favorable regime) | ΔAUC | +0.2% |
| γ-τ diagnostic correlation | Pearson r (vs CEIWC magnitude) | −0.722 (p < 0.001) |
| Full 46 datasets | ΔAUC (overall) | +0.2% (p = 0.31) |
| Favorable regime (n_min ≥ 500, N=15) | ΔAUC | +0.3% (p = 0.026) |
| Challenging regime (n_min < 500, N=31) | ΔAUC | −1.2% (p = 0.006) |
| PAC-Bayes bound tightness | vs naive global bound | 1.7–4.8× tighter |
| Greedy subset selection | Risk ratio (greedy/optimal) | 1.0015 |

## Core Algorithm

### Regionwise BMA Workflow

1. **Partition**: Use γ-τ criterion to recursively split regions based on error orthogonality
2. **Estimate**: Compute local error covariance Σ_k for each region
3. **Gate**: Apply weight floor ε_min to prevent CEIWC collapse
4. **Prune**: Greedy supermodular subset selection within each region
5. **Predict**: Region-weighted combination of base model predictions

### Base Model Pool

Default pool (M=5):
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM (RBF kernel)
- k-NN (k=7)

Extended pool (M=9):
- + Lasso, Ridge, k-NN (k=3)

## Citation

If you use this code, please cite:

```bibtex
@article{regionwise_bma2026,
  title={Uncertainty-Aware Regionwise Bayesian Model Averaging for Intelligent Decision Systems: Diagnosing and Repairing Covariance-Induced Weight Collapse},
  author={Anonymous},
  journal={Knowledge-Based Systems},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.
