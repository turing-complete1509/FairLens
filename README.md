# FairLens — Algorithmic Fairness Audit & Bias Mitigation Platform

> **Detect, measure, and mitigate demographic bias in machine learning pipelines — end to end.**

FairLens is a research-grade platform combining a Python ML pipeline with an interactive React dashboard. It ingests tabular datasets, evaluates fairness using state-of-the-art metrics, applies multiple imputation strategies, trains classifiers, and runs adversarial debiasing — all surfaced through a causal-inference-powered UI.

---

## Project Structure

```
FairLens/
├── dashboard/                  # React + Vite interactive audit dashboard
│   ├── src/
│   │   ├── components/         # Reusable UI components (sidebar, layout, cards)
│   │   ├── pages/              # Route-level pages (Upload, EDA, Imputation, Fairness Lab, Results)
│   │   ├── context/            # Global state (DataContext)
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Pure utilities (fairness metrics, data worker)
│   │   └── types/              # TypeScript type definitions
│   ├── public/
│   ├── index.html
│   └── package.json
│
├── pipeline/                   # Python ML research pipeline
│   ├── core/
│   │   ├── bias_engine.py          # BiasScanner, BiasAnalyzer, AntiBiasPipeline, Transformer debiaser
│   │   ├── bias_engine_pro.py      # ProAdversarialDebiaser with debiased dataset export
│   │   ├── replication_pipeline.py # Full end-to-end ELS dataset replication pipeline
│   │   ├── data_cleaning.py        # NCES codebook mapping, binarization, label encoding
│   │   ├── compare_features.py     # Subset vs. full-column model comparison
│   │   ├── feature_selection.py    # Mutual information justification report
│   │   └── verify_engine.py        # Synthetic bias verification test harness
│   │
│   ├── imputation/             # 11 standalone imputation strategy modules
│   │   ├── 01_mean_mode_imputation.py
│   │   ├── 02_knn_imputation.py
│   │   ├── 03_bayesian_imputation.py
│   │   ├── 04_regression_imputation.py
│   │   ├── 05_pmm_imputation.py
│   │   ├── 06_random_forest_imputation.py
│   │   ├── 07_mlr_bootstrap_imputation.py
│   │   ├── 08_mice_imputation.py
│   │   ├── 09_hot_cold_deck_imputation.py
│   │   ├── 10_carry_forward_backward_imputation.py
│   │   └── 11_interpolation_imputation.py
│   │
│   └── utils/
│       └── generate_biased_data.py # Synthetic biased hiring dataset generator
│
├── docs/
│   └── research/               # Reference research papers (PDF)
│       ├── VAINA-PRIMARY-2024.pdf
│       ├── bias 1.pdf
│       ├── bias 2.pdf
│       └── bias 3.pdf
│
├── requirements.txt
└── .gitignore
```

---

## Core Capabilities

### Python Pipeline (`pipeline/`)
| Module | Purpose |
|--------|---------|
| `bias_engine.py` | Fairness metrics (SPD, DI, EOD, AOD), bias type analysis, reweighting, Transformer adversarial debiaser |
| `bias_engine_pro.py` | Deep adversarial debiaser with debiased CSV export |
| `replication_pipeline.py` | NCES ELS dataset ingestion → 11 imputation methods → 3 classifiers → fairness evaluation |
| `data_cleaning.py` | Codebook-driven column selection, binarization, label encoding |
| `feature_selection.py` | Mutual information experiment: justifies the 29-column selection |
| `compare_features.py` | Compares subset (29 cols) vs full dataset (~4000 cols) model performance |
| `verify_engine.py` | Synthetic test bed: verifies bias detection + mitigation pipeline correctness |

### Fairness Metrics Computed
- **Statistical Parity Difference (SPD)**
- **Disparate Impact (DI)**
- **Equal Opportunity Difference (EOD)**
- **Average Odds Difference (AOD)**
- **Wasserstein Distance** (distribution shift)

### Bias Mitigation Strategies
1. **Pre-processing** — Targeted Importance Reweighting (TIW)
2. **In-processing** — Adversarial Debiasing (Zhang et al., 2018)
3. **In-processing (SOTA)** — Tabular Transformer with Multi-Head Attention + Fair Bottleneck
4. **Post-processing** — Optimal Transport distribution alignment

### Dashboard (`dashboard/`)
A full-featured React app for interactive bias auditing:

| Page | Description |
|------|-------------|
| **Upload** | Drag-and-drop CSV/XLSX ingest with worker-threaded parsing |
| **Overview** | Dataset statistics, missing value map, demographic breakdown |
| **Feature Selection** | Interactive column selection with justification |
| **EDA Dashboard** | Distribution plots, correlation heatmap, outlier detection |
| **Imputation Lab** | Compare 11 imputation strategies side-by-side |
| **Modeling Prep** | Feature engineering and preprocessing review |
| **Fairness Lab** | Causal DAG, SPD/DI gauges, counterfactual twin audit, adversarial mitigation |
| **Results & Insights** | Full audit report with exportable debiased dataset |

---

## Quick Start

### Dashboard
```bash
cd dashboard
npm install
npm run dev
```

### Python Pipeline
```bash
pip install -r requirements.txt

# Run the full NCES replication pipeline
python pipeline/core/replication_pipeline.py

# Verify the bias engine on synthetic data
python pipeline/core/verify_engine.py

# Generate a synthetic biased hiring dataset
python pipeline/utils/generate_biased_data.py
```

---

## Dataset

The Python pipeline is designed for the **Education Longitudinal Study (ELS:2002)** from NCES:
- File: `els_02_12_byf3pststu_v1_0.csv`
- Place it in the project root before running pipeline scripts.
- The dashboard accepts any CSV or Excel file.

---

## Research References

| Paper | Contribution |
|-------|-------------|
| Kusner et al. (2017) | Counterfactual Fairness via Structural Causal Models |
| Zhang et al. (2018) | Adversarial Learning for Fair Representations |
| Pearl (2016) | Do-Calculus & Causal Backdoor Criterion |
| VAINA (2024) | Bias in Educational Assessment Systems |

---

## Tech Stack

**Dashboard:** React 18, TypeScript, Vite, TailwindCSS, shadcn/ui, Recharts, Framer Motion, PapaParse  
**Pipeline:** Python 3.10+, PyTorch, scikit-learn, pandas, NumPy
