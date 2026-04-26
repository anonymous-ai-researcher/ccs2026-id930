# The Price of Geometry: Minimax Optimal Privacy for Similarity Search over Access-Controlled Vector Databases

> **Anonymous submission to ACM CCS 2026**

## Overview

This repository contains the implementation and experimental evaluation for the paper *"The Price of Geometry."* We introduce **CPFG** (Coupled Permute-and-Flip-Gaussian), the first differentially private mechanism that jointly protects both the result set and the distance vector of $k$-nearest-neighbor queries over access-controlled vector databases.

**Key contributions:**
- Three leakage channels (distance skew, angular gap, triangulation) that exploit the geometry of filtered query results
- Tight minimax lower bounds for the hybrid discrete-continuous output space via Le Cam detection and the Hardt-Talwar ℓ₂ minimax theorem
- CPFG, a matching mechanism coupling permute-and-flip (set selection) with calibrated Gaussian noise (distance perturbation)
- CPFG\*, an instance-optimal variant using smooth sensitivity of ordered $k$-NN statistics
- Composition bounds with shuffling amplification

## Repository Structure

```
cpfg-mechanism/
├── cpfg/                    # Core mechanism implementation
│   ├── mechanism.py         # CPFG and CPFG* (Algorithm 1)
│   ├── pf.py                # Permute-and-flip (Definition 3)
│   ├── gaussian.py          # Gaussian noise calibration
│   ├── budget_split.py      # Optimal budget split (Eq. 1)
│   └── sensitivity.py       # Sensitivity computation (Lemma 1 + Lemma 3 smooth)
├── attacks/                 # Leakage channel implementations
│   ├── channel1.py          # KS test on distance distributions
│   ├── channel2.py          # Rayleigh test on angular gaps
│   ├── channel3.py          # MLE triangulation
│   └── evaluate.py          # AUC evaluation framework
├── baselines/               # B1-B5 baseline mechanisms (B0 = no defense; B6-B7 in cpfg/)
│   ├── gumbel_topk.py       # B1: Gumbel top-k
│   ├── joint_exp.py         # B2: Joint exponential mechanism
│   ├── pf_only.py           # B3: PF-only (no distance noise)
│   ├── gumbel_laplace.py    # B4: Gumbel + Laplace
│   └── gumbel_gaussian.py   # B5: Gumbel + Gaussian
├── experiments/             # All 10 experiments from Section 7
│   ├── run_all.sh           # One-command reproduction
│   ├── exp1_tradeoff.py     # Privacy-utility tradeoff (Figure 4)
│   ├── exp2_channel_closure.py  # Channel closure (Table 3)
│   ├── exp3_pf_vs_gumbel.py
│   ├── exp4_instance_optimal.py # Instance optimality (Figure 5)
│   ├── exp5_budget_split.py
│   ├── exp6_k_sensitivity.py
│   ├── exp7_dimension.py
│   ├── exp8_defense_aware.py
│   ├── exp9_composition.py      # Composition (Figure 6)
│   └── exp10_scalability.py
├── data/
│   └── preprocess.py        # Dataset preprocessing scripts
├── scripts/
│   ├── verify_sensitivity.py    # Machine-checked Lemma 1 (ℓ₂ sensitivity)
│   └── generate_tables.py       # LaTeX table generation
├── configs/
│   └── default.yaml         # Default experiment configuration
├── requirements.txt
└── setup.py
```

## Quick Start

### Installation

```bash
git clone https://anonymous.4open.science/r/cpfg-mechanism
cd cpfg-mechanism
pip install -e .
```

### Run All Experiments

```bash
bash experiments/run_all.sh
```

### Minimal Example

```python
from cpfg import CPFG
import numpy as np

# Create a vector database
n, d, k = 10000, 768, 10
db = np.random.randn(n, d).astype(np.float32)
query = np.random.randn(d).astype(np.float32)

# Access control: 30% restricted
authorized = np.random.rand(n) > 0.3

# Run CPFG with epsilon=1.0
mechanism = CPFG(epsilon=1.0, delta=1e-6, k=k)
result_set, distances = mechanism.query(query, db, authorized)

print(f"Selected {len(result_set)} neighbors")
print(f"Privatized distances: {distances[:5]}")
```

## Datasets

| Dataset | Vectors | Dim | Access Control | Source |
|---------|---------|-----|----------------|--------|
| MIMIC-IV | 331K | 768 | Real (department) | [PhysioNet](https://physionet.org/content/mimiciv/) (credentialed) |
| LegalBench | 25K | 1024 | Real (privilege) | [GitHub](https://github.com/HazyResearch/legalbench) (public) |
| MS MARCO | 8.8M | 768 | Synthetic | [Microsoft](https://microsoft.github.io/msmarco/) (public) |
| GloVe-200 | 1.2M | 200 | Synthetic | [Stanford](https://nlp.stanford.edu/projects/glove/) (public) |

### Data Preparation

```bash
# Public datasets (automatic download)
python data/preprocess.py --dataset msmarco
python data/preprocess.py --dataset glove
python data/preprocess.py --dataset legalbench

# MIMIC-IV (requires PhysioNet credentials)
# 1. Request access at https://physionet.org/content/mimiciv/
# 2. Download discharge summaries
# 3. Run:
python data/preprocess.py --dataset mimiciv --data_dir /path/to/mimiciv
```

## Reproducing Paper Results

Each experiment script corresponds to a result in the paper:

| Script | Paper Reference | Output |
|--------|----------------|--------|
| `exp1_tradeoff.py` | Figure 4 | Recall/DistErr vs ε |
| `exp2_channel_closure.py` | Table 3 | Channel AUCs for all baselines |
| `exp3_pf_vs_gumbel.py` | Section 7.4, Exp 3 | PF vs Gumbel Recall comparison |
| `exp4_instance_optimal.py` | Figure 5 | DistErr vs restriction fraction α |
| `exp5_budget_split.py` | Section 7.4, Exp 5 | Budget split sensitivity |
| `exp6_k_sensitivity.py` | Section 7.4, Exp 6 | Metrics vs k |
| `exp7_dimension.py` | Section 7.4, Exp 7 | Cross-dimension comparison |
| `exp8_defense_aware.py` | Section 7.4, Exp 8 | Aware vs unaware adversary |
| `exp9_composition.py` | Figure 6 | Composition cost vs queries |
| `exp10_scalability.py` | Section 7.5 | QPS and latency breakdown |

### Example: Reproduce Table 3

```bash
python experiments/exp2_channel_closure.py \
    --dataset mimiciv \
    --epsilon 1.0 \
    --k 10 \
    --n_queries 1000 \
    --n_runs 5 \
    --output results/table3.csv
```

## Verification Scripts

Machine-checked verification of key sensitivity bounds:

```bash
# Verify Lemma 1 (ℓ₂ sensitivity bound)
python scripts/verify_sensitivity.py

# Verify smooth sensitivity computation (Lemma 3)
python scripts/verify_sensitivity.py --smooth
```

## Configuration

Default parameters in `configs/default.yaml`:

```yaml
privacy:
  epsilon: 1.0
  delta: 1.0e-6
  k: 10

mechanism:
  budget_split: "optimal"    # or fixed ratio
  sensitivity: "worst_case"  # or "smooth" for CPFG*
  smooth_beta_factor: 0.25   # beta = epsilon_d / (4 * k * ln(2/delta))

evaluation:
  n_queries: 1000
  n_attack_runs: 5
  n_positive: 1000
  n_negative: 1000

index:
  type: "hnsw"
  ef_construction: 200
  M: 32
  ef_search: 100
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.11
- FAISS ≥ 1.7.4
- scikit-learn ≥ 1.3
- PyYAML ≥ 6.0
- pandas ≥ 2.0
- tqdm ≥ 4.65

Optional (for embedding generation):
- transformers ≥ 4.35
- torch ≥ 2.1

## License

This code is released for academic research under the MIT License.
