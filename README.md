# Exoplanet Missing-Planet GNN Pipeline

Code for: *Graph Neural Networks for Predicting Undetected Planets in
Multi-Planet Exoplanetary Systems*.

This has been smoke-tested end-to-end on synthetic data (correct shapes,
zero train/test system overlap confirmed, no leakage in the fold splitter).
It has **not** been run against the real NASA Exoplanet Archive yet — that
requires network access to `exoplanetarchive.ipac.caltech.edu`, which run
this on your own machine or Colab.

## Files
- `data_pipeline.py` — fetch archive data, filter complete 3+-planet systems,
  build per-system graphs, generate leave-one-out instances, system-level
  stratified k-fold splitting.
- `models.py` — GCN, GAT, GraphSAGE, GIN, and the Deep Sets non-graph baseline.
  All share one interface so they can be swapped in the training loop.
- `baselines.py` — naive heuristic, DYNAMITE-style statistical baseline,
  XGBoost on hand-engineered summary features.
- `run_baselines.py` — Phase 3: run all non-GNN baselines under the same
  fold protocol as the GNN pipeline. **Run this first.**
- `train.py` — Phase 4: nested cross-validation. Outer loop = honest test
  evaluation (touched once per fold). Inner loop = architecture/hyperparameter
  search, using ONLY a validation split — this is what keeps the search
  from leaking into your reported numbers.

## Setup (Colab or local)
```bash
pip install torch torch_geometric astroquery pandas numpy scikit-learn xgboost wandb
```

## Usage

### 1. Pull and prepare the data
```python
from data_pipeline import fetch_raw_data, filter_complete_multiplanet_systems

df = fetch_raw_data()  # hits the real archive via astroquery
df = filter_complete_multiplanet_systems(df, min_planets=3)
print(df['hostname'].nunique(), "usable systems")  # should be ~290
df.to_csv("exoplanet_clean.csv", index=False)  # cache it, don't re-fetch every run
```

### 2. Run baselines (Phase 3 — do this before any GNN training)
```python
from run_baselines import run_all_baselines
baseline_results = run_all_baselines(df, n_splits=5)
```

### 3. Run the GNN nested CV (Phase 4)
```python
import wandb
wandb.login()  # paste your API key once

from train import run_nested_cv
results = run_nested_cv(df, edge_mode="adjacent", n_outer=5, n_inner=3,
                         epochs=150, log_wandb=True)
```

This will:
- Split into 5 outer folds (system-level, stratified by planet count)
- For each outer fold, search across {gcn, gat, sage, gin, deepsets} × {hidden_dim} × {n_layers} on an inner validation split
- Retrain the best inner config on the full outer-train set
- Evaluate ONCE on the untouched outer-test fold
- Report mean ± std across all 5 outer folds — this is your headline result

### 4. Edge-definition ablation (Phase 5)
Re-run step 3 with `edge_mode="complete"` instead of `"adjacent"` and compare
the aggregated results — this isolates whether sparse (adjacent-only) vs.
fully-connected graphs matter.

### 5. Interpretability (Phase 5)
Use `GATModel.forward(..., return_attention=True)` on your best fold's model
to extract attention weights per edge — visualize which planet-pairs the
model relies on most.

## Known limitations to state in your thesis (already documented in your
problem statement)
- Small N: ~290 independent systems. Report mean ± std, not single numbers.
- Composite-parameter fitting leakage: real archive values for co-existing
  planets were often derived with joint knowledge of the full system, which
  can't be fully engineered around — acknowledge this rather than claim it's solved.
- `run_nested_cv`'s search space (`search_space_grid()` in `train.py`) is
  intentionally small for the smoke test. Expand it once you've confirmed
  the real run works end-to-end, but keep the outer/inner separation intact —
  do not evaluate additional configs against the outer test fold.

## Next steps once you have real data
1. Run step 1 above and confirm system counts match what we found (290 usable systems, filtered on 3+).
2. Run baselines first (step 2) — sanity check that DYNAMITE-style beats naive, as expected.
3. Only then move to the GNN nested CV (step 3).
