"""
baselines.py
Non-GNN baselines, evaluated under the exact same leave-one-out /
system-level fold protocol as the GNN models. These MUST be run first
in Phase 3 to validate the data/eval pipeline before any GNN code runs.

All baselines predict normalized log(period) of the removed planet,
matching the GNN models' primary target for a fair comparison.
"""

import numpy as np
import xgboost as xgb


def naive_geometric_mean_baseline(instance, stats):
    """Predict the missing planet's period as the geometric mean of the
    periods of its two nearest neighbors in the remaining (sorted) graph.
    If it was the innermost/outermost planet, use the single nearest neighbor.
    Operates directly on normalized log-period node features (x[:, 0]),
    so 'geometric mean' in real units = arithmetic mean in log space.
    """
    x = instance["graph"].x.numpy()
    log_periods = x[:, 0]  # first feature column is pl_orbper (log-normalized)
    idx = instance["removed_index"]
    n_remaining = len(log_periods)

    if idx == 0:
        pred = log_periods[0]  # was innermost, use new innermost as proxy
    elif idx >= n_remaining:
        pred = log_periods[-1]  # was outermost
    else:
        pred = (log_periods[idx - 1] + log_periods[idx]) / 2.0
    return float(pred)


def dynamite_style_baseline(instance, population_log_period_ratio_mean):
    """Simplified reimplementation of DYNAMITE's core logic: use the
    POPULATION-level mean log period-ratio (fit once on the training
    fold only) to project outward/inward from the nearest known neighbor,
    rather than just averaging neighbors like the naive baseline does.

    population_log_period_ratio_mean must be computed on TRAIN-FOLD
    systems only -- pass it in, don't compute it here, to avoid leakage.
    """
    x = instance["graph"].x.numpy()
    log_periods = x[:, 0]
    idx = instance["removed_index"]
    n_remaining = len(log_periods)

    if idx == 0:
        pred = log_periods[0] - population_log_period_ratio_mean
    elif idx >= n_remaining:
        pred = log_periods[-1] + population_log_period_ratio_mean
    else:
        # average the inward and outward projections
        inward = log_periods[idx - 1] + population_log_period_ratio_mean
        outward = log_periods[idx] - population_log_period_ratio_mean
        pred = (inward + outward) / 2.0
    return float(pred)


def fit_population_log_period_ratio(train_instances):
    """Compute mean log period-ratio between adjacent planets across all
    TRAINING systems only. Use this to parametrize dynamite_style_baseline."""
    ratios = []
    for inst in train_instances:
        x = inst["graph"].x.numpy()
        lp = x[:, 0]
        if len(lp) >= 2:
            ratios.extend(np.diff(np.sort(lp)))
    return float(np.mean(ratios)) if ratios else 0.5


def extract_summary_features(instance):
    """Hand-engineered system-level summary features for the XGBoost
    baseline -- this mirrors the SPOCK-style 'collapse system to a flat
    feature vector' approach, as the direct architectural contrast to
    the GNN's learned relational representation."""
    x = instance["graph"].x.numpy()
    log_periods = np.sort(x[:, 0])
    n = len(log_periods)
    idx = instance["removed_index"]

    spacing = np.diff(log_periods) if n > 1 else np.array([0.0])
    feats = {
        "n_remaining": n,
        "mean_spacing": spacing.mean(),
        "std_spacing": spacing.std() if n > 2 else 0.0,
        "min_period": log_periods.min(),
        "max_period": log_periods.max(),
        "removed_position_frac": idx / max(n, 1),
        "nearest_neighbor_period": log_periods[min(idx, n - 1)],
        "mean_eccen": x[:, 4].mean(),
        "mean_mass": x[:, 3].mean(),
    }
    return feats


def train_xgboost_baseline(train_instances):
    import pandas as pd

    X = pd.DataFrame([extract_summary_features(i) for i in train_instances])
    y = np.array([i["target_period_only"].item() for i in train_instances])
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model


def predict_xgboost_baseline(model, instances):
    import pandas as pd

    X = pd.DataFrame([extract_summary_features(i) for i in instances])
    return model.predict(X)
