"""
run_baselines.py
Evaluates all Phase-3 baselines under the SAME system-level stratified
k-fold protocol as the GNN pipeline, so results are directly comparable.
Run this BEFORE train.py's nested CV -- it validates your data pipeline
on simpler models first.
"""

import numpy as np
from data_pipeline import compute_normalization_stats, system_level_stratified_folds
from train import instances_for_hosts
from baselines import (
    naive_geometric_mean_baseline, dynamite_style_baseline,
    fit_population_log_period_ratio, train_xgboost_baseline, predict_xgboost_baseline,
)


def evaluate_predictions(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    hit_rate = float(np.mean(np.abs(preds - targets) < 0.5))
    return {"mse": mse, "mae": mae, "hit_rate_0.5std": hit_rate}


def run_all_baselines(df, edge_mode="adjacent", n_splits=5, seed=42):
    folds = system_level_stratified_folds(df, n_splits=n_splits, seed=seed)
    results = {"naive": [], "dynamite_style": [], "xgboost": []}

    for fold_i, (train_hosts, test_hosts) in enumerate(folds):
        print(f"\n=== Fold {fold_i + 1}/{n_splits} ===")
        stats = compute_normalization_stats(df, train_hosts)
        train_inst = instances_for_hosts(df, train_hosts, stats, edge_mode)
        test_inst = instances_for_hosts(df, test_hosts, stats, edge_mode)
        targets = [i["target_period_only"].item() for i in test_inst]

        # naive
        preds = [naive_geometric_mean_baseline(i, stats) for i in test_inst]
        m = evaluate_predictions(preds, targets)
        results["naive"].append(m)
        print(f"  naive: {m}")

        # DYNAMITE-style (population stat fit on TRAIN fold only)
        ratio_mean = fit_population_log_period_ratio(train_inst)
        preds = [dynamite_style_baseline(i, ratio_mean) for i in test_inst]
        m = evaluate_predictions(preds, targets)
        results["dynamite_style"].append(m)
        print(f"  dynamite_style: {m}")

        # XGBoost on hand-engineered summary features
        xgb_model = train_xgboost_baseline(train_inst)
        preds = predict_xgboost_baseline(xgb_model, test_inst)
        m = evaluate_predictions(preds, targets)
        results["xgboost"].append(m)
        print(f"  xgboost: {m}")

    print("\n=== Baseline summary (mean ± std across folds) ===")
    for name, fold_metrics in results.items():
        for key in ["mse", "mae", "hit_rate_0.5std"]:
            vals = [f[key] for f in fold_metrics]
            print(f"{name:15s} {key:15s} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return results


if __name__ == "__main__":
    print("Import and call run_all_baselines(df) with your loaded dataframe.")
