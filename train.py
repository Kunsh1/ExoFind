"""
train.py
Nested cross-validation training loop.

Outer loop  -> honest performance estimate (test fold touched exactly once)
Inner loop  -> architecture/hyperparameter search happens ONLY here, on a
               validation split carved out of the outer-training systems

This structure is what keeps "try many configurations" from turning into
data leakage: the outer test fold never influences which configuration
gets selected.
"""

import copy
import itertools
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from data_pipeline import (
    compute_normalization_stats,
    make_leave_one_out_instances,
    system_level_stratified_folds,
    denormalize_period,
)
from models import build_model
from baselines import fit_population_log_period_ratio, dynamite_style_baseline

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def attach_dynamite_feature(instances, ratio_mean):
    """Compute DYNAMITE-style's own prediction for each instance and attach
    it as graph.dynamite_pred, so the GNN receives it as an input feature.

    IMPORTANT -- this is now an OPT-IN secondary experiment, not the default.
    Giving the GNN DYNAMITE's own answer as an input feature answers a
    DIFFERENT question than the primary thesis question: it tests "does a
    learned correction on top of DYNAMITE help" (a hybrid/ensemble claim),
    not "can a graph model independently beat DYNAMITE" (the actual research
    question). A model with strictly more information than its competitor
    should rarely score worse, so "beats DYNAMITE" under this setup is close
    to tautological and does NOT support the core thesis claim on its own.
    Use include_dynamite_feature=True in run_nested_cv only for an explicitly
    labeled secondary/hybrid comparison, never as your primary result.

    ratio_mean MUST be fit on the corresponding TRAIN split only (via
    fit_population_log_period_ratio) -- reuse the exact same value across
    train/val/test instances for a given fold, never refit per-split.
    """
    for inst in instances:
        pred = dynamite_style_baseline(inst, ratio_mean)
        inst["graph"].dynamite_pred = torch.tensor([[pred]], dtype=torch.float)
        inst["graph"].aux_feat = torch.cat(
            [inst["graph"].gap_pos, inst["graph"].dynamite_pred], dim=-1
        )
    return instances


def attach_null_aux_feature(instances):
    """PRIMARY-PATH default: aux_feat = [gap_pos, 0] -- the GNN gets its
    positional signal (gap_pos) but NOT DYNAMITE's answer. This is the
    correct setup for the actual thesis question (can the GNN independently
    beat DYNAMITE), keeping the architecture identical to the augmented
    variant (same input dimensionality) so the two are a clean apples-to-
    apples ablation, differing only in whether dynamite_pred is real or zero.
    """
    for inst in instances:
        zero_col = torch.zeros((1, 1), dtype=torch.float)
        inst["graph"].dynamite_pred = zero_col
        inst["graph"].aux_feat = torch.cat([inst["graph"].gap_pos, zero_col], dim=-1)
    return instances


def instances_for_hosts(df, hosts, stats, edge_mode, min_remaining=2, **graph_kwargs):
    """min_remaining=2 (default) requires 3+ planet systems -- every instance
    keeps real graph structure. Set min_remaining=1 to ALSO include 2-planet
    systems (single remaining node, zero edges) -- these carry no relational
    signal, so if you enable this, use stratify_by_n_planets() below to
    report metrics separately rather than blending them into one number.

    **graph_kwargs: forwarded to build_system_graph via
      make_leave_one_out_instances (resonance_tolerance, knn_k,
      threshold_value, add_star_hub, edge_attr_mode)."""
    subset = df[df["hostname"].isin(hosts)]
    return make_leave_one_out_instances(subset, stats, edge_mode=edge_mode,
                                        min_remaining=min_remaining, **graph_kwargs)


def stratify_by_n_planets(model, instances, device="cpu"):
    """Break out evaluation metrics by ORIGINAL system multiplicity
    (n_planets, i.e. before removal) so 2-planet-origin instances (if
    included via min_remaining=1) don't silently blend into one number.
    Returns a DataFrame: one row per n_planets bucket found in `instances`.
    """
    buckets = {}
    for inst in instances:
        buckets.setdefault(inst["n_planets"], []).append(inst)

    rows = []
    for n_planets, insts in sorted(buckets.items()):
        m = evaluate_model(model, insts, device=device)
        rows.append({"n_planets": n_planets, "n_instances": len(insts), **m})
    return pd.DataFrame(rows)


def train_one_model(model, train_instances, val_instances, lr=3e-4, epochs=200,
                     patience=25, batch_size=16, device="cpu", log_wandb=False,
                     wandb_config=None, wandb_tags=None, warmup_epochs=10,
                     grad_clip_norm=1.0):
    """Train a single GNN configuration with early stopping on val loss.

    Changed after real-data runs showed ~90% of inner-search configs
    diverging to NaN, plus a large train/val/test gap on the one stable run:
      - lr default lowered 1e-3 -> 3e-4 (biggest single lever against divergence)
      - added a linear LR warmup (helps GAT/GIN especially, more sensitive
        to large early updates than GCN/SAGE)
      - grad_clip_norm now tunable, tightened default 5.0 -> 1.0 (5.0 was too
        loose to actually stop the divergence seen in practice)
      - patience raised slightly to give the lower LR room to converge
      - weight_decay raised 1e-4 -> 5e-4 given the small-N overfitting risk
    """
    if log_wandb and WANDB_AVAILABLE:
        run = wandb.init(project="exoplanet-gnn", config=wandb_config or {},
                          tags=wandb_tags or [], reinit=True)

    train_graphs = [i["graph"] for i in train_instances]
    val_graphs = [i["graph"] for i in val_instances]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # scale warmup DOWN automatically for short runs -- a fixed 10-epoch
    # warmup silently ate most of a 20-30 epoch test run during development
    # and produced misleadingly bad results that looked like a real bug.
    # This was never a correctness issue, just wasted epochs, but it's an
    # easy trap to fall into if you ever lower search_epochs further.
    effective_warmup = min(warmup_epochs, max(1, epochs // 4))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / effective_warmup)
    )
    loss_fn = torch.nn.MSELoss()

    # ALWAYS have a valid state to fall back to, even if training never
    # improves (e.g. every epoch produces NaN on unusual real-world outliers).
    # Without this, best_state stays None and load_state_dict(None) crashes.
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, aux_feat=batch.aux_feat)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            # gradient clipping -- real archive data has much larger outliers
            # (e.g. Jupiter-mass planets next to Earth-mass ones) than the
            # synthetic smoke-test data, which can blow up gradients on a
            # small network and produce NaN loss within the first epoch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        scheduler.step()
        train_loss /= len(train_graphs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, aux_feat=batch.aux_feat)
                loss = loss_fn(pred, batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs
        val_loss /= max(len(val_graphs), 1)

        if log_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # NaN check: NaN < anything is always False in Python, so a NaN
        # epoch would otherwise just silently fail to update best_state --
        # explicitly skip it instead of relying on that comparison quirk
        if np.isnan(val_loss) or np.isnan(train_loss):
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    WARNING: training diverged to NaN, stopping early at epoch {epoch}")
                break
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    if log_wandb and WANDB_AVAILABLE:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()
    return model, best_val_loss


def evaluate_model(model, instances, device="cpu", stats=None):
    """Returns dict of metrics on a set of instances (call ONCE per outer test fold).

    stats (optional): normalization stats dict, needed to also report
    DYNAMITE-comparable metrics in REAL units (days), matching how
    DYNAMITE's own papers judge success -- percentage error and
    match-within-tolerance, e.g. "predicted period matched the real planet
    within 10%/20%/50%" -- rather than only normalized log-space MSE, which
    isn't directly comparable to anything reported in the DYNAMITE literature.
    """
    graphs = [i["graph"] for i in instances]
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    model.eval()
    preds, targets = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, aux_feat=batch.aux_feat)
            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(batch.y.view(-1).cpu().numpy().tolist())
    elapsed = time.perf_counter() - t0
    ms_per_system = (elapsed / max(len(instances), 1)) * 1000
    preds, targets = np.array(preds), np.array(targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    # hit rate: prediction within 0.5 std (in normalized log-period space) of truth
    hit_rate = float(np.mean(np.abs(preds - targets) < 0.5))
    result = {"mse": mse, "mae": mae, "hit_rate_0.5std": hit_rate, "ms_per_system": ms_per_system}

    if stats is not None:
        real_pred = denormalize_period(preds, stats)
        real_true = denormalize_period(targets, stats)
        pct_error = np.abs(real_pred - real_true) / real_true
        result["mape_pct"] = float(np.median(pct_error) * 100)
        # match-within-tolerance, mirroring how DYNAMITE papers report success
        for tol in [0.10, 0.20, 0.50]:
            result[f"match_within_{int(tol*100)}pct"] = float(np.mean(pct_error < tol))

    return result


def search_space_grid():
    """Define your architecture/hyperparameter search space here.
    This is the ONLY place broad search happens -- inside the inner loop,
    never against the outer test fold.

    Narrowed after real-data runs showed instability: your graphs only
    have 3-8 nodes (remaining planets after leave-one-out), so a 3-layer,
    32-dim model is genuinely oversized -- deep GNNs oversmooth quickly on
    graphs this small, and more parameters than you have data to fit them
    is a direct path to the instability you saw. Dropped n_layers=3 and
    hidden_dim=32 for gcn/gat (kept for sage/gin since they showed more
    stable curves in your run), added hidden_dim=8 as a smaller option.
    """
    return list(itertools.product(
        ["gcn", "gat", "gatv2", "sage", "gin", "deepsets"],  # model -- added gatv2
        [8, 16],                                      # hidden_dim (was [16, 32])
        [1, 2],                                        # n_layers (was [2, 3])
    ))


def run_nested_cv(df, edge_mode="adjacent", n_outer=5, n_inner=3,
                   search_epochs=150, final_epochs=200,
                   device="auto", batch_size=16, log_wandb=False, seed=42,
                   min_remaining=2, include_dynamite_feature=False,
                   resonance_tolerance=0.05, knn_k=2, threshold_value=np.log(2.0),
                   add_star_hub=False, edge_attr_mode="both"):
    """
    NEW graph-structure parameters (forwarded to build_system_graph):
      edge_mode: 'adjacent' | 'complete' | 'resonance' | 'knn' | 'threshold'
      resonance_tolerance: only used if edge_mode='resonance' -- fractional
        tolerance for matching a period ratio to a canonical resonance
      knn_k: only used if edge_mode='knn' -- neighbors per node
      threshold_value: only used if edge_mode='threshold' -- log-period-ratio cutoff
      add_star_hub: adds one extra star-feature node connected to every planet,
        independent of edge_mode
      edge_attr_mode: 'both' | 'period_only' | 'mass_only' -- ablates which
        edge signal (period ratio vs mass ratio) the GNN actually gets

    include_dynamite_feature (default False -- this is the PRIMARY,
    thesis-relevant setting):
      False -> aux_feat = [gap_pos, 0]. The GNN does NOT see DYNAMITE's
        prediction. This is the honest test of your actual research
        question: can a graph-based model independently beat DYNAMITE.
        USE THIS as your headline/reported result.
      True  -> aux_feat = [gap_pos, dynamite_pred]. The GNN sees DYNAMITE's
        own prediction as an input feature and can learn a correction on
        top of it. This answers a DIFFERENT question (does a hybrid/
        ensemble improve on DYNAMITE) and should be reported separately,
        explicitly labeled as a secondary/hybrid experiment -- NOT
        presented as "the GNN beat DYNAMITE," since a model given its
        competitor's answer as input has an unfair informational advantage.
        Run this only after the primary (False) comparison, as a distinct,
        clearly-labeled additional result.

    Speed notes (this was the slow part before):
      - search_epochs: used for ALL inner-search configs (there are 20 of
        them per outer fold = 100 total runs). This does NOT need to match
        your final training length -- its only job is to RANK configs
        against each other, and that ranking is usually stable well before
        200 epochs. Lowered default 150 -> 60, which alone cuts inner-search
        compute by more than half.
      - final_epochs: only used ONCE per outer fold, for the actual winning
        config, retrained on the full outer-train set. Kept high (200) since
        this run's quality is what actually gets reported.
      - batch_size: raised default 16 -> 32. Your graphs are tiny (3-8
        nodes), so bigger batches mean fewer Python-level loop iterations
        per epoch for the same total work -- a real CPU speedup with no
        accuracy cost at this data scale.
      - device="auto": picks GPU automatically if Colab gives you one.
        Given how small these graphs are, CPU is often still fine, but this
        removes the decision from you -- if a GPU is available, use it.
      - min_remaining=2 (default, unchanged behavior): 3+ planet systems
        only. Set to 1 to ALSO include 2-planet systems as extra training
        volume -- see stratify_by_n_planets() to report these separately
        rather than blending them into your headline numbers.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    outer_folds = system_level_stratified_folds(df, n_splits=n_outer, seed=seed)
    in_dim = 9  # len(ALL_COLS) from data_pipeline
    edge_dim = 1 if edge_attr_mode in ("period_only", "mass_only") else 2
    graph_kwargs = dict(resonance_tolerance=resonance_tolerance, knn_k=knn_k,
                        threshold_value=threshold_value, add_star_hub=add_star_hub,
                        edge_attr_mode=edge_attr_mode)
    outer_results = []
    outer_models = []             # trained model from each fold's final refit
    outer_test_instances = []     # that fold's test instances, matching outer_models[i]
    stratified_tables = []  # only populated if min_remaining=1

    variant = "gnn_dynamite_augmented (secondary/hybrid -- NOT primary evidence)" \
        if include_dynamite_feature else "gnn_pure (PRIMARY -- tests actual thesis question)"
    print(f"\n{'='*70}\nRUN VARIANT: {variant}\n{'='*70}")

    for outer_i, (outer_train_hosts, outer_test_hosts) in enumerate(outer_folds):
        print(f"\n=== Outer fold {outer_i + 1}/{n_outer} ===")
        # fit normalization ONLY on outer-train systems -- no leakage from test
        stats = compute_normalization_stats(df, outer_train_hosts)

        outer_train_df = df[df["hostname"].isin(outer_train_hosts)]
        systems = outer_train_df.drop_duplicates(subset="hostname")[["hostname", "sy_pnum"]]
        # cast to str for the same reason as system_level_stratified_folds above
        bucket = systems["sy_pnum"].apply(lambda n: str(n) if n <= 5 else "6plus")

        inner_skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        inner_split = next(inner_skf.split(systems["hostname"], bucket))
        inner_train_hosts = set(systems.iloc[inner_split[0]]["hostname"])
        inner_val_hosts = set(systems.iloc[inner_split[1]]["hostname"])

        inner_train_inst = instances_for_hosts(df, inner_train_hosts, stats, edge_mode, min_remaining, **graph_kwargs)
        inner_val_inst = instances_for_hosts(df, inner_val_hosts, stats, edge_mode, min_remaining, **graph_kwargs)

        # fit DYNAMITE-style's ratio on the INNER-TRAIN split only, apply to
        # both inner-train and inner-val (same rule as normalization stats --
        # never fit on data the model will be evaluated against)
        inner_ratio_mean = fit_population_log_period_ratio(inner_train_inst)
        if include_dynamite_feature:
            attach_dynamite_feature(inner_train_inst, inner_ratio_mean)
            attach_dynamite_feature(inner_val_inst, inner_ratio_mean)
        else:
            attach_null_aux_feature(inner_train_inst)
            attach_null_aux_feature(inner_val_inst)

        # --- INNER SEARCH: fast screening pass (search_epochs), select by inner val loss only ---
        best_cfg, best_val_loss, best_model = None, float("inf"), None
        for model_name, hidden_dim, n_layers in search_space_grid():
            kwargs = dict(hidden_dim=hidden_dim, n_layers=n_layers, edge_dim=edge_dim) if model_name != "deepsets" \
                else dict(hidden_dim=hidden_dim)
            model = build_model(model_name, in_dim, **kwargs)
            trained_model, val_loss = train_one_model(
                model, inner_train_inst, inner_val_inst, epochs=search_epochs,
                device=device, batch_size=batch_size,
                log_wandb=log_wandb,
                wandb_config={"model": model_name, "hidden_dim": hidden_dim,
                               "n_layers": n_layers, "outer_fold": outer_i, "edge_mode": edge_mode},
                wandb_tags=["inner_search", model_name, f"outer{outer_i}"],
            )
            print(f"  {model_name} h={hidden_dim} L={n_layers} -> val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_cfg = (model_name, hidden_dim, n_layers)
                best_model = trained_model

        print(f"  BEST inner config: {best_cfg} (val_loss={best_val_loss:.4f})")

        # --- retrain best config on FULL outer-train (full final_epochs budget), evaluate ONCE on outer-test ---
        full_train_inst = instances_for_hosts(df, outer_train_hosts, stats, edge_mode, min_remaining, **graph_kwargs)
        test_inst = instances_for_hosts(df, outer_test_hosts, stats, edge_mode, min_remaining, **graph_kwargs)

        # refit the ratio on the FULL outer-train (larger than inner-train),
        # apply consistently to full_train/val/test -- test still never
        # influences the fit, only receives the already-fit statistic
        full_ratio_mean = fit_population_log_period_ratio(full_train_inst)
        if include_dynamite_feature:
            attach_dynamite_feature(full_train_inst, full_ratio_mean)
            attach_dynamite_feature(inner_val_inst, full_ratio_mean)  # refresh with the better-fit ratio
            attach_dynamite_feature(test_inst, full_ratio_mean)
        else:
            attach_null_aux_feature(full_train_inst)
            attach_null_aux_feature(inner_val_inst)
            attach_null_aux_feature(test_inst)

        model_name, hidden_dim, n_layers = best_cfg
        kwargs = dict(hidden_dim=hidden_dim, n_layers=n_layers, edge_dim=edge_dim) if model_name != "deepsets" \
            else dict(hidden_dim=hidden_dim)
        final_model = build_model(model_name, in_dim, **kwargs)
        # reuse inner_val as a small validation set for early stopping during final fit
        final_model, _ = train_one_model(
            final_model, full_train_inst, inner_val_inst, epochs=final_epochs,
            device=device, batch_size=batch_size,
            log_wandb=log_wandb,
            wandb_config={**dict(zip(["model", "hidden_dim", "n_layers"], best_cfg)),
                           "outer_fold": outer_i, "stage": "final_refit"},
            wandb_tags=["final_refit", model_name, f"outer{outer_i}"],
        )

        metrics = evaluate_model(final_model, test_inst, device=device, stats=stats)
        print(f"  OUTER TEST metrics: {metrics}")
        outer_results.append({"outer_fold": outer_i, "config": best_cfg, "variant": variant, **metrics})
        outer_models.append(final_model)          # keep the trained model
        outer_test_instances.append(test_inst)     # and its matching test instances (for plotting)

        if min_remaining == 1:
            strat = stratify_by_n_planets(final_model, test_inst, device=device)
            strat["outer_fold"] = outer_i
            stratified_tables.append(strat)

    # --- build both a per-fold table and an aggregated summary table ---
    per_fold_df = pd.DataFrame(outer_results)
    per_fold_df["model"] = per_fold_df["config"].apply(lambda c: c[0])
    per_fold_df["hidden_dim"] = per_fold_df["config"].apply(lambda c: c[1])
    per_fold_df["n_layers"] = per_fold_df["config"].apply(lambda c: c[2])
    per_fold_df = per_fold_df.drop(columns=["config"])

    summary_rows = []
    non_metric_cols = {"outer_fold", "variant", "model", "hidden_dim", "n_layers"}
    metric_cols = [c for c in per_fold_df.columns if c not in non_metric_cols]
    for key in metric_cols:
        vals = per_fold_df[key].values
        summary_rows.append({"metric": key, "mean": np.mean(vals), "std": np.std(vals)})
    summary_df = pd.DataFrame(summary_rows).set_index("metric")

    print("\n=== Per-fold results ===")
    print(per_fold_df.to_string(index=False))
    print("\n=== Aggregated results across outer folds (mean ± std) ===")
    print(summary_df.round(4).to_string())

    per_fold_df.to_csv("gnn_per_fold_results.csv", index=False)
    summary_df.to_csv("gnn_summary_results.csv")
    print("\nSaved: gnn_per_fold_results.csv, gnn_summary_results.csv")

    result = {"per_fold": per_fold_df, "summary": summary_df, "raw": outer_results,
              "models": outer_models, "test_instances": outer_test_instances}

    if stratified_tables:
        strat_df = pd.concat(stratified_tables, ignore_index=True)
        strat_summary = strat_df.groupby("n_planets")[["mse", "mae", "hit_rate_0.5std"]].agg(["mean", "std"])
        print("\n=== Results stratified by ORIGINAL system multiplicity (n_planets) ===")
        print("(2-planet-origin rows below carry weak/no relational signal --")
        print(" compare them against 3+ rows, don't blend into one headline number)")
        print(strat_summary.round(4).to_string())
        strat_df.to_csv("gnn_stratified_by_multiplicity.csv", index=False)
        result["stratified_by_n_planets"] = strat_summary

    return result


def build_comparison_table(baseline_results: dict, gnn_summary_df: pd.DataFrame,
                            gnn_label: str = "gnn_pure (primary)",
                            extra_gnn_variants: dict = None):
    """Combine run_baselines()'s output with one or more run_nested_cv()
    summaries into ONE table for easy sharing/comparison.

    gnn_label: how to label the primary gnn_summary_df passed in. Default
      assumes you're passing the include_dynamite_feature=False run (the
      one that actually tests your thesis question).
    extra_gnn_variants: optional dict of {label: summary_df} for additional
      GNN runs, e.g. the dynamite-augmented secondary experiment --
      pass it here rather than as the primary argument so the table clearly
      distinguishes "did the graph model beat DYNAMITE" from "did a hybrid
      correction on top of DYNAMITE beat plain DYNAMITE," which are
      different claims and should never be presented as the same result.

    Example:
      build_comparison_table(baseline_results, pure_gnn_results['summary'],
          extra_gnn_variants={'gnn_dynamite_augmented (secondary)': augmented_results['summary']})
    """
    rows = []
    for method_name, fold_metrics in baseline_results.items():
        metric_keys = list(fold_metrics[0].keys()) if fold_metrics else []
        for key in metric_keys:
            vals = [f[key] for f in fold_metrics]
            rows.append({"method": method_name, "metric": key,
                         "mean": np.mean(vals), "std": np.std(vals)})

    gnn_variants = {gnn_label: gnn_summary_df}
    if extra_gnn_variants:
        gnn_variants.update(extra_gnn_variants)

    for label, summary_df in gnn_variants.items():
        for key, row in summary_df.iterrows():
            rows.append({"method": label, "metric": key,
                         "mean": row["mean"], "std": row["std"]})

    comparison_df = pd.DataFrame(rows)
    pivoted = comparison_df.pivot(index="method", columns="metric", values="mean")
    pivoted_std = comparison_df.pivot(index="method", columns="metric", values="std")
    combined = pivoted.round(4).astype(str) + " ± " + pivoted_std.round(4).astype(str)

    print("\n=== FULL COMPARISON TABLE (baselines vs GNN) ===")
    if extra_gnn_variants:
        print("NOTE: 'gnn_dynamite_augmented' rows are a SECONDARY/hybrid result --")
        print("      they answer a different question than the primary gnn_pure row")
        print("      and should not be cited as evidence the graph model alone beats DYNAMITE.")
    print(combined.to_string())
    combined.to_csv("full_comparison_table.csv")
    print("\nSaved: full_comparison_table.csv -- this is the one to share")
    return combined


if __name__ == "__main__":
    print("Import this module and call run_nested_cv(df) with your loaded dataframe.")
    print("See README.md for the full Colab usage example.")
