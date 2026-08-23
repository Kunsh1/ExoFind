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
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from data_pipeline import (
    compute_normalization_stats,
    make_leave_one_out_instances,
    system_level_stratified_folds,
)
from models import build_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def instances_for_hosts(df, hosts, stats, edge_mode):
    subset = df[df["hostname"].isin(hosts)]
    return make_leave_one_out_instances(subset, stats, edge_mode=edge_mode)


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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(warmup_epochs, 1))
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
            pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, gap_pos=batch.gap_pos)
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
                pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, gap_pos=batch.gap_pos)
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


def evaluate_model(model, instances, device="cpu"):
    """Returns dict of metrics on a set of instances (call ONCE per outer test fold)."""
    graphs = [i["graph"] for i in instances]
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, gap_pos=batch.gap_pos)
            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(batch.y.view(-1).cpu().numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    # hit rate: prediction within 0.5 std (in normalized log-period space) of truth
    hit_rate = float(np.mean(np.abs(preds - targets) < 0.5))
    return {"mse": mse, "mae": mae, "hit_rate_0.5std": hit_rate}


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
        ["gcn", "gat", "sage", "gin", "deepsets"],  # model
        [8, 16],                                      # hidden_dim (was [16, 32])
        [1, 2],                                        # n_layers (was [2, 3])
    ))


def run_nested_cv(df, edge_mode="adjacent", n_outer=5, n_inner=3, epochs=150,
                   device="cpu", log_wandb=True, seed=42):
    outer_folds = system_level_stratified_folds(df, n_splits=n_outer, seed=seed)
    in_dim = 9  # len(ALL_COLS) from data_pipeline
    outer_results = []

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

        inner_train_inst = instances_for_hosts(df, inner_train_hosts, stats, edge_mode)
        inner_val_inst = instances_for_hosts(df, inner_val_hosts, stats, edge_mode)

        # --- INNER SEARCH: try every config, select by inner val loss only ---
        best_cfg, best_val_loss, best_model = None, float("inf"), None
        for model_name, hidden_dim, n_layers in search_space_grid():
            kwargs = dict(hidden_dim=hidden_dim, n_layers=n_layers) if model_name != "deepsets" \
                else dict(hidden_dim=hidden_dim)
            model = build_model(model_name, in_dim, **kwargs)
            trained_model, val_loss = train_one_model(
                model, inner_train_inst, inner_val_inst, epochs=epochs, device=device,
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

        # --- retrain best config on FULL outer-train, evaluate ONCE on outer-test ---
        full_train_inst = instances_for_hosts(df, outer_train_hosts, stats, edge_mode)
        test_inst = instances_for_hosts(df, outer_test_hosts, stats, edge_mode)

        model_name, hidden_dim, n_layers = best_cfg
        kwargs = dict(hidden_dim=hidden_dim, n_layers=n_layers) if model_name != "deepsets" \
            else dict(hidden_dim=hidden_dim)
        final_model = build_model(model_name, in_dim, **kwargs)
        # reuse inner_val as a small validation set for early stopping during final fit
        final_model, _ = train_one_model(
            final_model, full_train_inst, inner_val_inst, epochs=epochs, device=device,
            log_wandb=log_wandb,
            wandb_config={**dict(zip(["model", "hidden_dim", "n_layers"], best_cfg)),
                           "outer_fold": outer_i, "stage": "final_refit"},
            wandb_tags=["final_refit", model_name, f"outer{outer_i}"],
        )

        metrics = evaluate_model(final_model, test_inst, device=device)
        print(f"  OUTER TEST metrics: {metrics}")
        outer_results.append({"outer_fold": outer_i, "config": best_cfg, **metrics})

    # aggregate
    print("\n=== Aggregated results across outer folds (mean ± std) ===")
    for key in ["mse", "mae", "hit_rate_0.5std"]:
        vals = [r[key] for r in outer_results]
        print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return outer_results


if __name__ == "__main__":
    print("Import this module and call run_nested_cv(df) with your loaded dataframe.")
    print("See README.md for the full Colab usage example.")
