"""
data_pipeline.py
Fetches NASA Exoplanet Archive data, builds per-system graphs, and generates
leave-one-out (LOO) instances with system-level stratified k-fold splitting.

Run this once to produce a cached dataset file, then reuse it for all
downstream experiments (baselines, GNNs, ablations).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold

FEATURE_COLS = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_orbeccen", "pl_orbincl"]
STAR_COLS = ["st_mass", "st_rad", "st_teff"]
ALL_COLS = FEATURE_COLS + STAR_COLS


def fetch_raw_data():
    """Pull the full composite parameters table from the archive.
    Requires: pip install astroquery
    """
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,hostname,sy_pnum," + ",".join(ALL_COLS),
    )
    return table.to_pandas()


def filter_complete_multiplanet_systems(df: pd.DataFrame, min_planets: int = 3) -> pd.DataFrame:
    """Keep only systems with >= min_planets AND complete data on ALL_COLS
    (planet AND star columns) for every planet in the system.

    Star columns are filled with the system median first (star properties
    should be constant within a system; a few archive rows have inconsistent
    nulls). But if an ENTIRE system is missing star data, the per-system
    median is itself NaN and fillna does nothing -- that NaN would otherwise
    flow silently into the model and cause NaN loss on the first epoch.
    So completeness is checked on ALL_COLS (after fillna), not just
    FEATURE_COLS, to catch and drop those systems instead.
    """
    df = df.copy()

    for col in STAR_COLS:
        df[col] = df.groupby("hostname")[col].transform(lambda s: s.fillna(s.median()))

    def system_ok(g):
        return len(g) >= min_planets and g[ALL_COLS].notna().all(axis=None)

    keep_hosts = df.groupby("hostname").filter(system_ok)["hostname"].unique()
    return df[df["hostname"].isin(keep_hosts)].reset_index(drop=True)


def compute_normalization_stats(df: pd.DataFrame, hostnames_train):
    """Fit log-scale mean/std ONLY on training-fold systems. Never call this
    with test-fold data included -- that would be preprocessing leakage."""
    train_df = df[df["hostname"].isin(hostnames_train)]
    stats = {}
    for col in ALL_COLS:
        vals = np.log1p(train_df[col].values.astype(float))
        stats[col] = (vals.mean(), vals.std() + 1e-8)
    return stats


def normalize_row(row, stats):
    out = []
    for col in ALL_COLS:
        mu, sigma = stats[col]
        v = np.log1p(float(row[col]))
        z = (v - mu) / sigma
        # clip extreme outliers (e.g. Jupiter-mass planets, very wide orbits)
        # to prevent huge input magnitudes from destabilizing training --
        # this is what was causing NaN losses on real archive data
        z = np.clip(z, -5.0, 5.0)
        out.append(z)
    return np.array(out, dtype=np.float32)


def build_system_graph(system_df: pd.DataFrame, stats: dict, edge_mode: str = "adjacent"):
    """Build a single graph for one system, planets sorted by orbital period.

    edge_mode:
      'adjacent'  -> edges only between period-adjacent planets (sparse)
      'complete'  -> fully connected graph

    edge_attr now carries [log_period_ratio, log_mass_ratio] for each edge
    (src->dst), computed directly from the already-normalized log-space
    node features (x[:,0]=period, x[:,3]=mass), i.e. edge_attr = x[dst] - x[src]
    on those two columns. This gives the GNN the SAME relational signal
    DYNAMITE's population statistics use explicitly (period-ratio spacing),
    instead of forcing it to infer relationships purely from unweighted
    topology -- this was likely the main reason the GNN underperformed the
    baselines in earlier runs, since baselines get this signal directly.
    """
    sys_sorted = system_df.sort_values("pl_orbper").reset_index(drop=True)
    n = len(sys_sorted)

    x = np.stack([normalize_row(sys_sorted.iloc[i], stats) for i in range(n)])
    x = torch.tensor(x, dtype=torch.float)

    edges = []
    if edge_mode == "adjacent":
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # bidirectional
    elif edge_mode == "complete":
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i, j])
    else:
        raise ValueError(f"unknown edge_mode {edge_mode}")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
        else torch.empty((2, 0), dtype=torch.long)
    # NOTE: torch.tensor([]).t() gives the wrong shape ([0], not [2,0]) --
    # this matters once single-node graphs are allowed (2-planet systems
    # with one planet removed leave a single remaining node with zero edges)

    # PERIOD_IDX=0, MASS_IDX=3 in ALL_COLS -- see FEATURE_COLS ordering above
    PERIOD_IDX, MASS_IDX = 0, 3
    edge_attr = []
    for src, dst in edges:
        d_period = x[dst, PERIOD_IDX].item() - x[src, PERIOD_IDX].item()
        d_mass = x[dst, MASS_IDX].item() - x[src, MASS_IDX].item()
        edge_attr.append([d_period, d_mass])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, 2))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), sys_sorted


def make_leave_one_out_instances(df: pd.DataFrame, stats: dict, edge_mode: str = "adjacent",
                                  min_remaining: int = 2):
    """For every system, for every planet, build one LOO instance:
      - graph WITHOUT that planet (built on the remaining planets)
      - target = normalized feature vector of the REMOVED planet
      - hostname is carried through for system-level fold splitting later
    Only pl_orbper is used as the primary regression target by default;
    the full normalized vector is kept in case you want to predict more.

    min_remaining controls whether 2-planet systems are usable:
      - min_remaining=2 (default): requires 3+ planet systems, so every
        instance retains real graph structure (at least one edge). This is
        the primary, reportable evaluation set matching the problem statement.
      - min_remaining=1: ALSO includes 2-planet systems (remaining=1 planet,
        zero edges after removal). This roughly doubles available systems
        (711 two-planet systems exist vs 290 three-plus), but these
        instances carry NO relational signal for the GNN to use -- a single
        remaining planet gives the model nothing to reason about beyond
        "given one planet, guess where a plausible companion would sit,"
        which is a fundamentally different, much less constrained task.
        RECOMMENDATION: if you use min_remaining=1, do so for TRAINING
        volume only, and keep your primary reported test/evaluation metrics
        restricted to the min_remaining=2 (3+ planet) instances, ideally
        reported separately (stratify results by n_planets) so a reviewer
        can see the weak-signal instances aren't propping up your headline
        numbers.

    Any instance whose graph or target still contains NaN after
    normalization is skipped and counted -- this should be rare after
    filter_complete_multiplanet_systems, but acts as a hard safety net so a
    single bad row can't silently poison an entire training run.
    """
    instances = []
    skipped_nan = 0
    min_system_size = min_remaining + 1
    for hostname, sys_df in df.groupby("hostname"):
        n = len(sys_df)
        if n < min_system_size:
            continue
        sys_sorted = sys_df.sort_values("pl_orbper").reset_index(drop=True)
        for i in range(n):
            remaining = sys_sorted.drop(index=i).reset_index(drop=True)
            if len(remaining) < min_remaining:
                continue
            graph, _ = build_system_graph(remaining, stats, edge_mode=edge_mode)
            target_vec = normalize_row(sys_sorted.iloc[i], stats)

            if torch.isnan(graph.x).any() or np.isnan(target_vec).any():
                skipped_nan += 1
                continue

            # gap position: WHERE in the sequence the missing planet sits
            # (0 = innermost gap, 1 = outermost). Baselines (naive, DYNAMITE-
            # style) explicitly use removed_index to decide whether to
            # extrapolate inward/outward or interpolate -- the GNN previously
            # had no equivalent signal at all, since graph-level pooling
            # discards positional information entirely. This was likely a
            # significant reason for underperforming the baselines.
            n_remaining = len(remaining)
            gap_pos = i / max(n_remaining, 1)
            graph.gap_pos = torch.tensor([[gap_pos]], dtype=torch.float)

            # attach target directly to the Data object so torch_geometric's
            # DataLoader batches it automatically alongside x/edge_index
            graph.y = torch.tensor([target_vec[0]], dtype=torch.float)
            instances.append({
                "hostname": hostname,
                "removed_index": i,
                "n_planets": n,
                "graph": graph,
                "target": torch.tensor(target_vec, dtype=torch.float),
                "target_period_only": graph.y,
            })
    if skipped_nan > 0:
        print(f"  WARNING: skipped {skipped_nan} leave-one-out instances containing NaN "
              f"after normalization -- investigate the source system(s) if this number is large")
    return instances


def system_level_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Split at the SYSTEM level (not instance level) to avoid leakage,
    stratified by planet-count bucket so folds have proportional
    representation of 3/4/5/6/7/8-planet systems."""
    systems = df.drop_duplicates(subset="hostname")[["hostname", "sy_pnum"]].reset_index(drop=True)

    # bucket rare high-multiplicity systems together so StratifiedKFold doesn't
    # choke on classes with fewer members than n_splits
    def bucket(n):
        # cast everything to str -- mixing int and str labels in one column
        # makes sklearn's type_of_target() raise "Got 'unknown' instead"
        return str(n) if n <= 5 else "6plus"

    systems["bucket"] = systems["sy_pnum"].apply(bucket)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(systems["hostname"], systems["bucket"]):
        train_hosts = set(systems.iloc[train_idx]["hostname"])
        test_hosts = set(systems.iloc[test_idx]["hostname"])
        folds.append((train_hosts, test_hosts))
    return folds


if __name__ == "__main__":
    # Example usage (requires astroquery + network access, run this part locally/Colab)
    print("This module is meant to be imported. See README.md for the full pipeline usage.")
