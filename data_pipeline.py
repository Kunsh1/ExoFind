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

# Optional, off-by-default extension -- most archive columns are metadata
# (error bars, provenance IDs, formatting strings, photometric magnitudes
# unrelated to orbital dynamics) and were deliberately excluded. These few
# ARE physically relevant and reasonably well-populated, but weren't in the
# original feature set. Not enabled by default because adding them requires
# re-running the completeness filter (fewer systems will have ALL of these
# populated) and re-validating the pipeline -- opt in explicitly.
EXTENDED_FEATURE_COLS = ["pl_eqt", "pl_insol"]      # equilibrium temp, insolation flux (planet-level)
EXTENDED_STAR_COLS = ["st_met", "st_age", "st_logg"]  # metallicity, age, surface gravity (star-level)


def get_columns(use_extended: bool = False):
    """Single source of truth for which columns are active. Everything
    downstream (fetch, filter, normalize, graph-building, model in_dim)
    should derive from this rather than hardcoding column lists, so turning
    use_extended on/off can never silently desync one part of the pipeline
    from another.
    """
    feature_cols = FEATURE_COLS + (EXTENDED_FEATURE_COLS if use_extended else [])
    star_cols = STAR_COLS + (EXTENDED_STAR_COLS if use_extended else [])
    return feature_cols, star_cols, feature_cols + star_cols


def fetch_raw_data(use_extended: bool = False):
    """Pull the full composite parameters table from the archive.
    Requires: pip install astroquery

    use_extended=True also pulls pl_eqt, pl_insol, st_met, st_age, st_logg
    (see EXTENDED_FEATURE_COLS/EXTENDED_STAR_COLS above for rationale).
    """
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    _, _, all_cols = get_columns(use_extended)
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,hostname,sy_pnum," + ",".join(all_cols),
    )
    return table.to_pandas()


def filter_complete_multiplanet_systems(df: pd.DataFrame, min_planets: int = 3,
                                        use_extended: bool = False) -> pd.DataFrame:
    """Keep only systems with >= min_planets AND complete data on the active
    column set (planet AND star columns) for every planet in the system.

    Star columns are filled with the system median first (star properties
    should be constant within a system; a few archive rows have inconsistent
    nulls). But if an ENTIRE system is missing star data, the per-system
    median is itself NaN and fillna does nothing -- that NaN would otherwise
    flow silently into the model and cause NaN loss on the first epoch.
    So completeness is checked on the full active column set (after fillna),
    not just FEATURE_COLS, to catch and drop those systems instead.

    use_extended=True: with more required columns, expect FEWER systems to
    pass the completeness filter than the original ~290 (3+ planet) baseline
    -- e.g. metallicity/age aren't measured for every star. Check
    df['hostname'].nunique() after filtering and compare before committing
    to the extended set as your primary pipeline.
    """
    df = df.copy()
    _, star_cols, all_cols = get_columns(use_extended)

    for col in star_cols:
        df[col] = df.groupby("hostname")[col].transform(lambda s: s.fillna(s.median()))

    def system_ok(g):
        return len(g) >= min_planets and g[all_cols].notna().all(axis=None)

    keep_hosts = df.groupby("hostname").filter(system_ok)["hostname"].unique()
    return df[df["hostname"].isin(keep_hosts)].reset_index(drop=True)


def compute_normalization_stats(df: pd.DataFrame, hostnames_train, use_extended: bool = False):
    """Fit log-scale mean/std ONLY on training-fold systems. Never call this
    with test-fold data included -- that would be preprocessing leakage.

    The returned dict's keys (and their insertion order) become the single
    source of truth for which columns are active downstream -- normalize_row
    and build_system_graph both derive their column set FROM this dict
    rather than a hardcoded list, so use_extended can never desync between
    functions."""
    train_df = df[df["hostname"].isin(hostnames_train)]
    _, _, all_cols = get_columns(use_extended)
    stats = {}
    for col in all_cols:
        vals = np.log1p(train_df[col].values.astype(float))
        stats[col] = (vals.mean(), vals.std() + 1e-8)
    return stats


def normalize_row(row, stats):
    """Iterates over stats.keys() (NOT a hardcoded column list) so the
    active column set is always whatever compute_normalization_stats() was
    actually built with."""
    out = []
    for col in stats.keys():
        mu, sigma = stats[col]
        v = np.log1p(float(row[col]))
        z = (v - mu) / sigma
        # clip extreme outliers (e.g. Jupiter-mass planets, very wide orbits)
        # to prevent huge input magnitudes from destabilizing training --
        # this is what was causing NaN losses on real archive data
        z = np.clip(z, -5.0, 5.0)
        out.append(z)
    return np.array(out, dtype=np.float32)


def denormalize_period(pred_norm, stats):
    """Inverse of normalize_row's pl_orbper transform: normalized log-space
    prediction -> real period in days. Needed to report metrics DYNAMITE's
    own papers actually use (percentage error, match-within-tolerance),
    which are all in real units, not normalized log-space."""
    mu, sigma = stats["pl_orbper"]
    log_val = pred_norm * sigma + mu
    return np.expm1(log_val)


def build_system_graph(system_df: pd.DataFrame, stats: dict, edge_mode: str = "adjacent",
                        resonance_tolerance: float = 0.05, knn_k: int = 2,
                        threshold_value: float = np.log(2.0), add_star_hub: bool = False,
                        edge_attr_mode: str = "both"):
    """Build a single graph for one system, planets sorted by orbital period.

    edge_mode:
      'adjacent'   -> edges only between period-adjacent planets (sparse chain)
      'complete'   -> fully connected graph
      'resonance'  -> only connect pairs whose period ratio sits near a simple
                      fraction (mean-motion resonance), e.g. 3:2, 2:1. Real,
                      physically-motivated dynamical relationships -- but can
                      leave non-resonant systems very sparse or disconnected.
                      Disconnected components still work fine (conv layers
                      self-loop by default), they just get zero cross-
                      component information, which is the honest, expected
                      behavior for a system with no real resonances.
      'knn'        -> each planet connects to its knn_k nearest neighbors in
                      log-period space (union over both directions, so the
                      graph stays undirected). Middle ground between
                      'adjacent' and 'complete'.
      'threshold'  -> connect any pair within threshold_value of each other
                      in |log period ratio| (default log(2), i.e. within a
                      factor of 2x). Similar spirit to knn, different knob
                      (fixed distance cutoff instead of fixed neighbor count).

    add_star_hub: if True, adds one extra node (index n, after all planets)
      carrying ONLY the star's features (mass/radius/teff; planet-specific
      slots zeroed), connected to every planet. Orthogonal to edge_mode --
      combine with any of the above. Rationale: star features are already
      copied onto every planet's own feature vector, so this isn't adding
      new information per se -- it adds a structural shortcut (any two
      planets are now at most 2 hops apart via the hub), which matters most
      for sparse modes like 'adjacent' or 'resonance' where distant planets
      might otherwise need many hops to exchange information.

    edge_attr_mode: 'both' (default) = [log_period_ratio, log_mass_ratio],
      'period_only' or 'mass_only' = single-column edge_attr, for ablating
      which relational signal actually matters. Star-hub edges always get
      a neutral [0, 0] (or single 0) attr, since a hub isn't a planet and
      has no period/mass ratio to compute.
    """
    sys_sorted = system_df.sort_values("pl_orbper").reset_index(drop=True)
    n = len(sys_sorted)
    periods = sys_sorted["pl_orbper"].values

    x = np.stack([normalize_row(sys_sorted.iloc[i], stats) for i in range(n)])
    x = torch.tensor(x, dtype=torch.float)

    # derived from stats.keys() (NOT hardcoded) -- correct regardless of
    # whether use_extended added more columns after these two
    stats_keys = list(stats.keys())
    PERIOD_IDX = stats_keys.index("pl_orbper")
    MASS_IDX = stats_keys.index("pl_bmasse")

    edges = []
    if edge_mode == "adjacent":
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
    elif edge_mode == "complete":
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i, j])
    elif edge_mode == "resonance":
        # canonical low-order mean-motion resonances (outer:inner ratios)
        canonical_ratios = [1.5, 2.0, 4.0/3.0, 2.5, 5.0/3.0, 3.0, 5.0/2.0]
        for i in range(n):
            for j in range(i + 1, n):
                ratio = periods[j] / periods[i]
                if any(abs(ratio / r - 1.0) < resonance_tolerance for r in canonical_ratios):
                    edges.append([i, j]); edges.append([j, i])
        # NOTE: non-resonant systems may end up with ZERO edges -- this is
        # intentional/honest, not a bug. Isolated nodes still get processed
        # via self-loops inside the conv layers.
    elif edge_mode == "knn":
        log_periods = np.log(periods)
        neighbor_pairs = set()
        for i in range(n):
            dists = np.abs(log_periods - log_periods[i])
            dists[i] = np.inf
            nearest = np.argsort(dists)[:knn_k]
            for j in nearest:
                neighbor_pairs.add((min(i, j), max(i, j)))
        for i, j in neighbor_pairs:
            edges.append([i, j]); edges.append([j, i])
    elif edge_mode == "threshold":
        log_periods = np.log(periods)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(log_periods[j] - log_periods[i]) < threshold_value:
                    edges.append([i, j]); edges.append([j, i])
    else:
        raise ValueError(f"unknown edge_mode {edge_mode}")

    edge_attr_dim = 1 if edge_attr_mode in ("period_only", "mass_only") else 2
    edge_attr = []
    for src, dst in edges:
        d_period = x[dst, PERIOD_IDX].item() - x[src, PERIOD_IDX].item()
        d_mass = x[dst, MASS_IDX].item() - x[src, MASS_IDX].item()
        if edge_attr_mode == "period_only":
            edge_attr.append([d_period])
        elif edge_attr_mode == "mass_only":
            edge_attr.append([d_mass])
        elif edge_attr_mode == "both":
            edge_attr.append([d_period, d_mass])
        else:
            raise ValueError(f"unknown edge_attr_mode {edge_attr_mode}")

    if add_star_hub:
        hub_idx = n
        # hub feature vector: zeros for planet-specific slots, real
        # (already-normalized) star values for the star slots -- reuse any
        # planet row's star columns since they're identical across planets
        # in one system. n_feature computed from stats.keys() rather than
        # hardcoded, so this stays correct whether or not use_extended added
        # more columns (EXTENDED_STAR_COLS also treated as "star" here).
        star_col_names = set(STAR_COLS + EXTENDED_STAR_COLS)
        n_feature = sum(1 for k in stats_keys if k not in star_col_names)
        hub_feat = x[0].clone()
        hub_feat[:n_feature] = 0.0
        x = torch.cat([x, hub_feat.unsqueeze(0)], dim=0)
        for i in range(n):
            edges.append([i, hub_idx]); edges.append([hub_idx, i])
            edge_attr.append([0.0] * edge_attr_dim)
            edge_attr.append([0.0] * edge_attr_dim)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
        else torch.empty((2, 0), dtype=torch.long)
    # NOTE: torch.tensor([]).t() gives the wrong shape ([0], not [2,0]) --
    # this matters for zero-edge cases (single remaining node, or a
    # non-resonant system under edge_mode='resonance')
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, edge_attr_dim))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), sys_sorted


def make_leave_one_out_instances(df: pd.DataFrame, stats: dict, edge_mode: str = "adjacent",
                                  min_remaining: int = 2, **graph_kwargs):
    """For every system, for every planet, build one LOO instance:
      - graph WITHOUT that planet (built on the remaining planets)
      - target = normalized feature vector of the REMOVED planet
      - hostname is carried through for system-level fold splitting later
    Only pl_orbper is used as the primary regression target by default;
    the full normalized vector is kept in case you want to predict more.

    **graph_kwargs: forwarded to build_system_graph (resonance_tolerance,
      knn_k, threshold_value, add_star_hub, edge_attr_mode) -- see that
      function's docstring for what each does.

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
            graph, _ = build_system_graph(remaining, stats, edge_mode=edge_mode, **graph_kwargs)
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
                # original (pre-removal) system table, kept for visualization
                # (visualize_graph.plot_system_graph needs the FULL system,
                # not just the post-removal graph, to show what was masked)
                "system_df": sys_sorted,
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
