"""
visualize_graph.py
Plot a system's graph structure using networkx -- useful both for your
own understanding while debugging the pipeline, and as an actual
methodology-chapter figure showing how a real system gets represented.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_network_graph(system_df, edge_mode="complete", removed_index=None,
                        title=None, save_path=None, seed=42):
    """A genuine node-link NETWORK diagram (force-directed spring layout),
    not the linear/sequence view plot_system_graph() gives you. This is
    the actual SNA-style visualization: edge WEIGHT (based on how close two
    planets' periods/masses are) pulls related nodes together and pushes
    unrelated ones apart, so the layout itself reflects graph structure --
    same convention as a social network diagram, not a timeline.

    edge_mode:
      'adjacent' -> same sparse chain the GNN trains on by default (each
        planet connected only to its period-neighbors)
      'complete' -> fully connected graph (every planet pairs with every
        other) -- this is the one that actually LOOKS like a network, with
        crossing edges, and is what edge_mode='complete' in the training
        pipeline corresponds to
    """
    sys_sorted = system_df.sort_values("pl_orbper").reset_index(drop=True)
    n = len(sys_sorted)

    G = nx.Graph()
    labels = {}
    for i in range(n):
        row = sys_sorted.iloc[i]
        name = row["pl_name"].split()[-1] if "pl_name" in row else str(i)
        G.add_node(i, period=row["pl_orbper"], mass=row.get("pl_bmasse", 1.0))
        labels[i] = name

    periods = sys_sorted["pl_orbper"].values
    pairs = []
    if edge_mode == "adjacent":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif edge_mode == "complete":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"unknown edge_mode {edge_mode}")

    for i, j in pairs:
        # edge weight = inverse log-period-ratio distance -- planets closer
        # in period get a STRONGER edge, exactly like edge_attr in the real
        # training graphs, and this is what drives the spring layout below
        log_ratio_dist = abs(np.log(periods[j] / periods[i]))
        weight = 1.0 / (1.0 + log_ratio_dist)
        G.add_edge(i, j, weight=weight)

    # SNA-flavor summary stats -- degree centrality, printed for reference
    centrality = nx.degree_centrality(G)
    print("Degree centrality (structural importance in this system's graph):")
    for i in range(n):
        print(f"  {labels[i]}: {centrality[i]:.3f}")

    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.5 / np.sqrt(n))

    fig, ax = plt.subplots(figsize=(7, 6))
    node_colors = ["crimson" if removed_index is not None and i == removed_index
                   else "steelblue" for i in range(n)]
    masses = np.array([G.nodes[i]["mass"] for i in range(n)])
    sizes = 600 + 400 * np.log1p(masses)

    weights = np.array([G.edges[e]["weight"] for e in G.edges()])
    widths = 1 + 5 * (weights / weights.max() if len(weights) else 1)

    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

    ax.set_title(title or f"{edge_mode} graph -- network layout (edge strength = period similarity)")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return ax, G


def plot_network_from_pipeline(instances, n_examples=3, edge_mode="complete", model=None,
                                device="cpu", save_prefix="network_pipeline_example", seed=0):
    """Same idea as plot_instances_from_pipeline(), but renders the
    force-directed NETWORK diagram (plot_network_graph) instead of the
    linear sequence view -- pulls real system_df/removed_index straight
    from your fold data, no hand-typed examples.

    edge_mode='complete' (default) shows the fully-connected graph, which
    is the one that actually reads as a "network" (crossing edges); use
    'adjacent' to see the sparser chain structure the default training
    pipeline uses instead.
    """
    import torch as _torch
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(instances), size=min(n_examples, len(instances)), replace=False)

    saved_paths = []
    for k, idx in enumerate(chosen):
        inst = instances[idx]
        if "system_df" not in inst:
            raise KeyError(
                "instance is missing 'system_df' -- re-run make_leave_one_out_instances() "
                "from the current data_pipeline.py (older cached instances won't have this key)"
            )

        title = f"{inst['hostname']} -- {edge_mode} graph (removed: index {inst['removed_index']} of {inst['n_planets']})"
        if model is not None:
            model.eval()
            with _torch.no_grad():
                g = inst["graph"]
                batch = _torch.zeros(g.x.size(0), dtype=_torch.long)
                pred = model(g.x, g.edge_index, batch, edge_attr=g.edge_attr, aux_feat=g.aux_feat)
            true_val = inst["target_period_only"].item()
            title += f"\npredicted (norm. log-period)={pred.item():.3f}  true={true_val:.3f}"

        save_path = f"{save_prefix}_{k}.png"
        plot_network_graph(inst["system_df"], edge_mode=edge_mode,
                            removed_index=inst["removed_index"],
                            title=title, save_path=save_path)
        saved_paths.append(save_path)
        plt.close()

    print(f"Saved {len(saved_paths)} real network-diagram examples: {saved_paths}")
    return saved_paths


def plot_instances_from_pipeline(instances, n_examples=3, model=None, device="cpu",
                                  save_prefix="pipeline_example", seed=0):
    """Plot REAL leave-one-out instances straight from your fold data
    (instances_for_hosts() / run_nested_cv() output) -- not a hand-typed
    example. Requires instances to include 'system_df' (added to
    make_leave_one_out_instances() output).

    If model is given, also runs a prediction and shows predicted vs true
    (denormalized back to real units is NOT done here -- these are still
    in normalized log-space, matching what the model actually sees/predicts;
    say so in any figure caption).
    """
    import torch as _torch
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(instances), size=min(n_examples, len(instances)), replace=False)

    saved_paths = []
    for k, idx in enumerate(chosen):
        inst = instances[idx]
        if "system_df" not in inst:
            raise KeyError(
                "instance is missing 'system_df' -- re-run make_leave_one_out_instances() "
                "from the current data_pipeline.py (older cached instances won't have this key)"
            )

        title = f"{inst['hostname']} (removed: index {inst['removed_index']} of {inst['n_planets']})"
        if model is not None:
            model.eval()
            with _torch.no_grad():
                g = inst["graph"]
                batch = _torch.zeros(g.x.size(0), dtype=_torch.long)
                pred = model(g.x, g.edge_index, batch, edge_attr=g.edge_attr, aux_feat=g.aux_feat)
            true_val = inst["target_period_only"].item()
            title += f"\npredicted (norm. log-period)={pred.item():.3f}  true={true_val:.3f}"

        save_path = f"{save_prefix}_{k}.png"
        plot_system_graph(inst["system_df"], removed_index=inst["removed_index"],
                           title=title, save_path=save_path)
        saved_paths.append(save_path)
        plt.close()

    print(f"Saved {len(saved_paths)} real pipeline examples: {saved_paths}")
    return saved_paths


def plot_system_graph(system_df, removed_index=None, title=None, save_path=None,
                       ax=None):
    """Plot one system's planets as a graph, sorted by orbital period.

    system_df: a pandas DataFrame for ONE system (all planets, pre-removal),
               must have pl_name, pl_orbper, pl_bmasse, pl_rade columns.
    removed_index: if set, highlight this planet in red (the "missing" one
                   in a leave-one-out instance) and gray out its edges to
                   show what the model actually saw.
    """
    sys_sorted = system_df.sort_values("pl_orbper").reset_index(drop=True)
    n = len(sys_sorted)

    G = nx.Graph()
    labels = {}
    for i in range(n):
        row = sys_sorted.iloc[i]
        name = row["pl_name"].split()[-1] if "pl_name" in row else str(i)
        G.add_node(i, period=row["pl_orbper"], mass=row.get("pl_bmasse", 1.0))
        labels[i] = f"{name}\nP={row['pl_orbper']:.1f}d"

    for i in range(n - 1):
        ratio = sys_sorted.iloc[i + 1]["pl_orbper"] / sys_sorted.iloc[i]["pl_orbper"]
        G.add_edge(i, i + 1, ratio=ratio)

    # linear layout, ordered by orbital period (innermost to outermost)
    pos = {i: (i, 0) for i in range(n)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.2 * n, 3))

    node_colors = []
    for i in range(n):
        if removed_index is not None and i == removed_index:
            node_colors.append("crimson")
        else:
            node_colors.append("steelblue")

    # node size scaled by (log) mass -- purely visual, not used by the model
    masses = np.array([G.nodes[i]["mass"] for i in range(n)])
    sizes = 800 + 400 * np.log1p(masses)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if removed_index is not None and (u == removed_index or v == removed_index):
            edge_colors.append("lightgray")
            edge_widths.append(1.0)
        else:
            edge_colors.append("black")
            edge_widths.append(2.0)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)

    edge_labels = {(u, v): f"{G.edges[u, v]['ratio']:.1f}x" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.set_title(title or f"System graph ({n} planets, sorted by period)")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return ax


def plot_gat_attention(system_df, edge_index, attn_weights, title=None, save_path=None):
    """Visualize GAT attention weights on a system graph (Phase 5
    interpretability). attn_weights should be the per-edge alpha values
    from GATModel.forward(..., return_attention=True) for ONE layer,
    averaged across attention heads.

    edge_index: [2, num_edges] tensor/array of node index pairs
    attn_weights: [num_edges] array of attention scores, same order as edge_index
    """
    sys_sorted = system_df.sort_values("pl_orbper").reset_index(drop=True)
    n = len(sys_sorted)

    G = nx.DiGraph()
    labels = {}
    for i in range(n):
        row = sys_sorted.iloc[i]
        name = row["pl_name"].split()[-1] if "pl_name" in row else str(i)
        G.add_node(i)
        labels[i] = f"{name}\nP={row['pl_orbper']:.1f}d"

    edge_index = np.asarray(edge_index)
    attn_weights = np.asarray(attn_weights).flatten()
    for k in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, k]), int(edge_index[1, k])
        G.add_edge(src, dst, weight=float(attn_weights[k]))

    pos = {i: (i, 0) for i in range(n)}
    fig, ax = plt.subplots(figsize=(2.5 * n, 3))
    nx.draw_networkx_nodes(G, pos, node_color="steelblue", node_size=800, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    weights = [G.edges[u, v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    for (u, v), w in zip(G.edges(), weights):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            width=1 + 6 * (w / max_w), edge_color="darkred", alpha=0.4 + 0.6 * (w / max_w),
            connectionstyle="arc3,rad=0.15", arrows=True,
        )

    ax.set_title(title or "GAT attention weights (thicker/darker = more attended)")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return ax
