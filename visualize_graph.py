"""
visualize_graph.py
Plot a system's graph structure using networkx -- useful both for your
own understanding while debugging the pipeline, and as an actual
methodology-chapter figure showing how a real system gets represented.
"""

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


def _build_full_system_graph(sys_df, edge_mode="complete"):
    """Build the FULL, unmasked graph for one system -- no leave-one-out,
    no removed node, every planet present. This is the 'before we skip any
    planet' view."""
    sys_sorted = sys_df.sort_values("pl_orbper").reset_index(drop=True)
    n = len(sys_sorted)
    periods = sys_sorted["pl_orbper"].values

    G = nx.Graph()
    labels = {}
    for i in range(n):
        row = sys_sorted.iloc[i]
        name = row["pl_name"].split()[-1] if "pl_name" in row else str(i)
        G.add_node(i, mass=row.get("pl_bmasse", 1.0))
        labels[i] = name

    if edge_mode == "adjacent":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif edge_mode == "complete":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"unknown edge_mode {edge_mode}")

    for i, j in pairs:
        log_ratio_dist = abs(np.log(periods[j] / periods[i]))
        G.add_edge(i, j, weight=1.0 / (1.0 + log_ratio_dist),
                   ratio=periods[max(i, j)] / periods[min(i, j)])

    return G, labels


def plot_all_systems_grid(df, min_planets=1, max_systems=16, edge_mode="complete",
                           ncols=4, seed=42, show_edge_labels=True, save_path=None):
    """ALL systems (or up to max_systems) in ONE figure, one subplot per
    system, full unmasked graphs (every planet present -- this is the view
    BEFORE any leave-one-out masking, showing exactly what nodes/edges look
    like for the whole dataset). Renders inline in Colab by default
    (save_path=None); pass a path only if you actually want a file.
    """
    systems = df.groupby("hostname").filter(lambda g: len(g) >= min_planets)
    hostnames = list(systems["hostname"].unique())

    if len(hostnames) > max_systems:
        print(f"NOTE: {len(hostnames)} systems match min_planets={min_planets}, "
              f"showing a random sample of {max_systems} (pass max_systems=N to change this).")
        rng = np.random.RandomState(seed)
        hostnames = list(rng.choice(hostnames, size=max_systems, replace=False))

    n = len(hostnames)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, hostname in zip(axes, hostnames):
        sys_df = df[df["hostname"] == hostname]
        G, labels = _build_full_system_graph(sys_df, edge_mode=edge_mode)
        pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.5 / np.sqrt(max(len(G), 1)))

        masses = np.array([G.nodes[i]["mass"] for i in G.nodes()])
        sizes = 300 + 250 * np.log1p(masses)

        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color="gray", ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color="steelblue", node_size=sizes, ax=ax, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight="bold", ax=ax)
        if show_edge_labels and len(G) <= 5:  # avoid unreadable clutter on dense/large systems
            edge_labels = {(u, v): f"{G.edges[u, v]['ratio']:.1f}x" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

        ax.set_title(hostname, fontsize=9)
        ax.axis("off")

    for ax in axes[len(hostnames):]:
        ax.axis("off")

    fig.suptitle(f"All systems, min_planets={min_planets}, edge_mode={edge_mode} "
                 f"(full graphs, before any leave-one-out masking)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    return fig


def plot_all_systems_3d(df, min_planets=1, max_systems=9, edge_mode="complete",
                         ncols=3, seed=42, save_path=None):
    """3D version of the same idea -- node-link network per system, laid
    out in 3D space via networkx's dim=3 spring layout, all systems in one
    figure. Kept to a smaller default max_systems since 3D subplots get
    visually crowded fast."""
    systems = df.groupby("hostname").filter(lambda g: len(g) >= min_planets)
    hostnames = list(systems["hostname"].unique())

    if len(hostnames) > max_systems:
        print(f"NOTE: {len(hostnames)} systems match min_planets={min_planets}, "
              f"showing a random sample of {max_systems} (pass max_systems=N to change this).")
        rng = np.random.RandomState(seed)
        hostnames = list(rng.choice(hostnames, size=max_systems, replace=False))

    n = len(hostnames)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    for idx, hostname in enumerate(hostnames):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        sys_df = df[df["hostname"] == hostname]
        G, labels = _build_full_system_graph(sys_df, edge_mode=edge_mode)
        pos3d = nx.spring_layout(G, weight="weight", seed=seed, dim=3,
                                 k=1.5 / np.sqrt(max(len(G), 1)))

        xs = [pos3d[i][0] for i in G.nodes()]
        ys = [pos3d[i][1] for i in G.nodes()]
        zs = [pos3d[i][2] for i in G.nodes()]
        masses = np.array([G.nodes[i]["mass"] for i in G.nodes()])
        sizes = 100 + 150 * np.log1p(masses)

        edge_segs = [[pos3d[u], pos3d[v]] for u, v in G.edges()]
        if edge_segs:
            ax.add_collection3d(Line3DCollection(edge_segs, colors="gray", alpha=0.4, linewidths=1.2))

        ax.scatter(xs, ys, zs, s=sizes, c="steelblue", edgecolors="black", depthshade=True)
        for i in G.nodes():
            ax.text(pos3d[i][0], pos3d[i][1], pos3d[i][2], labels[i], fontsize=7, fontweight="bold")

        ax.set_title(hostname, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.suptitle(f"All systems (3D), min_planets={min_planets}, edge_mode={edge_mode} "
                 f"(full graphs, before any leave-one-out masking)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    return fig



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


def _build_combined_graph(df, min_planets=1, edge_mode="complete"):
    """One networkx Graph containing ALL systems as separate connected
    components -- node ids are prefixed with hostname so systems don't
    collide, no edges are added BETWEEN systems (only within each), so
    each system stays visually distinct as its own cluster."""
    systems = df.groupby("hostname").filter(lambda g: len(g) >= min_planets)
    G = nx.Graph()
    labels = {}
    for hostname, sys_df in systems.groupby("hostname"):
        sys_sorted = sys_df.sort_values("pl_orbper").reset_index(drop=True)
        n = len(sys_sorted)
        periods = sys_sorted["pl_orbper"].values
        node_ids = [f"{hostname}::{i}" for i in range(n)]

        for i in range(n):
            row = sys_sorted.iloc[i]
            name = row["pl_name"].split()[-1] if "pl_name" in row else str(i)
            G.add_node(node_ids[i], mass=row.get("pl_bmasse", 1.0), hostname=hostname)
            labels[node_ids[i]] = name

        if edge_mode == "adjacent":
            pairs = [(i, i + 1) for i in range(n - 1)]
        elif edge_mode == "complete":
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            raise ValueError(f"unknown edge_mode {edge_mode}")

        for i, j in pairs:
            log_ratio_dist = abs(np.log(periods[j] / periods[i]))
            G.add_edge(node_ids[i], node_ids[j], weight=1.0 / (1.0 + log_ratio_dist))

    return G, labels


def plot_combined_network(df, min_planets=1, edge_mode="complete", seed=42,
                           figsize=(16, 16), save_path=None):
    """ALL systems on ONE networkx.draw() call -- every system is a
    separate cluster (component) in a single combined graph, not separate
    subplots. This is the full, unmasked structure (before any
    leave-one-out removal)."""
    G, labels = _build_combined_graph(df, min_planets=min_planets, edge_mode=edge_mode)
    n_systems = df.groupby("hostname").filter(lambda g: len(g) >= min_planets)["hostname"].nunique()
    print(f"Combined graph: {n_systems} systems, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if G.number_of_nodes() > 2000:
        print("WARNING: large graph -- this may take a while to lay out and render, and be hard to read.")

    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)))
    masses = np.array([G.nodes[i]["mass"] for i in G.nodes()])
    sizes = 80 + 60 * np.log1p(masses)

    plt.figure(figsize=figsize)
    nx.draw(G, pos, labels=labels, with_labels=True, node_color="steelblue",
            node_size=sizes, edge_color="gray", width=0.6, alpha=0.85,
            font_size=6, font_weight="bold")
    plt.title(f"All systems combined -- {n_systems} systems, min_planets={min_planets}, "
             f"edge_mode={edge_mode}\n(one networkx.draw() call, before any leave-one-out masking)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def plot_combined_network_3d_plotly(df, min_planets=1, edge_mode="complete", seed=42,
                                    save_path=None):
    """Interactive 3D version using Plotly -- rotate/zoom/hover, all
    systems as one combined graph. fig.show() renders inline in Colab."""
    import plotly.graph_objects as go

    G, labels = _build_combined_graph(df, min_planets=min_planets, edge_mode=edge_mode)
    n_systems = df.groupby("hostname").filter(lambda g: len(g) >= min_planets)["hostname"].nunique()
    print(f"Combined graph: {n_systems} systems, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pos = nx.spring_layout(G, weight="weight", seed=seed, dim=3,
                           k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)))

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
        edge_z += [pos[u][2], pos[v][2], None]
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                              line=dict(color="lightgray", width=2), hoverinfo="none")

    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_z = [pos[i][2] for i in G.nodes()]
    hover_text = [f"{G.nodes[i]['hostname']} - {labels[i]}" for i in G.nodes()]
    masses = np.array([G.nodes[i]["mass"] for i in G.nodes()])
    sizes = 4 + 3 * np.log1p(masses)

    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode="markers+text",
                              text=[labels[i] for i in G.nodes()], textposition="top center",
                              textfont=dict(size=8),
                              hovertext=hover_text, hoverinfo="text",
                              marker=dict(size=sizes, color="steelblue", line=dict(color="black", width=0.5)))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"All systems combined (3D, interactive) -- {n_systems} systems, "
              f"min_planets={min_planets}, edge_mode={edge_mode}",
        showlegend=False,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    if save_path:
        # static PNG export needs Chrome/kaleido installed -- not required
        # for interactive use, only if you specifically want a saved image
        fig.write_image(save_path)
    fig.show()
    return fig
