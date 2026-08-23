"""
visualize_graph.py
Plot a system's graph structure using networkx -- useful both for your
own understanding while debugging the pipeline, and as an actual
methodology-chapter figure showing how a real system gets represented.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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
