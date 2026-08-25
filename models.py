"""
models.py
GNN architectures for the missing-planet prediction task.

Task formulation: given a graph of the REMAINING planets in a system,
predict the target feature(s) of the REMOVED planet via a graph-level
readout (global pooling) followed by a small MLP head.

edge_attr support: graphs now carry [log_period_ratio, log_mass_ratio] per
edge (see data_pipeline.build_system_graph). This gives the GNN the same
relational signal DYNAMITE's population statistics use explicitly, instead
of forcing it to infer relationships from unweighted topology alone --
added after real-data runs showed the GNN underperforming the
period-ratio-based baselines by a wide margin.

Each architecture handles edge features differently:
  - GCN: GCNConv only accepts a single scalar edge_weight, so we use the
    period-ratio component of edge_attr (the physically dominant signal).
  - GAT: GATConv natively supports multi-dim edge_attr via edge_dim.
  - GraphSAGE: SAGEConv has no built-in edge feature support -- left as
    topology-only, which makes it a useful ablation point (does edge
    weighting matter, or does GraphSAGE's neighbor-sampling design not need it?).
  - GIN: switched from GINConv to GINEConv, which supports edge_attr.
  - Deep Sets: ignores edges entirely by design (the whole point of that baseline).

All models share the same interface (forward(x, edge_index, batch, edge_attr))
so they can be swapped in the training loop for the architecture ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINEConv, global_mean_pool, global_max_pool
)


class BaseGNN(nn.Module):
    """Shared scaffolding: N conv layers -> pooling -> MLP head.
    Subclasses define which conv layer type to use and how (or whether)
    they consume edge_attr.
    """

    def __init__(self, in_dim, hidden_dim=16, out_dim=1, n_layers=2, dropout=0.2,
                 pooling="mean", edge_dim=2, aux_dim=2):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.pooling = pooling
        self.edge_dim = edge_dim
        self.aux_dim = aux_dim
        self.convs = nn.ModuleList()
        self._build_convs(in_dim, hidden_dim, n_layers)
        # aux_dim=2 by default: [gap_pos, dynamite_pred].
        #   gap_pos: WHERE in the sequence the missing planet sits (0=innermost
        #     gap, 1=outermost) -- pooling alone discards this; baselines get
        #     it via removed_index.
        #   dynamite_pred: the DYNAMITE-style baseline's OWN prediction for
        #     this instance, fit on the same training fold. Giving the GNN
        #     this as an input lets it learn a CORRECTION on top of the
        #     statistical baseline (using the extra node features DYNAMITE
        #     ignores -- mass, eccentricity, star properties) instead of
        #     having to relearn the population-level period-ratio pattern
        #     from scratch on ~230 systems. If the GNN finds no useful
        #     correction, it can simply learn to pass dynamite_pred through
        #     unchanged -- so this is a strict superset of DYNAMITE-style,
        #     not a replacement, and should be at least as good as it archi
        #     -tecturally, unlike the pure end-to-end version.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _build_convs(self, in_dim, hidden_dim, n_layers):
        raise NotImplementedError

    def _conv_forward(self, conv, x, edge_index, edge_attr):
        """Override in subclasses that consume edge_attr differently."""
        return conv(x, edge_index)

    def forward(self, x, edge_index, batch, edge_attr=None, aux_feat=None):
        for conv in self.convs:
            x = self._conv_forward(conv, x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == "mean":
            g = global_mean_pool(x, batch)
        elif self.pooling == "max":
            g = global_max_pool(x, batch)
        else:
            raise ValueError(f"unknown pooling {self.pooling}")
        if aux_feat is None:
            aux_feat = torch.zeros((g.size(0), self.aux_dim), device=g.device)
        g = torch.cat([g, aux_feat], dim=-1)
        return self.head(g).squeeze(-1)


class GCNModel(BaseGNN):
    """Uses edge_attr[:, 0] (log period ratio) as a scalar edge_weight --
    GCNConv only supports single-dimensional edge weighting."""

    def _build_convs(self, in_dim, hidden_dim, n_layers):
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

    def _conv_forward(self, conv, x, edge_index, edge_attr):
        # GCNConv's internal normalization (D^-1/2 A D^-1/2) assumes
        # non-negative edge weights -- our period-ratio edge_attr is SIGNED
        # (opposite sign for the two directions of each bidirectional edge),
        # which broke GCN's normalization and caused it to diverge to NaN.
        # Use the magnitude of the spacing instead; direction is still
        # implicitly available to the model via which node is src vs dst.
        edge_weight = edge_attr[:, 0].abs().clamp(min=1e-3) if edge_attr is not None and edge_attr.numel() > 0 else None
        return conv(x, edge_index, edge_weight=edge_weight)


class GATModel(BaseGNN):
    """GAT natively supports full multi-dim edge_attr and also exposes
    attention weights for interpretability (Phase 5)."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=1, n_layers=2, dropout=0.2,
                 pooling="mean", heads=4, edge_dim=2, aux_dim=2):
        self.heads = heads
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, dropout, pooling, edge_dim, aux_dim)

    def _build_convs(self, in_dim, hidden_dim, n_layers):
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            concat = i < n_layers - 1
            out_channels = hidden_dim // self.heads if concat else hidden_dim
            self.convs.append(GATConv(dims[i], out_channels, heads=self.heads, concat=concat,
                                       dropout=self.dropout, edge_dim=self.edge_dim))

    def forward(self, x, edge_index, batch, edge_attr=None, aux_feat=None, return_attention=False):
        attn_weights = []
        for conv in self.convs:
            ea = edge_attr if (edge_attr is not None and edge_attr.numel() > 0) else None
            if return_attention:
                x, (edge_idx, alpha) = conv(x, edge_index, edge_attr=ea, return_attention_weights=True)
                attn_weights.append((edge_idx, alpha))
            else:
                x = conv(x, edge_index, edge_attr=ea)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch) if self.pooling == "mean" else global_max_pool(x, batch)
        if aux_feat is None:
            aux_feat = torch.zeros((g.size(0), self.aux_dim), device=g.device)
        g = torch.cat([g, aux_feat], dim=-1)
        out = self.head(g).squeeze(-1)
        if return_attention:
            return out, attn_weights
        return out


class GraphSAGEModel(BaseGNN):
    """SAGEConv has no native edge-feature support -- topology-only by
    construction. Kept as a useful ablation: does edge weighting actually
    help, or does neighbor aggregation alone capture enough signal?"""

    def _build_convs(self, in_dim, hidden_dim, n_layers):
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))


class GINModel(BaseGNN):
    """GINEConv (edge-feature-aware GIN variant) instead of plain GINConv,
    so this architecture also benefits from period/mass-ratio edge_attr."""

    def _build_convs(self, in_dim, hidden_dim, n_layers):
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            mlp = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Linear(dims[i + 1], dims[i + 1]))
            self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim))

    def _conv_forward(self, conv, x, edge_index, edge_attr):
        ea = edge_attr if (edge_attr is not None and edge_attr.numel() > 0) else \
            torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
        return conv(x, edge_index, edge_attr=ea)


class DeepSetsModel(nn.Module):
    """Non-graph baseline: permutation-invariant, no edges at all.
    Tests whether relational (edge) structure adds value over the GNNs."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=1, dropout=0.2, **kwargs):
        super().__init__()
        self.aux_dim = kwargs.get("aux_dim", 2)
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim + self.aux_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch, edge_attr=None, aux_feat=None):
        # edge_index/edge_attr ignored on purpose -- this is the whole point of the baseline
        h = self.phi(x)
        g = global_mean_pool(h, batch)
        if aux_feat is None:
            aux_feat = torch.zeros((g.size(0), self.aux_dim), device=g.device)
        g = torch.cat([g, aux_feat], dim=-1)
        return self.rho(g).squeeze(-1)


MODEL_REGISTRY = {
    "gcn": GCNModel,
    "gat": GATModel,
    "sage": GraphSAGEModel,
    "gin": GINModel,
    "deepsets": DeepSetsModel,
}


def build_model(name, in_dim, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name}, choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](in_dim, **kwargs)
