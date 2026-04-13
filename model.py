"""
Spatio-Temporal Graph Neural Network (Dynamic)
================================================
Architecture: 2-layer GraphSAGE (spatial) → GRU (temporal) → MLP (classifier)

KEY NOVELTY: Dynamic graph topology.
  - Static ST-GNNs use the same edge_index for every timestep.
  - Our model accepts a DIFFERENT edge_index per timestep, reflecting
    which machines are actually co-located in the same cluster/collection
    during that specific 5-minute window.
  - This captures real topology evolution: machines join/leave collections
    as jobs start/finish, cluster membership changes with reassignment.

Why GraphSAGE?
  - Neighbor sampling (via edge dropout) keeps VRAM bounded
  - Full graph (~4,900 nodes) without subsampling
  - Edge dropout during training = graph-structure regularization

FIX LOG (memory & performance):
  - Replaced pure-Python sample_neighbors() with torch_geometric.utils.dropout_edge
    → eliminates CPU↔GPU round-trips, Python adjacency lists, and per-call allocations
  - Added edge_drop_rate parameter (default 0.3) to control graph regularization
  - Removed num_neighbors parameter (no longer needed with edge dropout)
  - dropout_edge runs in PyG's C++ backend on GPU — zero Python overhead
  - Added temporal attention pooling over all GRU outputs (replaces last-step-only)
    → model attends to the most predictive window; early cascade signals no longer lost
  - Switched FocalLoss to sigmoid-based (Lin et al. 2017): correct for binary node
    classification; softmax coupled the two class probabilities, which is incorrect
    because a node's failure probability is independent of its normal probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge


class SpatialEncoder(nn.Module):
    """
    2-layer GraphSAGE with residual connections.

    Layer 1: node sees its direct neighbors — 1-hop
    Layer 2: node sees neighbors-of-neighbors — 2-hop
    Residual: preserves the node's own features through aggregation
    """

    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x if (i > 0 and x.shape[-1] == conv.out_channels) else None
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual is not None:
                x = x + residual
        return x


class SpatioTemporalGNN(nn.Module):
    """
    Dynamic Spatio-Temporal GNN.

    For each timestep t:
      1. Get edge_index_t (dynamic — different graph structure per window)
      2. If training: randomly drop edges via dropout_edge (graph regularizer)
      3. h_t = GraphSAGE(x_t, edges_t)              → [N, hidden]

    Stack:  H = [h_1, ..., h_T]                      → [N, T, hidden]
    GRU:    z = GRU(H)[:, -1, :]                      → [N, hidden]
    MLP:    logits = classifier(z)                     → [N, 2]

    The key difference from static ST-GNNs: edge_index can be a LIST
    of per-timestep edge tensors, not just one shared tensor.
    If a single tensor is passed, it's reused for all timesteps (backward compatible).
    """

    def __init__(self, input_dim, hidden_dim=48, num_classes=2,
                 num_gnn_layers=2, dropout=0.3, edge_drop_rate=0.3,
                 # Keep num_neighbors for backward compat with saved checkpoints
                 num_neighbors=None):
        super().__init__()
        self.spatial = SpatialEncoder(input_dim, hidden_dim, num_gnn_layers, dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.hidden_dim = hidden_dim
        self.edge_drop_rate = edge_drop_rate
        # Temporal attention: learns which timestep carries the most failure signal.
        # Replaces the naive "always use the last GRU output" assumption — a machine
        # degrading sharply at t-3 is a stronger cascade indicator than its stable
        # readings at t-1, and attention lets the model discover that automatically.
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index, num_nodes=None, return_embeddings=False):
        """
        Args:
            x_seq: list of T tensors, each [N, F]
            edge_index: EITHER a single [2, E] tensor (static graph)
                        OR a list of T tensors (dynamic graph — one per timestep)
            num_nodes: int (unused, kept for API compatibility)
            return_embeddings: if True, also return node embeddings
        """
        # Handle both static and dynamic edge_index
        if isinstance(edge_index, list):
            edge_list = edge_index  # one per timestep
        else:
            edge_list = [edge_index] * len(x_seq)  # reuse static for all

        # Spatial encoding per timestep with per-timestep edges
        spatial_out = []
        for t, x_t in enumerate(x_seq):
            ei_t = edge_list[t]

            # Edge dropout during training: PyG's C++ backend, no Python loops
            # This replaces the old sample_neighbors() function entirely.
            # dropout_edge randomly removes edges, achieving the same
            # regularization as neighbor sampling but ~100x faster.
            if self.training and self.edge_drop_rate > 0:
                ei_t, _ = dropout_edge(ei_t, p=self.edge_drop_rate,
                                       training=True)

            h_t = self.spatial(x_t, ei_t)
            spatial_out.append(h_t)

        # Stack → [N, T, hidden]
        H = torch.stack(spatial_out, dim=1)

        # Temporal GRU + attention pooling over all timesteps.
        # A single last-step readout discards intermediate windows — e.g., a resource
        # spike at t-4 that precedes a cascade is as important as the final window.
        # Attention weights (softmax over T) let the model learn which window to focus
        # on, keeping the full sequence history while remaining compute-efficient.
        gru_out, _ = self.gru(H)                              # [N, T, hidden]
        attn_w = torch.softmax(self.attn(gru_out), dim=1)     # [N, T, 1]
        z = (gru_out * attn_w).sum(dim=1)                     # [N, hidden]

        # Classify
        logits = self.classifier(z)

        if return_embeddings:
            return logits, z
        return logits


class FocalLoss(nn.Module):
    """
    Sigmoid-based Focal Loss for imbalanced binary node classification (Lin et al. 2017).

    alpha: weight for the failing class — computed dynamically from training class
           distribution so it reflects actual failure rarity in the Borg traces
    gamma=2.0: down-weights easy predictions, forcing the model to focus on the hard
               cases (nodes on the boundary between normal and cascading failure)

    Why sigmoid, not softmax?
      Softmax treats the two classes as competitors (p_fail + p_normal = 1), which
      couples the predictions. For node-level failure detection, whether a server is
      failing is independent of how "normal" it looks — sigmoid respects this.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # p: predicted probability of being in the failing class
        p = torch.sigmoid(logits[:, 1])
        # pt: probability assigned to the TRUE class for each node
        pt = torch.where(targets == 1, p, 1 - p)
        # alpha_t: class-specific focal weight
        alpha_t = torch.where(targets == 1,
                              torch.full_like(p, self.alpha),
                              torch.full_like(p, 1 - self.alpha))
        loss = -alpha_t * ((1 - pt) ** self.gamma) * torch.log(pt.clamp(1e-7, 1.0))
        return loss.mean()
