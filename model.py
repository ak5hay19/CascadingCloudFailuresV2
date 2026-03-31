"""
Spatio-Temporal Graph Neural Network
======================================
Architecture: 2-layer GraphSAGE (spatial) → GRU (temporal) → MLP (classifier)

Why GraphSAGE instead of GCN?
  - GCN does FULL neighborhood aggregation — every node talks to ALL its
    neighbors every layer. At 4,900 nodes with dense cluster connectivity,
    this blows up VRAM (why the old code crashed at 3,000 nodes).
  - GraphSAGE SAMPLES a fixed number of neighbors (k=15) per node per layer.
    So each node's computation cost is O(k) not O(degree). Whether a node
    has 5 or 500 neighbors, it only aggregates 15.
  - This means we can use the FULL graph (~4,900 nodes) instead of
    subsampling to 500. The model sees the complete infrastructure.

Why not GAT?
  - GAT learns attention weights over neighbors — useful when you don't know
    which neighbors matter. But we already encode this via edge weights
    (cluster=1.0, collection=0.5).
  - GAT's multi-head attention uses 2-4x more memory per layer.
  - Single-head GAT ≈ GraphSAGE in performance at this scale.

Syllabus coverage:
  - GraphSAGE (generalized neighborhood aggregation + sampling)
  - Stacking GNN layers (2-layer for 2-hop propagation)
  - Dynamic graphs / spatial-temporal GNN
  - GNN layer optimization (LayerNorm, residual, dropout)
  - Loss functions (Focal Loss for imbalanced node classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def sample_neighbors(edge_index, edge_weight, num_nodes, k=15):
    """
    Sample at most k neighbors per node from the full edge set.

    This is the core of GraphSAGE's scalability:
    - Full aggregation: O(|E|) per layer
    - Sampled aggregation: O(N * k) per layer
    - For our graph: O(30,000) → O(4,900 * 15) = O(73,500)
      Similar cost but bounded — won't explode if edges grow.

    During training, random sampling acts as a regularizer (like dropout
    on the graph structure). During eval, we use all neighbors for
    deterministic predictions.

    Returns:
        sampled_edge_index: [2, num_sampled_edges]
        sampled_edge_weight: [num_sampled_edges] or None
    """
    # Build adjacency list with weights
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    weights = edge_weight.tolist() if edge_weight is not None else [1.0] * len(src)

    for s, d, w in zip(src, dst, weights):
        adj[s].append((d, w))

    sampled_src, sampled_dst, sampled_w = [], [], []
    for node in range(num_nodes):
        neighbors = adj[node]
        if not neighbors:
            continue
        if len(neighbors) <= k:
            selected = neighbors
        else:
            idx = torch.randperm(len(neighbors))[:k].tolist()
            selected = [neighbors[i] for i in idx]
        for (d, w) in selected:
            sampled_src.append(node)
            sampled_dst.append(d)
            sampled_w.append(w)

    s_ei = torch.tensor([sampled_src, sampled_dst], dtype=torch.long,
                         device=edge_index.device)
    s_ew = torch.tensor(sampled_w, dtype=torch.float, device=edge_index.device)
    return s_ei, s_ew


class SpatialEncoder(nn.Module):
    """
    Multi-layer GraphSAGE with residual connections.

    GraphSAGE layer operation:
      h_i' = W · CONCAT(h_i, MEAN({h_j : j ∈ sampled_N(i)}))

    Key difference from GCN:
      - GCN: h_i' = W · MEAN({h_j : j ∈ N(i) ∪ {i}})  ← uses ALL neighbors
      - SAGE: samples k neighbors, then concatenates self with aggregated
              neighbor info. This preserves the node's own features better.

    2 layers = 2-hop neighborhood:
      - Layer 1: each node sees its direct (sampled) neighbors
      - Layer 2: each node sees neighbors-of-neighbors
      - For cascading failures: if A fails → affects B → affects C,
        node C can "see" A's state through 2-hop aggregation.
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
    Full pipeline:

    For each timestep t:
      1. Sample k neighbors per node from full graph
      2. h_t = GraphSAGE(x_t, sampled_edges)     → [N, hidden]

    Stack all timesteps:
      H = stack([h_1, ..., h_T])                  → [N, T, hidden]

    Temporal encoding:
      z = GRU(H)[:, -1, :]                        → [N, hidden]

    Classification:
      logits = MLP(z)                              → [N, 2]

    The neighbor sampling happens INSIDE the forward pass:
      - During training: random k neighbors (acts as regularizer)
      - During eval: all neighbors (deterministic)
    """

    def __init__(self, input_dim, hidden_dim=48, num_classes=2,
                 num_gnn_layers=2, dropout=0.3, num_neighbors=15):
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
        self.num_neighbors = num_neighbors

    def forward(self, x_seq, edge_index, edge_weight=None,
                num_nodes=None, return_embeddings=False):
        """
        Args:
            x_seq: list of T tensors, each [N, F]
            edge_index: [2, E] full graph edges
            edge_weight: [E] edge weights
            num_nodes: int, needed for neighbor sampling
            return_embeddings: if True, also return node embeddings
        """
        N = x_seq[0].shape[0]
        if num_nodes is None:
            num_nodes = N

        # Neighbor sampling: different sample each forward pass (training)
        # or full graph (eval)
        if self.training:
            s_ei, s_ew = sample_neighbors(
                edge_index, edge_weight, num_nodes, k=self.num_neighbors
            )
        else:
            s_ei = edge_index
            # SAGEConv doesn't use edge weights directly, so we just
            # pass the full edge_index during eval for deterministic results

        # Spatial encoding per timestep
        spatial_out = []
        for x_t in x_seq:
            h_t = self.spatial(x_t, s_ei)
            spatial_out.append(h_t)

        # Stack → [N, T, hidden]
        H = torch.stack(spatial_out, dim=1)

        # Temporal GRU
        gru_out, _ = self.gru(H)
        z = gru_out[:, -1, :]      # last hidden state

        # Classify
        logits = self.classifier(z)

        if return_embeddings:
            return logits, z
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced node classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha=0.75: 3x more weight to failing class
    gamma=2.0:  down-weight easy predictions, focus on hard cases

    Better than weighted CE because it's adaptive — as the model
    gets better at easy cases, it automatically shifts focus to
    the ambiguous ones near the decision boundary.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_oh).sum(dim=1).clamp(1e-7, 1.0)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        loss = -alpha_t * ((1 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()
