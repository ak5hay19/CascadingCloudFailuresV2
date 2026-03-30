"""
Spatio-Temporal Graph Neural Network
======================================
Architecture: 2-layer GCN (spatial) → GRU (temporal) → MLP (classifier)

Syllabus concepts used:
  - GCN (Graph Convolution Network) for spatial message passing
  - Stacking GNN layers for multi-hop neighborhood aggregation
  - Dynamic graphs / spatial-temporal GNN
  - GNN layer optimization (LayerNorm, residual connections)
  - Generalized neighborhood aggregation
  - Focal Loss for imbalanced node classification

Design for RTX 4060 8GB:
  - hidden_dim=48 (not 64/128) — fits comfortably in VRAM
  - 2 GCN layers with residual connection — captures 2-hop failure propagation
  - Standard GRU (not GraphConvGRU) — 6x fewer GCN calls
  - No attention heads (GAT) — GCN is 3x faster, sufficient for our graph size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpatialEncoder(nn.Module):
    """
    Multi-layer GCN with residual connections.

    Why 2 layers?
    - 1 layer: each node sees only direct neighbors (1-hop)
    - 2 layers: each node sees neighbors-of-neighbors (2-hop)
    - This matters for cascading failures: if machine A fails and
      affects B, and B affects C, we need C to "see" A's state.
    - 3+ layers causes over-smoothing (all nodes look the same).

    Why residual connection?
    - Prevents the GCN from "washing out" a node's own features
      by over-aggregating neighbor information.
    - The node's own CPU/memory stats remain visible to the classifier.
    """

    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: input_dim → hidden_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Additional layers: hidden_dim → hidden_dim (with residual)
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x if (i > 0 and x.shape[-1] == conv.out_channels) else None

            x = conv(x, edge_index, edge_weight=edge_weight)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if residual is not None:
                x = x + residual  # Residual connection

        return x


class SpatioTemporalGNN(nn.Module):
    """
    Full pipeline:

    Input: sequence of T graph snapshots, each with N nodes and F features
           + edge_index defining the graph structure

    For each timestep t:
      h_t = GCN(x_t, edges)              → [N, hidden]   (spatial encoding)

    Stack all timesteps:
      H = stack([h_1, ..., h_T])          → [N, T, hidden]

    Temporal encoding (per-node GRU):
      z = GRU(H)[:, -1, :]               → [N, hidden]   (last hidden state)

    Classification:
      logits = MLP(z)                     → [N, 2]        (normal vs failing)

    Why this "decouple" approach (GCN first, then GRU)?
    - Alternative: Temporal GNN (DCRNN, A3TGCN) runs GCN *inside* each
      GRU step, so for T=6 timesteps and 2 GCN layers, that's 12 GCN calls.
    - Our approach: 6 GCN calls + 1 GRU pass = much faster.
    - Trade-off: we lose "spatial-temporal coupling" (graph structure
      can't change per timestep), but our graph IS static across
      the sequence (same machines, same edges), so no loss.
    """

    def __init__(self, input_dim, hidden_dim=48, num_classes=2,
                 num_gnn_layers=2, dropout=0.3):
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

    def forward(self, x_seq, edge_index, edge_weight=None, return_embeddings=False):
        """
        Args:
            x_seq: list of T tensors, each [N, F]
            edge_index: [2, E] — graph connectivity
            edge_weight: [E] — optional edge weights
            return_embeddings: if True, also return node embeddings

        Returns:
            logits: [N, 2]
            embeddings (optional): [N, hidden]
        """
        # === SPATIAL: encode each timestep ===
        spatial_out = []
        for x_t in x_seq:
            h_t = self.spatial(x_t, edge_index, edge_weight)
            spatial_out.append(h_t)

        # Stack → [N, T, hidden]
        H = torch.stack(spatial_out, dim=1)

        # === TEMPORAL: GRU over time ===
        gru_out, _ = self.gru(H)          # [N, T, hidden]
        z = gru_out[:, -1, :]             # [N, hidden] — last timestep

        # === CLASSIFY ===
        logits = self.classifier(z)        # [N, 2]

        if return_embeddings:
            return logits, z
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Standard cross-entropy treats all samples equally. With ~78% normal
    vs ~22% failing nodes, the model can get 78% accuracy by always
    predicting "normal." Focal Loss fixes this:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - alpha (0.75): gives 3x more weight to failing class
    - gamma (2.0): reduces loss for "easy" samples (high confidence correct
      predictions), forcing the model to focus on ambiguous cases

    This is better than simple class weighting because it's adaptive:
    as the model gets better at easy cases, it automatically shifts
    attention to the hard ones.
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
