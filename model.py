"""
Spatio-Temporal Graph Neural Network — V6 (Real Features)
==========================================================

With fixed preprocessing, the dataset has:
  - 28 real numeric features (CPU/memory usage with actual variance)
  - 70.5% positive rate (NOT 0.01% — the "extreme imbalance" was an artifact)

This changes the entire loss strategy:
  - No balanced sampling needed (70/30 split is nearly balanced)
  - Focal alpha=0.3 to slightly favor the minority NORMAL class
  - Label smoothing=0.05 to prevent overconfident predictions
  - Loss computed on ALL nodes (no subsampling — full gradient signal)

Architecture: 2-layer GraphSAGE + GRU + attention + MLP, hidden_dim=40.
  - 40 instead of 48 for VRAM safety (28 features × 40 hidden is still
    more capacity than the old 13 × 32).
  - Gradient checkpointing on GRU saves ~15% peak VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge


class SpatialEncoder(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=40, num_classes=2,
                 num_gnn_layers=2, dropout=0.3, edge_drop_rate=0.2,
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
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index, num_nodes=None, return_embeddings=False):
        if isinstance(edge_index, list):
            edge_list = edge_index
        else:
            edge_list = [edge_index] * len(x_seq)

        spatial_out = []
        for t, x_t in enumerate(x_seq):
            ei_t = edge_list[t]
            if self.training and self.edge_drop_rate > 0:
                ei_t, _ = dropout_edge(ei_t, p=self.edge_drop_rate, training=True)
            h_t = self.spatial(x_t, ei_t)
            spatial_out.append(h_t)

        H = torch.stack(spatial_out, dim=1)

        if self.training and H.requires_grad:
            gru_out = grad_checkpoint(self._gru_forward, H, use_reentrant=False)
        else:
            gru_out, _ = self.gru(H)

        attn_w = torch.softmax(self.attn(gru_out), dim=1)
        z = (gru_out * attn_w).sum(dim=1)
        logits = self.classifier(z)

        if return_embeddings:
            return logits, z
        return logits

    def _gru_forward(self, H):
        out, _ = self.gru(H)
        return out


class SmoothedFocalLoss(nn.Module):
    """
    Focal Loss with label smoothing on the FULL batch.

    With 70.5% positive rate, NO balanced subsampling is needed.
    Computing loss on ALL nodes gives maximum gradient signal — every
    node contributes to learning, not just a random subset of 400.

    Label smoothing (ε=0.05): replaces hard targets 0/1 with 0.05/0.95.
    Prevents the model from becoming overconfident on easy examples,
    which improves calibration and generalization.

    alpha=0.3: gives 30% weight to positives (majority), 70% to negatives
    (minority). This corrects the 70/30 class imbalance.
    """

    def __init__(self, alpha=0.3, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = label_smoothing

    def forward(self, logits, targets):
        logits_f = logits.float()

        # Label smoothing: hard targets -> soft targets
        targets_smooth = targets.float()
        if self.eps > 0:
            targets_smooth = targets_smooth * (1 - self.eps) + self.eps * 0.5

        p = torch.sigmoid(logits_f[:, 1])
        pt = torch.where(targets == 1, p, 1 - p)

        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p, self.alpha),
            torch.full_like(p, 1 - self.alpha),
        )

        # BCE with smoothed targets
        bce = -(targets_smooth * torch.log(p.clamp(min=1e-7)) +
                (1 - targets_smooth) * torch.log((1 - p).clamp(min=1e-7)))

        # Focal modulation on the original (non-smoothed) pt
        focal_weight = alpha_t * ((1 - pt) ** self.gamma)
        loss = focal_weight * bce

        return loss.mean()


# Backward compat for evaluate.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits_f = logits.float()
        p = torch.sigmoid(logits_f[:, 1])
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p, self.alpha),
            torch.full_like(p, 1 - self.alpha),
        )
        loss = -alpha_t * ((1 - pt) ** self.gamma) * torch.log(pt.clamp(min=1e-7, max=1.0))
        return loss.mean()


# Keep for import compat if old train.py references it
class BalancedFocalLoss(SmoothedFocalLoss):
    pass
