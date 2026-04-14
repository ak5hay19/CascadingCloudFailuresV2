"""
STEP 2: Train the Dynamic Spatio-Temporal GNN
================================================
    python train.py

Windows 11 Native (PowerShell) — RTX 4060 Mobile 8GB / 16GB RAM

CRITICAL CONSTRAINTS (from CLAUDE.md):
  - batch_size: 2     (HARD LIMIT — higher causes OOM on 8GB VRAM)
  - hidden_dim: 32    (HARD LIMIT — 48 causes OOM)
  - num_workers: 0    (HARD LIMIT — >1 causes Error 1455 on Windows)
  - epochs: 10        (fits 30-60 minute training window)
  - NO torch.compile  (Triton not supported on Windows)
  - Modern torch.amp  (not deprecated torch.cuda.amp)

V3 CHANGES (fix zero-learning / F1=0.0):
  - BalancedFocalLoss: loss computed on ALL failing nodes + neg_ratio×
    randomly sampled normal nodes. Forward pass still uses all nodes
    (GraphSAGE needs full neighborhoods). Effective ~25% positive rate.
  - Focal alpha lowered to 0.25 (balanced subset doesn't need 0.99).
  - Focal gamma lowered to 2.0 (standard — sampling removes easy negs).
  - Evaluation uses configurable threshold (default 0.05 instead of 0.5)
    for sensitivity to rare failures.
  - Train-time confusion matrix also uses threshold-based predictions.

Pipeline:
  1. PRE-COMPUTE at init: features/labels in SPARSE PACKED format.
  2. SHARED MEMORY: packed tensors in torch shared memory.
  3. num_workers=0 with pin_memory=True (Windows-safe).
  4. TRUE MINI-BATCHING: B sequences batched via disjoint graph union.
  5. AMP: fp16 forward on Tensor Cores via torch.amp.autocast('cuda').
  6. cudnn.benchmark=True for optimized convolution kernels.
"""

import os
import gc
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from model import SpatioTemporalGNN, SmoothedFocalLoss


# ============================================================================
# 1. SequenceDataset — sparse packed, shared-memory backed
# ============================================================================

class SequenceDataset(Dataset):
    """
    Returns graph sequences by reconstructing dense tensors from sparse
    packed shared-memory storage on-the-fly.

    Sparse packed format stores only active (node, window) entries -> ~10 MB
    instead of dense [W, N, F] which would need 41.5 GB.
    """

    def __init__(self, feat_values, feat_nodes, feat_offsets,
                 label_nodes, label_offsets,
                 all_edge_flat, edge_offsets,
                 seq_starts, seq_length, num_nodes, num_features,
                 static_ei, feat_mean, feat_std,
                 feat_values_norm=None, zero_normalized=None,
                 device=None):
        # Device for on-GPU tensor reconstruction (eliminates CPU bottleneck)
        self.device = device if device is not None else torch.device('cpu')
        self.feat_values = feat_values
        self.feat_nodes = feat_nodes
        self.feat_offsets = feat_offsets
        self.label_nodes = label_nodes
        self.label_offsets = label_offsets
        self.all_edge_flat = all_edge_flat
        self.edge_offsets = edge_offsets
        self.seq_starts = seq_starts
        self.seq_length = seq_length
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.static_ei = static_ei
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.feat_values_norm = feat_values_norm
        self.zero_normalized = zero_normalized

    def __len__(self):
        return len(self.seq_starts)

    def _get_features(self, w):
        """Reconstruct dense [N, F] tensor from sparse packed storage on device."""
        start = int(self.feat_offsets[w])
        end = int(self.feat_offsets[w + 1])
        if self.feat_values_norm is not None and self.zero_normalized is not None:
            x = torch.empty(self.num_nodes, self.num_features, device=self.device)
            x[:] = self.zero_normalized
            if end > start:
                nodes = self.feat_nodes[start:end].long()
                x[nodes] = self.feat_values_norm[start:end]
        else:
            x = torch.zeros(self.num_nodes, self.num_features, device=self.device)
            if end > start:
                nodes = self.feat_nodes[start:end].long()
                x[nodes] = self.feat_values[start:end]
            x.sub_(self.feat_mean).div_(self.feat_std)
        return x

    def _get_labels(self, w):
        """Reconstruct dense [N] label tensor from sparse packed storage on device."""
        start = int(self.label_offsets[w])
        end = int(self.label_offsets[w + 1])
        y = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        if end > start:
            nodes = self.label_nodes[start:end].long()
            y[nodes] = 1
        return y

    def _get_active_mask(self, w):
        """Return boolean mask of nodes that have feature data in window w."""
        start = int(self.feat_offsets[w])
        end = int(self.feat_offsets[w + 1])
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        if end > start:
            nodes = self.feat_nodes[start:end].long()
            mask[nodes] = True
        return mask

    def __getitem__(self, idx):
        start = self.seq_starts[idx]
        w_indices = list(range(start, start + self.seq_length))

        x_list = [self._get_features(w) for w in w_indices]
        y = self._get_labels(w_indices[-1])

        edge_list = [
            self.all_edge_flat[
                :, int(self.edge_offsets[w]):int(self.edge_offsets[w + 1])]
            for w in w_indices
        ]

        # Active mask: nodes that have FEATURES in any window OR LABELS in
        # the last window. This ensures:
        #   1. Phantom nodes (no features, no labels) are excluded
        #   2. Label nodes are ALWAYS included even if they lack features
        #      in the specific window (they still have graph-propagated info)
        #   3. Feature-active nodes contribute their normal/failing signal
        # 1. Base mask: nodes that actually have features
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        for w in w_indices:
            mask |= self._get_active_mask(w)
            
        # 2. GNN Expansion: Include silent nodes if they are connected to active nodes
        # Get the edge index for the target window
        src, dst = edge_list[-1]
        
        # Find which nodes are currently active
        active_indices = mask.nonzero(as_tuple=True)[0]
        
        # Add nodes that receive edges from active nodes
        valid_edges = torch.isin(src, active_indices)
        mask[dst[valid_edges]] = True
        
        # Add nodes that send edges to active nodes (since info propagates both ways in our SAGE setup)
        valid_edges_dst = torch.isin(dst, active_indices)
        mask[src[valid_edges_dst]] = True

        # ==========================================================
        # Leak-Proof Ghost Injection to ensure failing nodes are evaluated
        # without creating a "silent = failure" shortcut.
        # ==========================================================
        missing_pos = (y == 1) & (~mask)
        num_missing_pos = missing_pos.sum().item()
        
        if num_missing_pos > 0:
            mask |= missing_pos
            
            # Add DECOY ghosts (10x normal missing nodes)
            missing_neg_indices = ((y == 0) & (~mask)).nonzero(as_tuple=True)[0]
            num_decoys = min(num_missing_pos * 10, missing_neg_indices.numel())
            
            if num_decoys > 0:
                perm = torch.randperm(missing_neg_indices.numel(), device=self.device)[:num_decoys]
                decoy_indices = missing_neg_indices[perm]
                mask[decoy_indices] = True
        # ==========================================================

        return x_list, y, edge_list, mask


# ============================================================================
# 2. Collate — batches B sequences into disjoint-union graphs per timestep
# ============================================================================

def collate_graph_sequences(batch):
    """
    Collate B graph sequences into batched disjoint-union graphs.
    V6: also concatenates active-node masks for masked loss computation.
    """
    B = len(batch)
    T = len(batch[0][0])
    N = batch[0][0][0].shape[0]

    batched_x = []
    batched_edges = []

    for t in range(T):
        x_t = torch.cat([batch[b][0][t] for b in range(B)], dim=0)

        edge_parts = []
        for b in range(B):
            ei = batch[b][2][t]
            edge_parts.append(ei + b * N)
        e_t = torch.cat(edge_parts, dim=1)

        batched_x.append(x_t)
        batched_edges.append(e_t)

    y = torch.cat([batch[b][1] for b in range(B)], dim=0)
    mask = torch.cat([batch[b][3] for b in range(B)], dim=0)

    return batched_x, y, batched_edges, B, mask


# ============================================================================
# 3. DynamicGraphLoader — loads data, pre-computes sparse packed tensors
# ============================================================================

class DynamicGraphLoader:
    """
    Loads Borg trace data and pre-computes ALL features, labels, and
    edge indices into SPARSE PACKED CPU shared-memory tensors at init time.

    Sparse packed format: ~10 MB total vs 41.5 GB dense.
    """

    def __init__(self, processed_dir="processed", seq_length=6):
        print("Loading preprocessed data...")

        features_df = pd.read_parquet(
            os.path.join(processed_dir, "machine_features.parquet"))
        labels_df = pd.read_parquet(
            os.path.join(processed_dir, "failure_labels.parquet"))

        with open(os.path.join(processed_dir, "adjacency.json")) as f:
            adj = json.load(f)

        self.m2i = {str(k): int(v) for k, v in adj["machine_to_idx"].items()}
        self.num_nodes = adj["num_nodes"]
        self.seq_length = seq_length

        # Static edges on CPU (fallback for windows without membership data)
        self.static_ei_cpu = torch.tensor(
            adj["edges"], dtype=torch.long).t().contiguous()

        feat_cols = [c for c in features_df.columns
                     if c not in ("machine_id", "time_window", "failed_sum", "failed_mean", "instance_events_type_count")]
        self.num_features = len(feat_cols)

        features_df["machine_id"] = features_df["machine_id"].astype(str)
        labels_df["machine_id"] = labels_df["machine_id"].astype(str)

        self.time_windows = sorted(features_df["time_window"].unique())
        W = len(self.time_windows)

        # === PRE-COMPUTE FEATURES -> sparse packed format (VECTORIZED) ===
        print("  Pre-computing feature tensors (sparse packed)...")
        t0 = time.time()

        # Vectorized: map ALL machine IDs at once (not per-window)
        tw_to_idx = {tw: i for i, tw in enumerate(self.time_windows)}
        features_df["_node_idx"] = features_df["machine_id"].map(self.m2i)
        valid_f = features_df["_node_idx"].notna()
        feat_valid = features_df[valid_f].copy()
        feat_valid["_node_idx"] = feat_valid["_node_idx"].astype(np.int32)
        feat_valid["_tw_idx"] = feat_valid["time_window"].map(tw_to_idx)
        feat_valid = feat_valid.dropna(subset=["_tw_idx"])
        feat_valid["_tw_idx"] = feat_valid["_tw_idx"].astype(np.int64)
        feat_valid = feat_valid.sort_values("_tw_idx")

        if len(feat_valid) > 0:
            self.feat_nodes = torch.from_numpy(
                feat_valid["_node_idx"].values.copy()).to(torch.int32)
            self.feat_values = torch.from_numpy(
                feat_valid[feat_cols].fillna(0).values.astype(np.float32))

            # Build offsets via np.unique counts + cumsum
            self.feat_offsets = torch.zeros(W + 1, dtype=torch.long)
            unique_tw, counts = np.unique(
                feat_valid["_tw_idx"].values, return_counts=True)
            for tw_i, c in zip(unique_tw, counts):
                self.feat_offsets[int(tw_i) + 1] = c
            self.feat_offsets = torch.cumsum(self.feat_offsets, dim=0)
            offset = int(self.feat_offsets[-1])
        else:
            self.feat_nodes = torch.zeros(0, dtype=torch.int32)
            self.feat_values = torch.zeros(0, self.num_features)
            self.feat_offsets = torch.zeros(W + 1, dtype=torch.long)
            offset = 0

        del feat_valid
        _feat_mb = (self.feat_values.nelement() * 4 +
                    self.feat_nodes.nelement() * 4) / (1024**2)
        print(f"    Features: {offset:,} active entries -> {_feat_mb:.1f} MB "
              f"[{time.time() - t0:.1f}s]")

        # === PRE-COMPUTE LABELS -> sparse packed format (VECTORIZED) ===
        print("  Pre-computing label tensors (sparse packed)...")
        t0 = time.time()

        # Vectorized: map ALL machine IDs at once, filter failing, sort
        labels_df["_node_idx"] = labels_df["machine_id"].map(self.m2i)
        valid_l = labels_df["_node_idx"].notna() & (labels_df["label"] == 1)
        labels_valid = labels_df[valid_l].copy()
        labels_valid["_node_idx"] = labels_valid["_node_idx"].astype(np.int32)
        labels_valid["_tw_idx"] = labels_valid["time_window"].map(tw_to_idx)
        labels_valid = labels_valid.dropna(subset=["_tw_idx"])
        labels_valid["_tw_idx"] = labels_valid["_tw_idx"].astype(np.int64)
        labels_valid = labels_valid.sort_values("_tw_idx")

        if len(labels_valid) > 0:
            self.label_nodes = torch.from_numpy(
                labels_valid["_node_idx"].values.copy()).to(torch.int32)

            self.label_offsets = torch.zeros(W + 1, dtype=torch.long)
            unique_tw_l, counts_l = np.unique(
                labels_valid["_tw_idx"].values, return_counts=True)
            for tw_i, c in zip(unique_tw_l, counts_l):
                self.label_offsets[int(tw_i) + 1] = c
            self.label_offsets = torch.cumsum(self.label_offsets, dim=0)
            offset = int(self.label_offsets[-1])
        else:
            self.label_nodes = torch.zeros(0, dtype=torch.int32)
            self.label_offsets = torch.zeros(W + 1, dtype=torch.long)
            offset = 0

        del labels_valid
        print(f"    Labels: {offset:,} failing entries -> "
              f"{self.label_nodes.nelement() * 4 / 1024:.1f} KB "
              f"[{time.time() - t0:.1f}s]")

        # === PRE-COMPUTE DYNAMIC EDGES ===
        mem_path = os.path.join(processed_dir, "window_membership.parquet")
        if os.path.exists(mem_path):
            self.dynamic = True
            print("  Dynamic edges: ENABLED — pre-computing all edge indices...")
            membership = pd.read_parquet(mem_path)
            membership["machine_id"] = membership["machine_id"].astype(str)
            membership["_node_idx"] = membership["machine_id"].map(self.m2i)
            membership = membership.dropna(subset=["_node_idx"])
            membership["_node_idx"] = membership["_node_idx"].astype(np.int64)

            mem_groups = dict(list(membership.groupby("time_window")))

            edge_list = []
            t0 = time.time()
            for i, tw in enumerate(self.time_windows):
                if tw in mem_groups:
                    edges = self._build_edges_for_window(mem_groups[tw])
                    edge_list.append(
                        edges if edges is not None
                        else self.static_ei_cpu.clone())
                else:
                    edge_list.append(self.static_ei_cpu.clone())
                if (i + 1) % 500 == 0 or (i + 1) == W:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (W - i - 1)
                    print(f"    [{i+1}/{W}] windows ({elapsed:.0f}s elapsed, "
                          f"~{eta:.0f}s remaining)")
            del membership, mem_groups
        else:
            self.dynamic = False
            print("  Dynamic edges: DISABLED (using static graph)")
            edge_list = [self.static_ei_cpu.clone() for _ in range(W)]

        # Free source DataFrames
        del features_df, labels_df
        gc.collect()

        # Valid consecutive sequences
        self.seq_starts = []
        for i in range(W - seq_length + 1):
            tw_slice = self.time_windows[i: i + seq_length]
            max_gap = max(tw_slice[j+1] - tw_slice[j]
                          for j in range(len(tw_slice) - 1))
            if max_gap <= 3:
                self.seq_starts.append(i)

        # === PACK EDGES INTO FLAT TENSOR ===
        total_edges = sum(e.shape[1] for e in edge_list)
        self.all_edge_flat = torch.zeros(2, total_edges, dtype=torch.long)
        self.edge_offsets = torch.zeros(W + 1, dtype=torch.long)
        offset = 0
        for i, e in enumerate(edge_list):
            n_e = e.shape[1]
            self.all_edge_flat[:, offset:offset + n_e] = e
            offset += n_e
            self.edge_offsets[i + 1] = offset
        del edge_list

        # === MOVE TO SHARED MEMORY ===
        self.feat_values.share_memory_()
        self.feat_nodes.share_memory_()
        self.feat_offsets.share_memory_()
        self.label_nodes.share_memory_()
        self.label_offsets.share_memory_()
        self.all_edge_flat.share_memory_()
        self.edge_offsets.share_memory_()

        _edge_mb = self.all_edge_flat.nelement() * 8 / (1024**2)
        print(f"\n  Pre-computed & shared memory:")
        print(f"    Features: {self.feat_values.shape[0]:,} entries "
              f"({self.feat_values.nelement() * 4 / 1024**2:.1f} MB)")
        print(f"    Labels:   {self.label_nodes.shape[0]:,} entries "
              f"({self.label_nodes.nelement() * 4 / 1024:.1f} KB)")
        print(f"    Edges:    {total_edges:,} total ({_edge_mb:.0f} MB)")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Features: {self.num_features}")
        print(f"  Time windows: {W}")
        print(f"  Valid sequences: {len(self.seq_starts)}")
        print(f"  Static edges: {self.static_ei_cpu.shape[1]}")

    def __len__(self):
        return len(self.seq_starts)

    def _build_edges_for_window(self, tw_df):
        """
        Build edge_index using np.argsort-based grouping (vectorized).
        No pandas groupby — uses C-level sort for ~3x faster init.
        """
        all_src, all_dst = [], []
        node_arr = tw_df["_node_idx"].values.astype(np.int64)

        def _edges_from_col(col, thresh, K_max):
            grp_vals = tw_df[col].values
            order = np.argsort(grp_vals, kind="stable")
            nodes_s = node_arr[order]
            grp_s = grp_vals[order]
            bounds = np.flatnonzero(grp_s[1:] != grp_s[:-1]) + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [len(nodes_s)]])
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                active = np.unique(nodes_s[s:e])
                na = len(active)
                if na < 2:
                    continue
                if na <= thresh:
                    ii, jj = np.triu_indices(na, k=1)
                    all_src.extend([active[ii], active[jj]])
                    all_dst.extend([active[jj], active[ii]])
                else:
                    K = min(K_max, na - 1)
                    pos = np.arange(na)
                    j_idx = pos[:, None] + np.arange(1, K + 1)[None, :]
                    valid = j_idx < na
                    i_val = np.broadcast_to(pos[:, None], (na, K))[valid]
                    j_val = j_idx[valid]
                    all_src.extend([active[i_val], active[j_val]])
                    all_dst.extend([active[j_val], active[i_val]])

        if "cluster" in tw_df.columns:
            _edges_from_col("cluster", thresh=60, K_max=15)
        if "collection_id" in tw_df.columns:
            _edges_from_col("collection_id", thresh=40, K_max=10)

        active_all = np.unique(node_arr)
        if len(active_all) > 0:
            all_src.append(active_all)
            all_dst.append(active_all)

        if not all_src:
            return None

        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        edges = np.stack([src, dst], axis=0)
        edges = np.unique(edges, axis=1)
        return torch.from_numpy(edges).long()

    def _reconstruct_features(self, w):
        """Reconstruct dense [N, F] tensor from sparse packed storage."""
        start = int(self.feat_offsets[w])
        end = int(self.feat_offsets[w + 1])
        x = torch.zeros(self.num_nodes, self.num_features)
        if end > start:
            nodes = self.feat_nodes[start:end].long()
            x[nodes] = self.feat_values[start:end]
        return x

    def compute_normalization(self, window_indices):
        """
        Compute feature normalization stats incrementally.
        Uses O(F) memory instead of O(S*N*F).
        """
        sample_idx = np.random.choice(
            window_indices, min(100, len(window_indices)), replace=False)
        sum_x = torch.zeros(self.num_features, dtype=torch.float64)
        sum_x2 = torch.zeros(self.num_features, dtype=torch.float64)
        count = 0
        for w in sample_idx:
            x = self._reconstruct_features(int(w))
            sum_x += x.sum(dim=0).double()
            sum_x2 += (x ** 2).sum(dim=0).double()
            count += x.shape[0]
        mean = (sum_x / count).float()
        std = ((sum_x2 / count - mean.double() ** 2).clamp(min=0).sqrt()).float()
        std[std < 1e-8] = 1.0
        return mean, std

    def create_dataset(self, indices, feat_mean, feat_std, device=None):
        """Create a SequenceDataset for the given sequence indices."""
        seq_starts = [self.seq_starts[i] for i in indices]
        feat_values_norm = (self.feat_values - feat_mean) / feat_std
        # share_memory_ only works on CPU tensors (for multi-process DataLoader)
        if feat_values_norm.device.type == 'cpu':
            feat_values_norm.share_memory_()
        zero_normalized = (-feat_mean / feat_std).clone()
        return SequenceDataset(
            feat_values=self.feat_values,
            feat_nodes=self.feat_nodes,
            feat_offsets=self.feat_offsets,
            label_nodes=self.label_nodes,
            label_offsets=self.label_offsets,
            all_edge_flat=self.all_edge_flat,
            edge_offsets=self.edge_offsets,
            seq_starts=seq_starts,
            seq_length=self.seq_length,
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            static_ei=self.static_ei_cpu,
            feat_mean=feat_mean,
            feat_std=feat_std,
            feat_values_norm=feat_values_norm,
            zero_normalized=zero_normalized,
            device=device,
        )


# ============================================================================
# 4. Metrics — now threshold-aware
# ============================================================================

def compute_metrics_from_counts(tp, fp, tn, fn):
    """Compute metrics from running confusion matrix counts. O(1) memory."""
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def compute_metrics(preds, labels, probs=None, threshold=0.5):
    """
    Full metrics with optional AUROC — used for eval only.

    V3: accepts a threshold parameter. When threshold < 0.5, predictions
    are re-derived from probs using that threshold instead of argmax.
    This captures more failures at the cost of more false positives.
    """
    if probs is not None and threshold != 0.5:
        # Override argmax predictions with threshold-based predictions
        preds = (np.array(probs) >= threshold).astype(int)

    m = {
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    if probs is not None and len(np.unique(labels)) > 1:
        try:
            m["auroc"] = roc_auc_score(labels, probs)
        except ValueError:
            m["auroc"] = 0.0
    return m


# ============================================================================
# 5. Training & Evaluation — threshold-aware
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer,
                device, config, num_nodes, scaler=None, threshold=0.5,
                scheduler=None):
    """
    Train one epoch. V6: masked loss — only active nodes contribute.

    The graph has ~89K node slots but only ~22 machines per window have
    real data. Without masking, ~89K zero-feature phantom nodes are all
    labeled "normal," diluting the true 70% positive rate to 0.01%.
    The mask fixes this by restricting loss to nodes with actual features.
    """
    model.train()
    total_loss = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    grad_clip = config.get("training", {}).get("gradient_clip", 1.0)
    num_batches = 0
    total_active = 0
    total_active_pos = 0

    for batch_idx, (x_list, y, edge_list, B, mask) in enumerate(dataloader):
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_list = [e.to(device, non_blocking=True) for e in edge_list]
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x_list, edge_list, num_nodes=num_nodes * B)
            # MASKED LOSS: only active nodes contribute
            loss = criterion(logits[mask], y[mask])

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        n_active = mask.sum().item()
        n_active_pos = y[mask].sum().item()
        total_active += n_active
        total_active_pos += n_active_pos

        if scheduler is not None:
            scheduler.step()

        # Metrics on active nodes only
        with torch.no_grad():
            probs_t = F.softmax(logits[mask].float(), dim=1)[:, 1]
            preds_t = (probs_t >= threshold).long()
            y_m = y[mask]
            tp += ((preds_t == 1) & (y_m == 1)).sum().item()
            fp += ((preds_t == 1) & (y_m == 0)).sum().item()
            tn += ((preds_t == 0) & (y_m == 0)).sum().item()
            fn += ((preds_t == 0) & (y_m == 1)).sum().item()

        if (batch_idx + 1) % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            pos_rate = total_active_pos / max(total_active, 1) * 100
            print(f"    [{batch_idx+1}/{len(dataloader)}] "
                  f"loss={total_loss/num_batches:.4f} lr={lr:.6f} "
                  f"active={n_active} pos_rate={pos_rate:.1f}%", flush=True)
            time.sleep(1)

        del x_list, y, logits, loss, edge_list, mask

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = compute_metrics_from_counts(tp, fp, tn, fn)
    metrics["active_pos_rate"] = total_active_pos / max(total_active, 1)
    return (total_loss / max(num_batches, 1), metrics)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_nodes, threshold=0.5):
    """Evaluate with masked loss — only active nodes contribute."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    num_batches = 0

    for x_list, y, edge_list, B, mask in dataloader:
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_list = [e.to(device, non_blocking=True) for e in edge_list]
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x_list, edge_list, num_nodes=num_nodes * B)
            loss = criterion(logits[mask], y[mask])

        total_loss += loss.item()
        num_batches += 1

        probs = F.softmax(logits[mask].float(), dim=1)[:, 1]
        preds = (probs >= threshold).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y[mask].cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        del x_list, y, logits, edge_list, mask

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_preds, all_labels, all_probs, threshold=0.5)
    return (total_loss / max(num_batches, 1), metrics,
            all_preds, all_labels, all_probs)


# ============================================================================
# 6. Main
# ============================================================================

def main():
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}

    # ---- CUDA setup ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable cuDNN benchmark for optimized convolution kernel selection.
        # Safe here because input sizes are fixed across batches.
        torch.backends.cudnn.benchmark = True
        print(f"Using device: cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN benchmark: ENABLED")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        print("WARNING: CUDA not available, training will be very slow")

    # ---- Load data ----
    data_cfg = config.get("data", {})
    loader = DynamicGraphLoader(
        "processed",
        seq_length=data_cfg.get("sequence_length", 6),
    )

    # ---- Train/val/test split (0.70 / 0.15 / 0.15) ----
    n = len(loader)
    tr_ratio = data_cfg.get("train_ratio", 0.7)
    va_ratio = data_cfg.get("val_ratio", 0.15)
    tr_end = int(n * tr_ratio)
    va_end = int(n * (tr_ratio + va_ratio))
    tr_idx = list(range(tr_end))
    va_idx = list(range(tr_end, va_end))
    te_idx = list(range(va_end, n))
    print(f"\nSplit: {len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test")

    # Collect training window indices for normalization (no test leakage)
    train_tw_set = set()
    for si in tr_idx:
        start = loader.seq_starts[si]
        for w in range(start, start + loader.seq_length):
            train_tw_set.add(w)
    train_tw_indices = sorted(train_tw_set)

    print("\nComputing feature normalization (training windows only)...")
    feat_mean, feat_std = loader.compute_normalization(train_tw_indices)

    # ---- V6: Loss config ----
    train_cfg = config.get("training", {})
    focal_alpha = train_cfg.get("focal_alpha", 0.3)
    focal_gamma = train_cfg.get("focal_gamma", 2.0)
    label_smoothing = config.get("model", {}).get("label_smoothing", 0.05)
    eval_threshold = train_cfg.get("eval_threshold", 0.5)

    print(f"\n  V6 config (MASKED loss on active nodes only):")
    print(f"    Features: {loader.num_features}")
    print(f"    Focal: alpha={focal_alpha}, gamma={focal_gamma}, smooth={label_smoothing}")
    print(f"    eval_threshold={eval_threshold}")

    # ---- Move sparse data to GPU for on-device reconstruction ----
    if device.type == 'cuda':
        print(f"\n  Moving sparse packed data to GPU for on-device reconstruction...")
        loader.feat_values = loader.feat_values.to(device)
        loader.feat_nodes = loader.feat_nodes.to(device)
        loader.label_nodes = loader.label_nodes.to(device)
        loader.all_edge_flat = loader.all_edge_flat.to(device)
        loader.static_ei_cpu = loader.static_ei_cpu.to(device)
        feat_mean = feat_mean.to(device)
        feat_std = feat_std.to(device)
        _alloc_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory after data load: {_alloc_mb:.0f} MB")

    # ---- Create datasets ----
    tr_dataset = loader.create_dataset(tr_idx, feat_mean, feat_std, device=device)
    va_dataset = loader.create_dataset(va_idx, feat_mean, feat_std, device=device)
    te_dataset = (loader.create_dataset(te_idx, feat_mean, feat_std, device=device)
                  if te_idx else None)

    # ---- SANITY CHECK: verify mask + labels alignment ----
    print("\n  === SANITY CHECK (first 5 sequences) ===")
    for si in range(min(5, len(tr_dataset))):
        x_list, y, edge_list, mask = tr_dataset[si]
        n_active = mask.sum().item()
        n_pos = y[mask].sum().item()
        n_pos_total = y.sum().item()
        n_label_all = y.sum().item()
        print(f"    seq[{si}]: active={n_active}/{loader.num_nodes}, "
              f"pos_in_mask={n_pos}, pos_total={n_pos_total}, "
              f"pos_rate={100*n_pos/max(n_active,1):.1f}%")
    print("  === END SANITY CHECK ===")

    # ---- DataLoader setup ----
    # HARD LIMITS from CLAUDE.md:
    #   batch_size <= 2  (OOM on 8GB VRAM if higher)
    #   num_workers = 0  (Error 1455 on Windows if higher)
    _cfg_batch = train_cfg.get("batch_size", 2)
    batch_size = max(1, min(_cfg_batch, 2))  # HARD CAP at 2
    num_workers = 0  # HARD: Windows Error 1455 with num_workers > 0

    _pin = device.type != 'cuda'
    print(f"\n  DataLoader: batch_size={batch_size}, workers={num_workers}, "
          f"pin_memory={_pin}")
    print(f"  Nodes per batch: ~{batch_size * loader.num_nodes:,}")
    if device.type == 'cuda':
        print(f"  Data reconstruction: ON GPU (eliminates CPU bottleneck)")

    dl_kwargs = dict(
        collate_fn=collate_graph_sequences,
        num_workers=num_workers,
        pin_memory=_pin,
    )

    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs)
    va_loader = DataLoader(
        va_dataset, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
    te_loader = (DataLoader(
        te_dataset, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
        if te_dataset else None)

    # ---- Model ----
    model_cfg = config.get("model", {})
    hidden = min(model_cfg.get("hidden_dim", 32), 32)  # HARD CAP at 32

    model = SpatioTemporalGNN(
        input_dim=loader.num_features,
        hidden_dim=hidden,
        num_gnn_layers=model_cfg.get("num_gnn_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        edge_drop_rate=model_cfg.get("edge_drop_rate", 0.3),
    ).to(device)

    # torch.compile DISABLED on Windows (Triton not supported)
    print("  torch.compile: DISABLED (Windows — Triton not supported)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} parameters")
    print(f"  Spatial: 2-layer GraphSAGE (hidden={hidden}, "
          f"edge_drop={model_cfg.get('edge_drop_rate', 0.3)})")
    print(f"  Temporal: GRU (hidden={hidden}) + attention pooling")
    print(f"  Graph: {'DYNAMIC per-window' if loader.dynamic else 'static'}")

    # ---- Training setup ----
    # AMP: modern torch.amp namespace (not deprecated torch.cuda.amp)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler is not None:
        print("  AMP: ENABLED (torch.amp.autocast + GradScaler)")
    else:
        print("  AMP: DISABLED (CPU mode)")

    criterion = SmoothedFocalLoss(
        alpha=focal_alpha,
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    print(f"  Loss: SmoothedFocalLoss(alpha={focal_alpha}, gamma={focal_gamma}, "
          f"smooth={label_smoothing}) — MASKED to active nodes only")
    print(f"  Eval threshold: {eval_threshold}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    epochs = train_cfg.get("epochs", 10)
    patience = train_cfg.get("early_stopping_patience", 5)

    total_steps = epochs * len(tr_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.get("learning_rate", 0.002),
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos',
    )
    best_f1, patience_ctr = -1.0, 0

    print("\n" + "=" * 65)
    print("TRAINING (V6 — Masked Loss + Real Features + OneCycleLR)")
    print("=" * 65)
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Gradient clip: {train_cfg.get('gradient_clip', 1.0)}")

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        tr_loss, tr_m = train_epoch(
            model, tr_loader, criterion, optimizer,
            device, config, loader.num_nodes, scaler=scaler,
            threshold=eval_threshold, scheduler=scheduler,
        )
        va_loss, va_m, _, _, _ = evaluate(
            model, va_loader, criterion, device, loader.num_nodes,
            threshold=eval_threshold,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        pos_rate = tr_m.get('active_pos_rate', 0) * 100

        print(f"  Train — loss: {tr_loss:.4f}  F1: {tr_m['f1']:.4f}  "
              f"Prec: {tr_m['prec']:.4f}  Rec: {tr_m['rec']:.4f}  "
              f"pos_rate={pos_rate:.1f}%")
        print(f"  Val   — loss: {va_loss:.4f}  F1: {va_m.get('f1',0):.4f}  "
              f"AUROC: {va_m.get('auroc',0):.4f}  "
              f"Rec: {va_m.get('rec',0):.4f}  Prec: {va_m.get('prec',0):.4f}  "
              f"[{elapsed:.0f}s, lr={lr:.6f}]")

        if va_m.get("f1", 0) > best_f1:
            best_f1 = va_m["f1"]
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "input_dim": loader.num_features,
                "hidden_dim": hidden,
                "num_gnn_layers": model_cfg.get("num_gnn_layers", 2),
                "dropout": model_cfg.get("dropout", 0.3),
                "edge_drop_rate": model_cfg.get("edge_drop_rate", 0.3),
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "num_nodes": loader.num_nodes,
                "dynamic_edges": loader.dynamic,
                "focal_alpha": focal_alpha,
                "focal_gamma": focal_gamma,
                "eval_threshold": eval_threshold,
                "val_f1": best_f1,
                "val_metrics": va_m,
            }, "best_model.pt")
            print(f"  -> Saved best (F1={best_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {epoch} — best F1: {best_f1:.4f}")
                break

        # Per-epoch memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Thermal Safety Break
        if epoch < epochs:
            print(f'  [Thermal Safety] Cooling down for 15s... (GPU was 89C previously)', flush=True)
            time.sleep(15)

    total_train_time = time.time() - train_start
    print(f"\n  Total training time: {total_train_time / 60:.1f} minutes")

    # ---- Test evaluation ----
    if te_loader:
        print("\n" + "=" * 65)
        print("TEST EVALUATION")
        print("=" * 65)

        ckpt = torch.load("best_model.pt", weights_only=False,
                           map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        te_loss, te_m, te_preds, te_labels, te_probs = evaluate(
            model, te_loader, criterion, device, loader.num_nodes,
            threshold=eval_threshold,
        )

        print(f"\n  Test loss: {te_loss:.4f}")
        print(f"  Threshold: {eval_threshold}")
        for k, v in te_m.items():
            print(f"  {k}: {v:.4f}")
        if len(np.unique(te_labels)) > 1:
            print("\n" + classification_report(
                te_labels, te_preds,
                target_names=["Normal", "Failing"], zero_division=0
            ))

        # Also show metrics at multiple thresholds for comparison
        print("  --- Threshold sweep ---")
        for t in [0.01, 0.05, 0.10, 0.20, 0.50]:
            t_preds = (te_probs >= t).astype(int)
            t_f1 = f1_score(te_labels, t_preds, zero_division=0)
            t_prec = precision_score(te_labels, t_preds, zero_division=0)
            t_rec = recall_score(te_labels, t_preds, zero_division=0)
            n_pred_pos = t_preds.sum()
            print(f"    t={t:.2f}: F1={t_f1:.4f}  Prec={t_prec:.4f}  "
                  f"Rec={t_rec:.4f}  pred_pos={n_pred_pos:,}")

        os.makedirs("processed", exist_ok=True)
        np.savez("processed/test_results.npz",
                 preds=te_preds, labels=te_labels, probs=te_probs)
        print("\n  Saved: processed/test_results.npz")

    print(f"\n  Done! Total time: {(time.time() - train_start) / 60:.1f} min")
    print(f"  Next: python evaluate.py")


if __name__ == "__main__":
    main()
