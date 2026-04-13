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

Pipeline:
  1. PRE-COMPUTE at init: features/labels in SPARSE PACKED format.
  2. SHARED MEMORY: packed tensors in torch shared memory.
  3. num_workers=0 with pin_memory=True (Windows-safe).
  4. TRUE MINI-BATCHING: B sequences batched via disjoint graph union.
  5. AMP: fp16 forward on Tensor Cores via torch.amp.autocast('cuda').
  6. cudnn.benchmark=True for optimized convolution kernels.

FIX LOG:
  - Removed torch.compile (Windows incompatible)
  - Set num_workers=0 (prevents Error 1455 shared file mapping crash)
  - Modern torch.amp namespace throughout
  - non_blocking=True on all .to(device) calls
  - Explicit gc.collect() + empty_cache() per epoch
  - cudnn.benchmark=True for GPU kernel optimization
  - Dynamic focal_alpha from metadata.json if available
  - Auto-scaled batch_size capped at 2 for 8GB VRAM safety
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
from model import SpatioTemporalGNN, FocalLoss


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

        return x_list, y, edge_list


# ============================================================================
# 2. Collate — batches B sequences into disjoint-union graphs per timestep
# ============================================================================

def collate_graph_sequences(batch):
    """
    Collate B graph sequences into batched disjoint-union graphs.

    At each timestep t:
      - Cat B copies of [N, F] features -> [B*N, F]
      - Offset each graph's edges by b*N and cat -> [2, sum(E_b)]

    Creates B disconnected sub-graphs. GraphSAGE processes all B*N nodes
    in one call — O(B) better GPU utilization than sequential processing.
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

    return batched_x, y, batched_edges, B


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
                     if c not in ("machine_id", "time_window")]
        self.num_features = len(feat_cols)

        features_df["machine_id"] = features_df["machine_id"].astype(str)
        labels_df["machine_id"] = labels_df["machine_id"].astype(str)

        self.time_windows = sorted(features_df["time_window"].unique())
        W = len(self.time_windows)

        # === PRE-COMPUTE FEATURES -> sparse packed format ===
        print("  Pre-computing feature tensors (sparse packed)...")
        t0 = time.time()
        feat_groups = dict(list(features_df.groupby("time_window")))

        _feat_node_parts = []
        _feat_val_parts = []
        self.feat_offsets = torch.zeros(W + 1, dtype=torch.long)
        offset = 0

        for i, tw in enumerate(self.time_windows):
            if tw in feat_groups:
                df = feat_groups[tw]
                idx_map = df["machine_id"].map(self.m2i)
                valid = idx_map.notna()
                if valid.any():
                    nidx = idx_map[valid].astype(int).values.astype(np.int32)
                    vals = df.loc[valid.values, feat_cols].fillna(0) \
                             .values.astype(np.float32)
                    _feat_node_parts.append(nidx)
                    _feat_val_parts.append(vals)
                    offset += len(nidx)
            self.feat_offsets[i + 1] = offset

        del feat_groups

        if _feat_node_parts:
            self.feat_nodes = torch.from_numpy(
                np.concatenate(_feat_node_parts)).to(torch.int32)
            self.feat_values = torch.from_numpy(
                np.concatenate(_feat_val_parts))  # float32
        else:
            self.feat_nodes = torch.zeros(0, dtype=torch.int32)
            self.feat_values = torch.zeros(0, self.num_features)

        del _feat_node_parts, _feat_val_parts
        _feat_mb = (self.feat_values.nelement() * 4 +
                    self.feat_nodes.nelement() * 4) / (1024**2)
        print(f"    Features: {offset:,} active entries -> {_feat_mb:.1f} MB "
              f"[{time.time() - t0:.1f}s]")

        # === PRE-COMPUTE LABELS -> sparse packed format ===
        print("  Pre-computing label tensors (sparse packed)...")
        t0 = time.time()
        label_groups = dict(list(labels_df.groupby("time_window")))

        _label_node_parts = []
        self.label_offsets = torch.zeros(W + 1, dtype=torch.long)
        offset = 0

        for i, tw in enumerate(self.time_windows):
            if tw in label_groups:
                df = label_groups[tw]
                idx_map = df["machine_id"].map(self.m2i)
                valid = idx_map.notna()
                if valid.any():
                    label_vals = df.loc[valid.values, "label"].values
                    nidx_all = idx_map[valid].astype(int).values
                    failing_mask = (label_vals == 1)
                    if failing_mask.any():
                        nidx = nidx_all[failing_mask].astype(np.int32)
                        _label_node_parts.append(nidx)
                        offset += len(nidx)
            self.label_offsets[i + 1] = offset

        del label_groups

        if _label_node_parts:
            self.label_nodes = torch.from_numpy(
                np.concatenate(_label_node_parts)).to(torch.int32)
        else:
            self.label_nodes = torch.zeros(0, dtype=torch.int32)

        del _label_node_parts
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
# 4. Metrics
# ============================================================================

def compute_metrics_from_counts(tp, fp, tn, fn):
    """Compute metrics from running confusion matrix counts. O(1) memory."""
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def compute_metrics(preds, labels, probs=None):
    """Full metrics with optional AUROC — used for eval only."""
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
# 5. Training & Evaluation
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer,
                device, config, num_nodes, scaler=None):
    """
    Train one epoch using DataLoader with true mini-batching.

    Each batch contains B graph sequences batched into one disjoint-union
    graph of B*N nodes. One forward + backward pass per batch.
    All .to(device) calls use non_blocking=True for async DMA transfers.
    """
    model.train()
    total_loss = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    grad_clip = config.get("training", {}).get("gradient_clip", 1.0)
    num_batches = 0

    for batch_idx, (x_list, y, edge_list, B) in enumerate(dataloader):
        # Move to GPU — non_blocking overlaps DMA with previous GPU work
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_list = [e.to(device, non_blocking=True) for e in edge_list]
        y = y.to(device, non_blocking=True)

        # Forward + backward
        # AMP: fp16 forward on Tensor Cores -> ~2x GPU throughput on RTX 4060
        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x_list, edge_list, num_nodes=num_nodes * B)
            loss = criterion(logits, y)

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

        # O(1) running confusion matrix
        with torch.no_grad():
            preds_t = logits.argmax(1)
            tp += ((preds_t == 1) & (y == 1)).sum().item()
            fp += ((preds_t == 1) & (y == 0)).sum().item()
            tn += ((preds_t == 0) & (y == 0)).sum().item()
            fn += ((preds_t == 0) & (y == 1)).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"    [{batch_idx+1}/{len(dataloader)}] batches", flush=True)

        # Explicit cleanup to prevent memory leaks across batches
        del x_list, y, logits, loss, edge_list

    # End-of-epoch cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (total_loss / max(num_batches, 1),
            compute_metrics_from_counts(tp, fp, tn, fn))


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_nodes):
    """Evaluate using DataLoader with true mini-batching."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    num_batches = 0

    for x_list, y, edge_list, B in dataloader:
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_list = [e.to(device, non_blocking=True) for e in edge_list]
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x_list, edge_list, num_nodes=num_nodes * B)
            loss = criterion(logits, y)

        total_loss += loss.item()
        num_batches += 1

        probs = F.softmax(logits.float(), dim=1)[:, 1]
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        del x_list, y, logits, edge_list

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = compute_metrics(
        np.array(all_preds), np.array(all_labels), np.array(all_probs)
    )
    return (total_loss / max(num_batches, 1), metrics,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


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

    # ---- Dynamic focal_alpha ----
    # Try loading from metadata.json first (computed in preprocess.py)
    train_cfg = config.get("training", {})
    metadata_path = os.path.join("processed", "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        dynamic_alpha = metadata.get("recommended_focal_alpha",
                                      train_cfg.get("focal_alpha", 0.99))
        print(f"  Focal alpha from metadata.json: {dynamic_alpha:.3f} "
              f"(failure_ratio={metadata.get('failure_ratio', 'N/A')})")
    else:
        # Fallback: compute from training labels
        print("  Computing class balance for dynamic focal alpha...")
        _sample = tr_idx[:min(200, len(tr_idx))]
        _fail, _total = 0, 0
        for si in _sample:
            start = loader.seq_starts[si]
            tw_last = start + loader.seq_length - 1
            l_start = int(loader.label_offsets[tw_last])
            l_end = int(loader.label_offsets[tw_last + 1])
            _fail += (l_end - l_start)
            _total += loader.num_nodes
        _pos_rate = _fail / max(_total, 1)
        dynamic_alpha = float(max(0.5, min(0.99, 1.0 - _pos_rate)))
        print(f"  Failure rate: {_pos_rate:.6f} -> focal_alpha={dynamic_alpha:.3f}")

    # Override with config if explicitly set
    focal_alpha = train_cfg.get("focal_alpha", dynamic_alpha)
    focal_gamma = train_cfg.get("focal_gamma", 3.0)

    # ---- Move sparse data to GPU for on-device reconstruction ----
    # ROOT CAUSE FIX: With num_workers=0, DataLoader reconstructs dense [N,F]
    # tensors from sparse format ON CPU in the main thread. The tiny model
    # (hidden=32) finishes GPU forward/backward in ms, but CPU spends seconds
    # on dense reconstruction -> GPU sits idle at 1W.
    # FIX: Move sparse packed data (~10 MB) to GPU. Scatter-based reconstruction
    # now runs on GPU (massively parallel) -> GPU stays busy.
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

    # pin_memory only useful for CPU->GPU DMA. With on-device reconstruction,
    # data is already on GPU — pinning CUDA tensors raises an error.
    dl_kwargs = dict(
        collate_fn=collate_graph_sequences,
        num_workers=num_workers,
        pin_memory=_pin,
        # persistent_workers and prefetch_factor only valid with num_workers>0
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
    # Do NOT attempt torch.compile on any platform to avoid issues.
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

    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    print(f"  Loss: FocalLoss(alpha={focal_alpha:.3f}, gamma={focal_gamma:.1f})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    epochs = train_cfg.get("epochs", 10)
    patience = train_cfg.get("early_stopping_patience", 5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=train_cfg.get("learning_rate", 0.001) * 0.01,
    )
    best_f1, patience_ctr = -1.0, 0

    print("\n" + "=" * 65)
    print("TRAINING")
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
            device, config, loader.num_nodes, scaler=scaler
        )
        va_loss, va_m, _, _, _ = evaluate(
            model, va_loader, criterion, device, loader.num_nodes
        )

        elapsed = time.time() - t0
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Train — loss: {tr_loss:.4f}  F1: {tr_m['f1']:.4f}  "
              f"Prec: {tr_m['prec']:.4f}  Rec: {tr_m['rec']:.4f}")
        print(f"  Val   — loss: {va_loss:.4f}  F1: {va_m.get('f1',0):.4f}  "
              f"AUROC: {va_m.get('auroc',0):.4f}  "
              f"Rec: {va_m.get('rec',0):.4f}  "
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
            model, te_loader, criterion, device, loader.num_nodes
        )

        print(f"\n  Test loss: {te_loss:.4f}")
        for k, v in te_m.items():
            print(f"  {k}: {v:.4f}")
        if len(np.unique(te_labels)) > 1:
            print("\n" + classification_report(
                te_labels, te_preds,
                target_names=["Normal", "Failing"], zero_division=0
            ))

        os.makedirs("processed", exist_ok=True)
        np.savez("processed/test_results.npz",
                 preds=te_preds, labels=te_labels, probs=te_probs)
        print("  Saved: processed/test_results.npz")

    print(f"\n  Done! Total time: {(time.time() - train_start) / 60:.1f} min")
    print(f"  Next: python evaluate.py")


if __name__ == "__main__":
    main()
