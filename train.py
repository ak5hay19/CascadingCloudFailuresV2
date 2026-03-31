"""
STEP 2: Train the Spatio-Temporal GNN
=======================================
    python train.py

Now uses ALL nodes (~4,900) with GraphSAGE neighbor sampling.
No more 500-node subsampling — the full infrastructure graph is used.

GraphSAGE samples 15 neighbors per node per layer, keeping VRAM
bounded regardless of graph density. ~17MB per forward pass at
4,900 nodes — fits easily on RTX 4060 (8GB).
"""

import os
import gc
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from model import SpatioTemporalGNN, FocalLoss


class GraphSequenceLoader:
    """
    Loads preprocessed data and builds graph sequences on-the-fly.

    No node subsampling — we use ALL machines. GraphSAGE's neighbor
    sampling handles the scalability, not data reduction.
    """

    def __init__(self, processed_dir="processed", seq_length=6):
        print("Loading preprocessed data...")

        self.features = pd.read_parquet(os.path.join(processed_dir, "machine_features.parquet"))
        self.labels = pd.read_parquet(os.path.join(processed_dir, "failure_labels.parquet"))

        with open(os.path.join(processed_dir, "adjacency.json")) as f:
            adj = json.load(f)

        self.m2i = {str(k): int(v) for k, v in adj["machine_to_idx"].items()}
        self.num_nodes = adj["num_nodes"]
        self.seq_length = seq_length

        self.edge_index = torch.tensor(adj["edges"], dtype=torch.long).t().contiguous()
        self.edge_weight = torch.tensor(
            adj.get("edge_weights", [1.0] * len(adj["edges"])), dtype=torch.float
        )

        self.feat_cols = [c for c in self.features.columns
                          if c not in ("machine_id", "time_window")]
        self.num_features = len(self.feat_cols)

        self.features["machine_id"] = self.features["machine_id"].astype(str)
        self.labels["machine_id"] = self.labels["machine_id"].astype(str)

        self.time_windows = sorted(self.features["time_window"].unique())

        self._feat_groups = dict(list(self.features.groupby("time_window")))
        self._label_groups = dict(list(self.labels.groupby("time_window")))

        # Valid consecutive sequences
        self.seq_starts = []
        for i in range(len(self.time_windows) - seq_length + 1):
            tw_slice = self.time_windows[i: i + seq_length]
            max_gap = max(tw_slice[j+1] - tw_slice[j] for j in range(len(tw_slice)-1))
            if max_gap <= 3:
                self.seq_starts.append(i)

        print(f"  Nodes: {self.num_nodes} (ALL machines, no subsampling)")
        print(f"  Features: {self.num_features}")
        print(f"  Time windows: {len(self.time_windows)}")
        print(f"  Valid sequences: {len(self.seq_starts)}")
        print(f"  Edges: {self.edge_index.shape[1]}")

    def __len__(self):
        return len(self.seq_starts)

    def get_snapshot(self, tw):
        x = np.zeros((self.num_nodes, self.num_features), dtype=np.float32)
        y = np.zeros(self.num_nodes, dtype=np.int64)

        if tw in self._feat_groups:
            df = self._feat_groups[tw]
            idx_map = df["machine_id"].map(self.m2i)
            valid = idx_map.notna()
            if valid.any():
                indices = idx_map[valid].astype(int).values
                vals = df.loc[valid.values, self.feat_cols].fillna(0).values.astype(np.float32)
                x[indices] = vals

        if tw in self._label_groups:
            df = self._label_groups[tw]
            idx_map = df["machine_id"].map(self.m2i)
            valid = idx_map.notna()
            if valid.any():
                y[idx_map[valid].astype(int).values] = 1

        return torch.tensor(x), torch.tensor(y)

    def get_sequence(self, seq_idx):
        start = self.seq_starts[seq_idx]
        tws = self.time_windows[start: start + self.seq_length]
        x_list = [self.get_snapshot(tw)[0] for tw in tws]
        _, y = self.get_snapshot(tws[-1])
        return x_list, y

    def compute_normalization(self):
        sample_tws = np.random.choice(
            self.time_windows, min(100, len(self.time_windows)), replace=False
        )
        all_x = torch.cat([self.get_snapshot(tw)[0] for tw in sample_tws], dim=0)
        mean = all_x.mean(dim=0)
        std = all_x.std(dim=0)
        std[std < 1e-8] = 1.0
        return mean, std


def compute_metrics(preds, labels, probs=None):
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


def train_epoch(model, loader, train_idx, criterion, optimizer,
                device, feat_mean, feat_std, config):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    batch_size = config.get("training", {}).get("batch_size", 8)
    grad_clip = config.get("training", {}).get("gradient_clip", 1.0)

    idx = list(train_idx)
    np.random.shuffle(idx)

    edge_index = loader.edge_index.to(device)
    edge_weight = loader.edge_weight.to(device)
    fm = feat_mean.to(device)
    fs = feat_std.to(device)

    optimizer.zero_grad()
    batch_loss = 0.0

    for i, si in enumerate(idx):
        x_list, y = loader.get_sequence(si)
        x_list = [((x.to(device) - fm) / fs) for x in x_list]
        y = y.to(device)

        logits = model(x_list, edge_index, edge_weight, num_nodes=loader.num_nodes)
        loss = criterion(logits, y) / batch_size
        loss.backward()

        batch_loss += loss.item() * batch_size
        all_preds.extend(logits.detach().argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        if (i + 1) % batch_size == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss
            batch_loss = 0.0

        del x_list, y, logits, loss
        if (i + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(idx)}]", flush=True)

    # Final partial batch
    if len(idx) % batch_size != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += batch_loss

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / max(len(idx), 1), compute_metrics(all_preds, all_labels)


@torch.no_grad()
def evaluate(model, loader, indices, criterion, device, feat_mean, feat_std):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    edge_index = loader.edge_index.to(device)
    edge_weight = loader.edge_weight.to(device)
    fm = feat_mean.to(device)
    fs = feat_std.to(device)

    for si in indices:
        x_list, y = loader.get_sequence(si)
        x_list = [((x.to(device) - fm) / fs) for x in x_list]
        y = y.to(device)

        logits = model(x_list, edge_index, edge_weight, num_nodes=loader.num_nodes)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = F.softmax(logits, dim=1)[:, 1]
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        del x_list, y, logits

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = compute_metrics(
        np.array(all_preds), np.array(all_labels), np.array(all_probs)
    )
    return (total_loss / max(len(indices), 1), metrics,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def main():
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    data_cfg = config.get("data", {})
    loader = GraphSequenceLoader(
        "processed",
        seq_length=data_cfg.get("sequence_length", 6),
    )

    print("\nComputing feature normalization...")
    feat_mean, feat_std = loader.compute_normalization()

    n = len(loader)
    tr_end = int(n * data_cfg.get("train_ratio", 0.7))
    va_end = int(n * (data_cfg.get("train_ratio", 0.7) + data_cfg.get("val_ratio", 0.15)))
    tr_idx = list(range(tr_end))
    va_idx = list(range(tr_end, va_end))
    te_idx = list(range(va_end, n))
    print(f"Split: {len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test")

    model_cfg = config.get("model", {})
    hidden = model_cfg.get("hidden_dim", 48)
    num_neighbors = model_cfg.get("num_neighbors", 15)

    model = SpatioTemporalGNN(
        input_dim=loader.num_features,
        hidden_dim=hidden,
        num_gnn_layers=model_cfg.get("num_gnn_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        num_neighbors=num_neighbors,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} parameters")
    print(f"  Spatial: 2-layer GraphSAGE (hidden={hidden}, k={num_neighbors} neighbors)")
    print(f"  Temporal: GRU (hidden={hidden})")
    print(f"  Classifier: MLP ({hidden} -> {hidden//2} -> 2)")

    train_cfg = config.get("training", {})
    criterion = FocalLoss(
        alpha=train_cfg.get("focal_alpha", 0.75),
        gamma=train_cfg.get("focal_gamma", 2.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    epochs = train_cfg.get("epochs", 30)
    patience = train_cfg.get("early_stopping_patience", 10)
    best_f1, patience_ctr = -1.0, 0

    print("\n" + "=" * 65)
    print("TRAINING")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        tr_loss, tr_m = train_epoch(
            model, loader, tr_idx, criterion, optimizer,
            device, feat_mean, feat_std, config
        )
        va_loss, va_m, _, _, _ = evaluate(
            model, loader, va_idx, criterion, device, feat_mean, feat_std
        )

        elapsed = time.time() - t0
        scheduler.step(va_m.get("f1", 0))
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Train — loss: {tr_loss:.4f}  F1: {tr_m['f1']:.4f}")
        print(f"  Val   — loss: {va_loss:.4f}  F1: {va_m.get('f1',0):.4f}  "
              f"AUROC: {va_m.get('auroc',0):.4f}  Recall: {va_m.get('rec',0):.4f}  "
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
                "num_neighbors": num_neighbors,
                "dropout": model_cfg.get("dropout", 0.3),
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "num_nodes": loader.num_nodes,
                "val_f1": best_f1,
                "val_metrics": va_m,
            }, "best_model.pt")
            print(f"  -> Saved best (F1={best_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping — best F1: {best_f1:.4f}")
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Test
    if te_idx:
        print("\n" + "=" * 65)
        print("TEST EVALUATION")
        print("=" * 65)

        ckpt = torch.load("best_model.pt", weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        te_loss, te_m, te_preds, te_labels, te_probs = evaluate(
            model, loader, te_idx, criterion, device, feat_mean, feat_std
        )

        print(f"\n  Test loss: {te_loss:.4f}")
        for k, v in te_m.items():
            print(f"  {k}: {v:.4f}")
        if len(np.unique(te_labels)) > 1:
            print("\n" + classification_report(
                te_labels, te_preds,
                target_names=["Normal", "Failing"], zero_division=0
            ))

        np.savez("processed/test_results.npz",
                 preds=te_preds, labels=te_labels, probs=te_probs)
        print("  Saved: processed/test_results.npz")

    print(f"\n✓ Done! Run: python evaluate.py")


if __name__ == "__main__":
    main()
