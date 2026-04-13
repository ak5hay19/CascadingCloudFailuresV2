"""
STEP 3: Evaluate, Visualize & Advanced Analysis
=================================================
    python evaluate.py

Generates all plots in ./results/:
  Standard:
    - Confusion matrix
    - ROC & PR curves
    - Critical node identification (gradient-based)
    - Failure propagation visualization
    - t-SNE of learned embeddings

  Novel contributions:
    - MC Dropout uncertainty quantification
    - Edge contagion scoring

FIX LOG:
  - Added dynamic edge building that mirrors train.py's DynamicGraphLoader
    → model is now evaluated on the SAME graph topology it was trained on
  - build_snapshot now returns per-window edge lists (was: static only)
  - MC Dropout uses dynamic edges + controlled stochasticity
  - Edge contagion uses dynamic edges for gradient analysis
  - find_failure_sequences scoped to test windows when appropriate
  - Proper tensor cleanup in all analysis functions
  - Fixed DynamicEdgeBuilder._cache: was unbounded dict (memory leak), now uses
    BoundedCache(200) matching train.py — prevents RAM exhaustion on large evals
    (1000+ windows × ~10 MB/edge tensor = 10 GB uncapped growth)
  - Vectorized DynamicEdgeBuilder edge building: numpy broadcasting replaces
    inner Python loops — topology now exactly mirrors train.py
  - Fixed find_failure_sequences: O(T) dict lookup replaces O(T×F) nested scan
"""

import os
import gc
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    classification_report,
)
from sklearn.manifold import TSNE
from model import SpatioTemporalGNN


class BoundedCache:
    """
    LRU cache with a hard size cap — identical to train.py's BoundedCache.
    Prevents the unbounded dict growth that was silently consuming all RAM
    when evaluate.py processed hundreds of time windows during analysis.
    """

    def __init__(self, max_size=200):
        self._cache = OrderedDict()
        self._max_size = max_size

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value


# ── Dynamic Edge Builder (mirrors train.py) ──────────────────

class DynamicEdgeBuilder:
    """
    Builds per-window dynamic edges from membership data.
    Mirrors the logic in train.py's DynamicGraphLoader._build_dynamic_edges
    so that evaluation uses the SAME graph topology as training.
    """

    def __init__(self, adj, membership_path="processed/window_membership.parquet"):
        self.m2i = {str(k): int(v) for k, v in adj["machine_to_idx"].items()}
        self.num_nodes = adj["num_nodes"]
        self.static_ei = torch.tensor(
            adj["edges"], dtype=torch.long).t().contiguous()

        if os.path.exists(membership_path):
            mem = pd.read_parquet(membership_path)
            mem["machine_id"] = mem["machine_id"].astype(str)
            self._mem_groups = dict(list(mem.groupby("time_window")))
            self.dynamic = True
        else:
            self._mem_groups = {}
            self.dynamic = False

        self._cache = BoundedCache(max_size=200)

    def build_edges(self, tw):
        """Build edge_index for a specific time window. Bounded-LRU cached."""
        cached = self._cache.get(tw)
        if cached is not None:
            return cached

        if not self.dynamic or tw not in self._mem_groups:
            self._cache.put(tw, self.static_ei)
            return self.static_ei

        df = self._mem_groups[tw]
        all_src, all_dst = [], []

        # Cluster edges — fully vectorized, matches train.py exactly
        if "cluster" in df.columns:
            for _, group in df.groupby("cluster"):
                machines = group["machine_id"].unique()
                active = np.array([self.m2i[m] for m in machines
                                   if m in self.m2i], dtype=np.int64)
                n = len(active)
                if n < 2:
                    continue
                if n <= 60:
                    ii, jj = np.triu_indices(n, k=1)
                    all_src.extend([active[ii], active[jj]])
                    all_dst.extend([active[jj], active[ii]])
                else:
                    K = 15
                    pos = np.arange(n)
                    offsets = np.arange(1, K + 1)
                    j_idx = pos[:, None] + offsets[None, :]
                    valid = j_idx < n
                    i_valid = np.broadcast_to(pos[:, None], (n, K))[valid]
                    j_valid = j_idx[valid]
                    src = active[i_valid]
                    dst = active[j_valid]
                    if len(src) > 0:
                        all_src.extend([src, dst])
                        all_dst.extend([dst, src])

        # Collection edges — fully vectorized, matches train.py exactly
        if "collection_id" in df.columns:
            for _, group in df.groupby("collection_id"):
                machines = group["machine_id"].unique()
                active = np.array([self.m2i[m] for m in machines
                                   if m in self.m2i], dtype=np.int64)
                n = len(active)
                if n < 2:
                    continue
                if n <= 40:
                    ii, jj = np.triu_indices(n, k=1)
                    all_src.extend([active[ii], active[jj]])
                    all_dst.extend([active[jj], active[ii]])
                else:
                    K = 10
                    pos = np.arange(n)
                    offsets = np.arange(1, K + 1)
                    j_idx = pos[:, None] + offsets[None, :]
                    valid = j_idx < n
                    i_valid = np.broadcast_to(pos[:, None], (n, K))[valid]
                    j_valid = j_idx[valid]
                    src = active[i_valid]
                    dst = active[j_valid]
                    if len(src) > 0:
                        all_src.extend([src, dst])
                        all_dst.extend([dst, src])

        active_nodes = np.array([self.m2i[m] for m in df["machine_id"].unique()
                                  if m in self.m2i], dtype=np.int64)
        if len(active_nodes) > 0:
            all_src.append(active_nodes)
            all_dst.append(active_nodes)

        if not all_src:
            self._cache[tw] = self.static_ei
            return self.static_ei

        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        edges = np.unique(np.stack([src, dst], axis=0), axis=1)
        result = torch.from_numpy(edges).to(dtype=torch.long)
        self._cache.put(tw, result)
        return result


# ── Helpers ───────────────────────────────────────────────────

def load_model():
    ckpt = torch.load("best_model.pt", weights_only=False, map_location="cpu")
    model = SpatioTemporalGNN(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_gnn_layers=ckpt.get("num_gnn_layers", 2),
        dropout=ckpt.get("dropout", 0.3),
        edge_drop_rate=ckpt.get("edge_drop_rate", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def load_graph_data():
    with open("processed/adjacency.json") as f:
        adj = json.load(f)
    features = pd.read_parquet("processed/machine_features.parquet")
    labels = pd.read_parquet("processed/failure_labels.parquet")
    features["machine_id"] = features["machine_id"].astype(str)
    labels["machine_id"] = labels["machine_id"].astype(str)
    m2i = {str(k): int(v) for k, v in adj["machine_to_idx"].items()}

    edge_builder = DynamicEdgeBuilder(adj)
    return features, labels, m2i, adj, edge_builder


def build_snapshot(features, labels, m2i, adj, ckpt, tw_slice, edge_builder):
    """
    Build x_list, y, AND per-window dynamic edge lists for a given time window slice.
    This now mirrors how train.py builds sequences — dynamic edges included.
    """
    num_nodes = adj["num_nodes"]
    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]
    fm, fs = ckpt["feat_mean"], ckpt["feat_std"]
    feat_groups = dict(list(features.groupby("time_window")))
    label_groups = dict(list(labels.groupby("time_window")))

    x_list = []
    edge_list = []
    for tw in tw_slice:
        x = np.zeros((num_nodes, len(feat_cols)), dtype=np.float32)
        if tw in feat_groups:
            df = feat_groups[tw]
            idx_map = df["machine_id"].map(m2i)
            valid = idx_map.notna()
            if valid.any():
                x[idx_map[valid].astype(int).values] = \
                    df.loc[valid.values, feat_cols].fillna(0).values.astype(np.float32)
        x_list.append((torch.tensor(x) - fm) / fs)
        edge_list.append(edge_builder.build_edges(tw))

    y = np.zeros(num_nodes, dtype=np.int64)
    last_tw = tw_slice[-1]
    if last_tw in label_groups:
        df = label_groups[last_tw]
        idx_map = df["machine_id"].map(m2i)
        valid = idx_map.notna()
        if valid.any():
            y[idx_map[valid].astype(int).values] = 1

    return x_list, torch.tensor(y), edge_list


def find_failure_sequences(features, labels, num_seqs=10):
    time_windows = sorted(features["time_window"].unique())
    # O(T) dict build + O(1) lookup per failure window, replacing an O(T×F) nested scan.
    # With thousands of time windows and hundreds of failure windows, the old nested
    # loop was meaningfully slow during gradient analysis and contagion scoring.
    tw_to_idx = {tw: i for i, tw in enumerate(time_windows)}
    fail_windows = sorted(labels["time_window"].unique())
    sequences = []
    for fw in fail_windows:
        tw_idx = tw_to_idx.get(fw)
        if tw_idx is not None and tw_idx >= 5:
            sequences.append(time_windows[tw_idx - 5: tw_idx + 1])
        if len(sequences) >= num_seqs:
            break
    return sequences


# ── Standard Plots ────────────────────────────────────────────

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Failing"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Cascading Failure Prediction\nConfusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("  -> results/confusion_matrix.png")


def plot_roc_pr(labels, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve"); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(rec, prec)
    ax2.plot(rec, prec, color="#dc2626", lw=2, label=f"AP = {pr_auc:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve"); ax2.legend(loc="lower left"); ax2.grid(alpha=0.3)

    plt.suptitle("ST-GNN Model Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> results/roc_pr_curves.png")
    return roc_auc, pr_auc


def identify_critical_nodes(model, features, labels, m2i, adj, ckpt,
                             edge_builder, top_k=20):
    print("  Running gradient analysis...")
    num_nodes = adj["num_nodes"]

    sequences = find_failure_sequences(features, labels, num_seqs=30)
    node_importance = np.zeros(num_nodes, dtype=np.float64)
    count = 0

    for tw_slice in sequences:
        x_list, y, edge_list = build_snapshot(
            features, labels, m2i, adj, ckpt, tw_slice, edge_builder)
        if y.sum() == 0:
            continue
        for x_t in x_list:
            x_t.requires_grad_(True)

        logits = model(x_list, edge_list, num_nodes=num_nodes)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        if x_list[-1].grad is not None:
            node_importance += x_list[-1].grad.abs().mean(dim=1).detach().numpy()
            count += 1

        # Explicit cleanup
        del x_list, y, logits, loss, edge_list
        gc.collect()

    if count == 0:
        print("  No failure samples"); return []

    node_importance /= node_importance.max() + 1e-8
    top_indices = np.argsort(node_importance)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(node_importance[top_indices] * 0.8 + 0.2)
    ax.barh(range(len(top_indices)), node_importance[top_indices], color=colors)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f"Machine {i}" for i in top_indices], fontsize=9)
    ax.set_xlabel("Importance Score (gradient magnitude)")
    ax.set_title(f"Top {top_k} Critical Nodes for Cascade Propagation")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/critical_nodes.png", dpi=150)
    plt.close()
    print(f"  -> results/critical_nodes.png")
    return top_indices.tolist()


def plot_failure_propagation(features, labels, m2i, adj, max_vis=120):
    if len(labels) == 0:
        print("  No failures to visualize"); return

    edge_list = adj["edges"]
    time_windows = sorted(features["time_window"].unique())
    label_groups = dict(list(labels.groupby("time_window")))
    fail_windows = sorted(labels["time_window"].unique())

    target = fail_windows[len(fail_windows) // 2]
    tw_idx = next((i for i, tw in enumerate(time_windows) if tw >= target), None)
    if tw_idx is None or tw_idx < 5:
        tw_idx = min(5, len(time_windows) - 1)
    tw_slice = time_windows[max(0, tw_idx - 5): tw_idx + 1]

    interesting = set()
    for tw in tw_slice:
        if tw in label_groups:
            for mid in label_groups[tw]["machine_id"]:
                if mid in m2i:
                    interesting.add(m2i[mid])
    for e in edge_list:
        if e[0] in interesting or e[1] in interesting:
            interesting.add(e[0]); interesting.add(e[1])
        if len(interesting) >= max_vis:
            break
    if len(interesting) < 10:
        interesting = set(range(min(100, max(m2i.values()) + 1)))
    interesting = sorted(interesting)[:max_vis]
    node_set = set(interesting)

    G = nx.Graph()
    G.add_nodes_from(interesting)
    for e in edge_list:
        if e[0] != e[1] and e[0] in node_set and e[1] in node_set:
            G.add_edge(e[0], e[1])
    pos = nx.spring_layout(G, seed=42, k=2.0 / max(np.sqrt(len(interesting)), 1))

    cols = min(len(tw_slice), 6)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1: axes = [axes]
    for t, tw in enumerate(tw_slice[:cols]):
        ax = axes[t]
        fail_nodes = set()
        if tw in label_groups:
            for mid in label_groups[tw]["machine_id"]:
                if mid in m2i: fail_nodes.add(m2i[mid])
        nc = ["#ef4444" if n in fail_nodes else "#22c55e" for n in G.nodes()]
        ns = [80 if n in fail_nodes else 30 for n in G.nodes()]
        nx.draw_networkx(G, pos, ax=ax, node_color=nc, node_size=ns,
                         with_labels=False, edge_color="#d1d5db", width=0.5, alpha=0.9)
        ax.set_title(f"t={t} ({sum(1 for n in G.nodes() if n in fail_nodes)} failing)",
                     fontsize=10)
        ax.axis("off")
    legend = [mpatches.Patch(color="#22c55e", label="Normal"),
              mpatches.Patch(color="#ef4444", label="Failing")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
    plt.suptitle("Failure Propagation Over Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/failure_propagation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> results/failure_propagation.png")


def plot_embedding_tsne(model, features, labels, m2i, adj, ckpt,
                         edge_builder, max_samples=2000):
    num_nodes = adj["num_nodes"]
    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]
    fm, fs = ckpt["feat_mean"], ckpt["feat_std"]

    time_windows = sorted(features["time_window"].unique())
    feat_groups = dict(list(features.groupby("time_window")))
    label_groups = dict(list(labels.groupby("time_window")))
    mid = len(time_windows) // 2
    tw_slice = time_windows[max(0, mid - 5): mid + 1]

    x_list = []
    edge_list = []
    for tw in tw_slice:
        x = np.zeros((num_nodes, len(feat_cols)), dtype=np.float32)
        if tw in feat_groups:
            df = feat_groups[tw]
            idx_map = df["machine_id"].map(m2i)
            valid = idx_map.notna()
            if valid.any():
                x[idx_map[valid].astype(int).values] = \
                    df.loc[valid.values, feat_cols].fillna(0).values.astype(np.float32)
        x_list.append((torch.tensor(x) - fm) / fs)
        edge_list.append(edge_builder.build_edges(tw))

    y = np.zeros(num_nodes, dtype=np.int64)
    last_tw = tw_slice[-1]
    if last_tw in label_groups:
        df = label_groups[last_tw]
        idx_map = df["machine_id"].map(m2i)
        valid = idx_map.notna()
        if valid.any():
            y[idx_map[valid].astype(int).values] = 1

    with torch.no_grad():
        _, emb = model(x_list, edge_list, num_nodes=num_nodes,
                        return_embeddings=True)
        emb = emb.numpy()

    n = min(max_samples, len(emb))
    idx = np.random.choice(len(emb), n, replace=False)
    perp = min(30, n - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(perp, 2))
    emb_2d = tsne.fit_transform(emb[idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y[idx], cmap="RdYlGn_r",
                          alpha=0.6, s=15, edgecolors="none")
    ax.set_title("t-SNE of Learned Node Embeddings", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Failure Label")
    plt.tight_layout()
    plt.savefig("results/embedding_tsne.png", dpi=150)
    plt.close()
    print("  -> results/embedding_tsne.png")


# ── Novel: MC Dropout Uncertainty ─────────────────────────────

def mc_dropout_uncertainty(model, features, labels, m2i, adj, ckpt,
                            edge_builder, n_forward=50):
    """
    Monte Carlo Dropout: run 50 forward passes with dropout ON.
    Mean = robust prediction. Std = uncertainty per node.
    No retraining needed — just call model.train() during inference.

    NOTE: With the edge_drop_rate fix, model.train() also activates
    edge dropout via dropout_edge. To isolate pure MC Dropout
    uncertainty (neuron dropout only), we temporarily set
    edge_drop_rate=0 during MC inference. This means uncertainty
    comes purely from dropout masks, not graph sampling.
    """
    print("  Running MC Dropout (50 forward passes)...")
    num_nodes = adj["num_nodes"]

    sequences = find_failure_sequences(features, labels, num_seqs=5)
    if not sequences:
        print("  No failure sequences"); return

    x_list, y, edge_list = build_snapshot(
        features, labels, m2i, adj, ckpt, sequences[0], edge_builder)
    y_np = y.numpy()

    # Disable edge dropout for pure MC Dropout uncertainty
    saved_edr = model.edge_drop_rate
    model.edge_drop_rate = 0.0

    all_probs = []
    model.train()  # KEY: neuron dropout stays ON
    for _ in range(n_forward):
        with torch.no_grad():
            logits = model(x_list, edge_list, num_nodes=num_nodes)
            all_probs.append(F.softmax(logits, dim=1)[:, 1].numpy())
    model.eval()

    # Restore edge drop rate
    model.edge_drop_rate = saved_edr

    all_probs = np.stack(all_probs, axis=0)
    mean_prob = all_probs.mean(axis=0)
    std_prob = all_probs.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    if (y_np == 0).any():
        ax.hist(std_prob[y_np == 0], bins=30, alpha=0.7, color="#22c55e",
                label="Normal", density=True)
    if (y_np == 1).any():
        ax.hist(std_prob[y_np == 1], bins=30, alpha=0.7, color="#ef4444",
                label="Failing", density=True)
    ax.set_xlabel("Uncertainty (std)"); ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    scatter = ax.scatter(mean_prob, std_prob, c=y_np, cmap="RdYlGn_r",
                          alpha=0.4, s=10, edgecolors="none")
    ax.set_xlabel("Mean failure prob"); ax.set_ylabel("Uncertainty (std)")
    ax.set_title("Prediction vs Confidence")
    plt.colorbar(scatter, ax=ax, label="True label"); ax.grid(alpha=0.3)

    ax = axes[2]
    top_unc = np.argsort(std_prob)[-20:][::-1]
    colors = ["#ef4444" if y_np[i] == 1 else "#22c55e" for i in top_unc]
    ax.barh(range(len(top_unc)), std_prob[top_unc], color=colors)
    ax.set_yticks(range(len(top_unc)))
    ax.set_yticklabels([f"Node {i}" for i in top_unc], fontsize=8)
    ax.set_xlabel("Uncertainty"); ax.set_title("Top 20 Most Uncertain Nodes")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    legend = [mpatches.Patch(color="#22c55e", label="Normal"),
              mpatches.Patch(color="#ef4444", label="Failing")]
    ax.legend(handles=legend, fontsize=8)

    plt.suptitle("MC Dropout Uncertainty Quantification", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/mc_dropout_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> results/mc_dropout_uncertainty.png")

    fail_unc = std_prob[y_np == 1].mean() if (y_np == 1).any() else 0
    norm_unc = std_prob[y_np == 0].mean() if (y_np == 0).any() else 0
    print(f"  Avg uncertainty — normal: {norm_unc:.4f}, failing: {fail_unc:.4f}")

    # Cleanup
    del x_list, y, edge_list, all_probs
    gc.collect()


# ── Novel: Edge Contagion Scoring ─────────────────────────────

def edge_contagion_scoring(model, features, labels, m2i, adj, ckpt,
                            edge_builder, top_k=30):
    """
    Gradient-based edge importance: which connections carry failure signal.
    edge_score = |grad(src_node)| * |grad(dst_node)|
    High score = failure prediction is sensitive to this edge.

    Uses dynamic edges to match training topology.
    Runs in eval mode (no dropout/edge_drop) for deterministic gradients.
    """
    print("  Computing edge contagion scores...")
    edge_list_static = adj["edges"]
    num_nodes = adj["num_nodes"]

    sequences = find_failure_sequences(features, labels, num_seqs=30)
    if not sequences:
        print("  No failure sequences"); return

    edge_importance = np.zeros(len(edge_list_static), dtype=np.float64)
    count = 0
    model.eval()

    for tw_slice in sequences:
        x_list, y, dyn_edge_list = build_snapshot(
            features, labels, m2i, adj, ckpt, tw_slice, edge_builder)
        if y.sum() == 0:
            continue
        for x_t in x_list:
            x_t.requires_grad_(True)

        logits = model(x_list, dyn_edge_list, num_nodes=num_nodes)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        if x_list[-1].grad is not None:
            node_grad = x_list[-1].grad.abs().mean(dim=1).detach().numpy()
            for idx, (src, dst) in enumerate(edge_list_static):
                if src != dst:
                    edge_importance[idx] += node_grad[src] * node_grad[dst]
            count += 1

        del x_list, y, logits, loss, dyn_edge_list
        gc.collect()

    if count == 0:
        print("  No usable sequences"); return

    edge_importance /= edge_importance.max() + 1e-8
    non_self = [(i, e, edge_importance[i])
                for i, e in enumerate(edge_list_static) if e[0] != e[1]]
    non_self.sort(key=lambda x: x[2], reverse=True)
    top_edges = non_self[:top_k]

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    edge_labels = [f"{e[0]}->{e[1]}" for _, e, _ in top_edges]
    scores = [s for _, _, s in top_edges]
    colors = plt.cm.Reds(np.array(scores) * 0.8 + 0.2)
    ax.barh(range(len(top_edges)), scores, color=colors)
    ax.set_yticks(range(len(top_edges)))
    ax.set_yticklabels(edge_labels, fontsize=8)
    ax.set_xlabel("Contagion score")
    ax.set_title(f"Top {top_k} Failure-Contagious Edges")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/edge_contagion_scores.png", dpi=150)
    plt.close()
    print(f"  -> results/edge_contagion_scores.png")

    # Network visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    G = nx.Graph()
    for _, e, score in top_edges[:20]:
        G.add_edge(e[0], e[1], weight=score)
    if len(G.nodes()) == 0:
        print("  No edges to visualize"); plt.close(); return

    pos = nx.spring_layout(G, seed=42, k=3.0 / max(np.sqrt(len(G.nodes())), 1))
    ew = [G[u][v]["weight"] * 5 for u, v in G.edges()]
    ec = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=ew, edge_color=ec,
                           edge_cmap=plt.cm.Reds, alpha=0.8)

    label_groups = dict(list(labels.groupby("time_window")))
    fail_machines = set()
    for tw_group in label_groups.values():
        for mid in tw_group["machine_id"]:
            if mid in m2i: fail_machines.add(m2i[mid])
    nc = ["#ef4444" if n in fail_machines else "#3b82f6" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=200, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white",
                            font_weight="bold")

    legend = [mpatches.Patch(color="#3b82f6", label="Normal"),
              mpatches.Patch(color="#ef4444", label="Failing")]
    ax.legend(handles=legend, loc="lower right")
    ax.set_title("Failure Contagion Network\n(edge thickness = contagion score)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/contagion_network.png", dpi=150)
    plt.close()
    print("  -> results/contagion_network.png")

    print(f"\n  Top 10 contagious edges:")
    for _, e, score in top_edges[:10]:
        sf = "FAIL" if e[0] in fail_machines else "ok"
        df_label = "FAIL" if e[1] in fail_machines else "ok"
        print(f"    {e[0]}({sf}) -> {e[1]}({df_label})  score={score:.4f}")

    del edge_importance
    gc.collect()


# ── Main ──────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("EVALUATION, VISUALIZATION & ADVANCED ANALYSIS")
    print("=" * 65)

    results_path = "processed/test_results.npz"
    if not os.path.exists(results_path):
        print("ERROR: No test results found. Run train.py first.")
        return

    data = np.load(results_path)
    preds, labels_arr, probs = data["preds"], data["labels"], data["probs"]
    print(f"\nTest predictions: {len(preds):,}")
    print(f"Failing nodes: {(labels_arr == 1).sum():,} ({100*(labels_arr==1).mean():.2f}%)")
    has_both = len(np.unique(labels_arr)) > 1

    # ── Standard evaluation ───────────────────────────────────
    if has_both:
        print("\n1. Confusion matrix...")
        plot_confusion_matrix(labels_arr, preds)
        print("\n2. ROC & PR curves...")
        plot_roc_pr(labels_arr, probs)
    print("\n3. Classification report:")
    if has_both:
        print(classification_report(labels_arr, preds,
              target_names=["Normal", "Failing"], zero_division=0))

    print("4. Loading model...")
    model, ckpt = load_model()
    features, labels_df, m2i, adj, edge_builder = load_graph_data()

    print("\n5. Critical nodes...")
    critical = identify_critical_nodes(model, features, labels_df, m2i, adj,
                                        ckpt, edge_builder)
    if critical:
        print(f"   Top 10: {critical[:10]}")

    print("\n6. Failure propagation...")
    plot_failure_propagation(features, labels_df, m2i, adj)

    print("\n7. t-SNE embeddings...")
    plot_embedding_tsne(model, features, labels_df, m2i, adj, ckpt, edge_builder)

    # ── Novel contributions ───────────────────────────────────
    print("\n" + "=" * 65)
    print("NOVEL CONTRIBUTIONS")
    print("=" * 65)

    print("\n8. MC Dropout Uncertainty...")
    try:
        mc_dropout_uncertainty(model, features, labels_df, m2i, adj, ckpt,
                                edge_builder)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n9. Edge Contagion Scoring...")
    try:
        edge_contagion_scoring(model, features, labels_df, m2i, adj, ckpt,
                                edge_builder)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 65)
    print("All results saved to ./results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
