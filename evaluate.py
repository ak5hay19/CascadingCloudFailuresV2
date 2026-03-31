"""
STEP 3: Evaluate & Visualize
==============================
    python evaluate.py

Generates all plots in ./results/:
  - Confusion matrix
  - ROC & PR curves
  - Critical node identification (gradient-based)
  - Failure propagation visualization
  - t-SNE of learned node embeddings
"""

import os
import gc
import json
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


def load_model():
    ckpt = torch.load("best_model.pt", weights_only=False, map_location="cpu")
    model = SpatioTemporalGNN(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_gnn_layers=ckpt.get("num_gnn_layers", 2),
        dropout=ckpt.get("dropout", 0.3),
        num_neighbors=ckpt.get("num_neighbors", 15),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_graph_data():
    with open("processed/adjacency.json") as f:
        adj = json.load(f)
    features = pd.read_parquet("processed/machine_features.parquet")
    labels = pd.read_parquet("processed/failure_labels.parquet")
    features["machine_id"] = features["machine_id"].astype(str)
    labels["machine_id"] = labels["machine_id"].astype(str)
    m2i = {str(k): int(v) for k, v in adj["machine_to_idx"].items()}
    return features, labels, m2i, adj


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
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(rec, prec)
    ax2.plot(rec, prec, color="#dc2626", lw=2, label=f"AP = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.suptitle("ST-GNN Model Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> results/roc_pr_curves.png")
    return roc_auc, pr_auc


def identify_critical_nodes(model, features, labels, m2i, adj, ckpt, top_k=20):
    print("  Running gradient analysis...")

    edge_index = torch.tensor(adj["edges"], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        adj.get("edge_weights", [1.0] * len(adj["edges"])), dtype=torch.float
    )
    num_nodes = adj["num_nodes"]
    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]

    fm, fs = ckpt["feat_mean"], ckpt["feat_std"]

    time_windows = sorted(features["time_window"].unique())
    feat_groups = dict(list(features.groupby("time_window")))
    label_groups = dict(list(labels.groupby("time_window")))
    fail_windows = labels["time_window"].unique() if len(labels) > 0 else []

    node_importance = np.zeros(num_nodes, dtype=np.float64)
    count = 0

    for fw in list(fail_windows)[:30]:
        tw_idx = None
        for i, tw in enumerate(time_windows):
            if tw == fw:
                tw_idx = i
                break
        if tw_idx is None or tw_idx < 5:
            continue

        tw_slice = time_windows[tw_idx - 5: tw_idx + 1]

        x_list = []
        for tw in tw_slice:
            x = np.zeros((num_nodes, len(feat_cols)), dtype=np.float32)
            if tw in feat_groups:
                df = feat_groups[tw]
                idx_map = df["machine_id"].map(m2i)
                valid = idx_map.notna()
                if valid.any():
                    x[idx_map[valid].astype(int).values] = \
                        df.loc[valid.values, feat_cols].fillna(0).values.astype(np.float32)
            x_t = (torch.tensor(x) - fm) / fs
            x_t.requires_grad_(True)
            x_list.append(x_t)

        y = np.zeros(num_nodes, dtype=np.int64)
        last_tw = tw_slice[-1]
        if last_tw in label_groups:
            df = label_groups[last_tw]
            idx_map = df["machine_id"].map(m2i)
            valid = idx_map.notna()
            if valid.any():
                y[idx_map[valid].astype(int).values] = 1
        y = torch.tensor(y)

        if y.sum() == 0:
            continue

        logits = model(x_list, edge_index, edge_weight, num_nodes=num_nodes)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        if x_list[-1].grad is not None:
            grad_mag = x_list[-1].grad.abs().mean(dim=1).detach().numpy()
            node_importance += grad_mag
            count += 1

        del x_list, y, logits, loss
        gc.collect()

    if count == 0:
        print("  No failure samples for gradient analysis")
        return []

    node_importance /= node_importance.max() + 1e-8
    top_indices = np.argsort(node_importance)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(node_importance[top_indices] * 0.8 + 0.2)
    ax.barh(range(len(top_indices)), node_importance[top_indices], color=colors)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f"Machine {i}" for i in top_indices], fontsize=9)
    ax.set_xlabel("Importance Score (gradient magnitude)")
    ax.set_title(f"Top {top_k} Critical Nodes for Cascade Propagation")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/critical_nodes.png", dpi=150)
    plt.close()
    print(f"  -> results/critical_nodes.png")
    return top_indices.tolist()


def plot_failure_propagation(features, labels, m2i, adj, max_vis=120):
    if len(labels) == 0:
        print("  No failures to visualize")
        return

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
            interesting.add(e[0])
            interesting.add(e[1])
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

    T = len(tw_slice)
    cols = min(T, 6)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    for t, tw in enumerate(tw_slice[:cols]):
        ax = axes[t]
        fail_nodes = set()
        if tw in label_groups:
            for mid in label_groups[tw]["machine_id"]:
                if mid in m2i:
                    fail_nodes.add(m2i[mid])

        node_colors = ["#ef4444" if n in fail_nodes else "#22c55e" for n in G.nodes()]
        node_sizes = [80 if n in fail_nodes else 30 for n in G.nodes()]

        nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                         with_labels=False, edge_color="#d1d5db", width=0.5, alpha=0.9)
        n_fail = sum(1 for n in G.nodes() if n in fail_nodes)
        ax.set_title(f"t={t} ({n_fail} failing)", fontsize=10)
        ax.axis("off")

    legend = [
        mpatches.Patch(color="#22c55e", label="Normal"),
        mpatches.Patch(color="#ef4444", label="Failing"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
    plt.suptitle("Failure Propagation Over Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/failure_propagation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> results/failure_propagation.png")


def plot_embedding_tsne(model, features, labels, m2i, adj, ckpt, max_samples=2000):
    edge_index = torch.tensor(adj["edges"], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        adj.get("edge_weights", [1.0] * len(adj["edges"])), dtype=torch.float
    )
    num_nodes = adj["num_nodes"]
    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]
    fm, fs = ckpt["feat_mean"], ckpt["feat_std"]

    time_windows = sorted(features["time_window"].unique())
    feat_groups = dict(list(features.groupby("time_window")))
    label_groups = dict(list(labels.groupby("time_window")))

    mid = len(time_windows) // 2
    tw_slice = time_windows[max(0, mid - 5): mid + 1]

    x_list = []
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

    y = np.zeros(num_nodes, dtype=np.int64)
    last_tw = tw_slice[-1]
    if last_tw in label_groups:
        df = label_groups[last_tw]
        idx_map = df["machine_id"].map(m2i)
        valid = idx_map.notna()
        if valid.any():
            y[idx_map[valid].astype(int).values] = 1

    with torch.no_grad():
        _, emb = model(x_list, edge_index, edge_weight,
                        num_nodes=num_nodes, return_embeddings=True)
        emb = emb.numpy()

    n = min(max_samples, len(emb))
    idx = np.random.choice(len(emb), n, replace=False)
    emb_sub, y_sub = emb[idx], y[idx]

    perp = min(30, n - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(perp, 2))
    emb_2d = tsne.fit_transform(emb_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_sub, cmap="RdYlGn_r",
                          alpha=0.6, s=15, edgecolors="none")
    ax.set_title("t-SNE of Learned Node Embeddings", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Failure Label")
    plt.tight_layout()
    plt.savefig("results/embedding_tsne.png", dpi=150)
    plt.close()
    print("  -> results/embedding_tsne.png")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EVALUATION & VISUALIZATION")
    print("=" * 60)

    results_path = "processed/test_results.npz"
    if not os.path.exists(results_path):
        print("ERROR: No test results found. Run train.py first.")
        return

    data = np.load(results_path)
    preds, labels_arr, probs = data["preds"], data["labels"], data["probs"]

    print(f"\nTest predictions: {len(preds):,}")
    print(f"Failing nodes: {(labels_arr == 1).sum():,} ({100*(labels_arr==1).mean():.2f}%)")

    has_both = len(np.unique(labels_arr)) > 1

    if has_both:
        print("\n1. Confusion matrix...")
        plot_confusion_matrix(labels_arr, preds)

        print("\n2. ROC & PR curves...")
        plot_roc_pr(labels_arr, probs)

    print("\n3. Classification report:")
    if has_both:
        print(classification_report(labels_arr, preds,
              target_names=["Normal", "Failing"], zero_division=0))

    print("4. Loading model for analysis...")
    model, ckpt = load_model()
    features, labels_df, m2i, adj = load_graph_data()

    print("\n5. Critical nodes...")
    critical = identify_critical_nodes(model, features, labels_df, m2i, adj, ckpt)
    if critical:
        print(f"   Top 10: {critical[:10]}")

    print("\n6. Failure propagation...")
    plot_failure_propagation(features, labels_df, m2i, adj)

    print("\n7. t-SNE embeddings...")
    plot_embedding_tsne(model, features, labels_df, m2i, adj, ckpt)

    print("\n" + "=" * 60)
    print("All results saved to ./results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
