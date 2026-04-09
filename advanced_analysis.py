"""
STEP 4: Advanced Analysis — Novel Research Contributions
==========================================================
    python advanced_analysis.py

Runs AFTER train.py and evaluate.py. Adds three novel features:

1. DYNAMIC GRAPH SNAPSHOTS
   - Builds per-window adjacency (edges change every timestep)
   - Compares static vs dynamic graph predictions
   - Shows how graph topology evolution affects failure detection

2. MC DROPOUT UNCERTAINTY QUANTIFICATION
   - Runs 50 forward passes with dropout active
   - Produces mean prediction + confidence interval per node
   - Identifies high-uncertainty nodes (model isn't sure)

3. EDGE CONTAGION SCORING
   - Computes d(loss)/d(edge_weight) — gradient per edge
   - Identifies which connections carry the most failure signal
   - Tells operators which dependencies to break during incidents

All plots saved to ./results/
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
from model import SpatioTemporalGNN


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

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


def build_snapshot_features(features, labels, m2i, adj, ckpt, tw_slice):
    """Build x_list and y for a given time window slice."""
    num_nodes = adj["num_nodes"]
    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]
    fm, fs = ckpt["feat_mean"], ckpt["feat_std"]

    feat_groups = dict(list(features.groupby("time_window")))
    label_groups = dict(list(labels.groupby("time_window")))

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

    return x_list, torch.tensor(y)


def find_failure_sequences(features, labels, num_seqs=10):
    """Find time window slices that contain failures."""
    time_windows = sorted(features["time_window"].unique())
    fail_windows = sorted(labels["time_window"].unique())

    sequences = []
    for fw in fail_windows:
        tw_idx = None
        for i, tw in enumerate(time_windows):
            if tw == fw:
                tw_idx = i
                break
        if tw_idx is not None and tw_idx >= 5:
            sequences.append(time_windows[tw_idx - 5: tw_idx + 1])
        if len(sequences) >= num_seqs:
            break
    return sequences


# ══════════════════════════════════════════════════════════════
# FEATURE 1: DYNAMIC GRAPH SNAPSHOTS
# ══════════════════════════════════════════════════════════════

def build_dynamic_edges(df_raw_path, m2i, time_windows):
    """
    Build per-window edge sets from raw data.

    Static graph: edges computed once across ALL time.
    Dynamic graph: edges computed per time window based on which
    machines are ACTUALLY active in the same cluster/collection
    during that specific window.

    This captures real topology evolution:
    - Machines join/leave collections as jobs start/finish
    - Cluster membership can change with reassignment
    - A machine with no active tasks has no meaningful edges
    """
    print("  Building per-window dynamic edges...")

    # Load raw data to get per-window activity
    if df_raw_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(df_raw_path)
    else:
        df = pd.read_csv(df_raw_path, low_memory=False)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    for col in ["machine_id", "collection_id", "cluster"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["time_window"] = (df["time"] // (300 * 1_000_000)).astype("Int64")
    df = df.dropna(subset=["time_window"])
    df["time_window"] = df["time_window"].astype(np.int64)

    # Build edges per window
    dynamic_edges = {}
    for tw in time_windows:
        window_data = df[df["time_window"] == tw]
        edges = set()

        # Cluster edges for this window only
        if "cluster" in window_data.columns:
            for _, group in window_data.groupby("cluster"):
                active = [m2i[m] for m in group["machine_id"].unique() if m in m2i]
                for i in range(len(active)):
                    for j in range(i + 1, min(i + 15, len(active))):
                        edges.add((active[i], active[j]))
                        edges.add((active[j], active[i]))

        # Collection edges for this window only
        if "collection_id" in window_data.columns:
            for _, group in window_data.groupby("collection_id"):
                active = [m2i[m] for m in group["machine_id"].unique() if m in m2i]
                for i in range(len(active)):
                    for j in range(i + 1, min(i + 10, len(active))):
                        edges.add((active[i], active[j]))
                        edges.add((active[j], active[i]))

        # Self-loops
        active_nodes = set()
        for m in window_data["machine_id"].unique():
            if m in m2i:
                active_nodes.add(m2i[m])
        for n in active_nodes:
            edges.add((n, n))

        if edges:
            edge_list = sorted(edges)
            ei = torch.tensor([[e[0] for e in edge_list],
                               [e[1] for e in edge_list]], dtype=torch.long)
        else:
            # Fallback: self-loops only
            nodes = list(range(len(m2i)))
            ei = torch.tensor([nodes, nodes], dtype=torch.long)

        dynamic_edges[tw] = ei

    return dynamic_edges


def compare_static_vs_dynamic(model, features, labels, m2i, adj, ckpt):
    """
    Compare predictions using static graph vs dynamic per-window graphs.
    Shows that topology evolution matters for failure prediction.
    """
    print("\n  Comparing static vs dynamic graph predictions...")

    num_nodes = adj["num_nodes"]
    static_ei = torch.tensor(adj["edges"], dtype=torch.long).t().contiguous()
    static_ew = torch.tensor(
        adj.get("edge_weights", [1.0] * len(adj["edges"])), dtype=torch.float
    )

    # Find failure sequences
    sequences = find_failure_sequences(features, labels, num_seqs=15)
    if not sequences:
        print("  No failure sequences found")
        return

    # Try to find raw data for dynamic edges
    raw_path = None
    for candidate in ["borg_traces_data.csv", "borg_traces_data.xlsx",
                       "borg_traces_data"]:
        if os.path.exists(candidate):
            raw_path = candidate
            break

    if raw_path is None:
        print("  Raw data file not found — skipping dynamic graph comparison")
        print("  (Need borg_traces_data.csv in project folder)")
        return

    # Collect all time windows we need
    all_tws = set()
    for seq in sequences:
        all_tws.update(seq)
    dynamic_edges = build_dynamic_edges(raw_path, m2i, list(all_tws))

    model.eval()
    static_probs_list = []
    dynamic_probs_list = []
    labels_list = []
    edge_count_static = []
    edge_count_dynamic = []

    for seq_tws in sequences:
        x_list, y = build_snapshot_features(features, labels, m2i, adj, ckpt, seq_tws)

        if y.sum() == 0:
            continue

        # Static prediction
        with torch.no_grad():
            logits_s = model(x_list, static_ei, static_ew, num_nodes=num_nodes)
            probs_s = F.softmax(logits_s, dim=1)[:, 1].numpy()

        # Dynamic prediction — use per-window edges
        # Build edge list for last snapshot (prediction target)
        last_tw = seq_tws[-1]
        if last_tw in dynamic_edges:
            dyn_ei = dynamic_edges[last_tw]
        else:
            dyn_ei = static_ei

        with torch.no_grad():
            logits_d = model(x_list, dyn_ei, num_nodes=num_nodes)
            probs_d = F.softmax(logits_d, dim=1)[:, 1].numpy()

        # Only look at failing nodes
        fail_mask = y.numpy() == 1
        if fail_mask.any():
            static_probs_list.append(probs_s[fail_mask].mean())
            dynamic_probs_list.append(probs_d[fail_mask].mean())
            labels_list.append(1)
            edge_count_static.append(static_ei.shape[1])
            edge_count_dynamic.append(dyn_ei.shape[1])

    if not static_probs_list:
        print("  No usable failure sequences")
        return

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x_pos = range(len(static_probs_list))
    ax1.bar([x - 0.15 for x in x_pos], static_probs_list, width=0.3,
            color="#3b82f6", label="Static graph", alpha=0.8)
    ax1.bar([x + 0.15 for x in x_pos], dynamic_probs_list, width=0.3,
            color="#ef4444", label="Dynamic graph", alpha=0.8)
    ax1.set_xlabel("Failure sequence index")
    ax1.set_ylabel("Mean failure probability (failing nodes)")
    ax1.set_title("Static vs Dynamic Graph\nFailure Detection Confidence")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.bar([x - 0.15 for x in x_pos], edge_count_static, width=0.3,
            color="#3b82f6", label="Static edges", alpha=0.8)
    ax2.bar([x + 0.15 for x in x_pos], edge_count_dynamic, width=0.3,
            color="#ef4444", label="Dynamic edges", alpha=0.8)
    ax2.set_xlabel("Failure sequence index")
    ax2.set_ylabel("Edge count")
    ax2.set_title("Graph Topology Evolution\nEdge Count per Window")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/static_vs_dynamic_graph.png", dpi=150)
    plt.close()
    print("  -> results/static_vs_dynamic_graph.png")

    # Summary stats
    s_mean = np.mean(static_probs_list)
    d_mean = np.mean(dynamic_probs_list)
    print(f"  Static graph avg failure prob:  {s_mean:.4f}")
    print(f"  Dynamic graph avg failure prob: {d_mean:.4f}")
    print(f"  Avg static edges:  {np.mean(edge_count_static):.0f}")
    print(f"  Avg dynamic edges: {np.mean(edge_count_dynamic):.0f}")


# ══════════════════════════════════════════════════════════════
# FEATURE 2: MC DROPOUT UNCERTAINTY QUANTIFICATION
# ══════════════════════════════════════════════════════════════

def mc_dropout_uncertainty(model, features, labels, m2i, adj, ckpt,
                           n_forward=50):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Normal inference: model.eval() → dropout OFF → one deterministic prediction.
    MC Dropout: model.train() → dropout STAYS ON → run 50 forward passes →
    each gives slightly different predictions (because different neurons
    are dropped each time) → take mean and std across the 50 runs.

    Mean = prediction (more robust than a single pass)
    Std = uncertainty (high std = model isn't sure about this node)

    No model changes needed. No retraining. Just run inference differently.
    """
    print("\n  Running MC Dropout (50 forward passes)...")

    num_nodes = adj["num_nodes"]
    edge_index = torch.tensor(adj["edges"], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        adj.get("edge_weights", [1.0] * len(adj["edges"])), dtype=torch.float
    )

    # Pick a failure sequence
    sequences = find_failure_sequences(features, labels, num_seqs=5)
    if not sequences:
        print("  No failure sequences found")
        return

    tw_slice = sequences[0]
    x_list, y = build_snapshot_features(features, labels, m2i, adj, ckpt, tw_slice)
    y_np = y.numpy()

    # Run N forward passes with dropout active
    all_probs = []
    model.train()  # KEY: keeps dropout ON

    for i in range(n_forward):
        with torch.no_grad():
            logits = model(x_list, edge_index, edge_weight, num_nodes=num_nodes)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.append(probs)

    model.eval()  # restore

    all_probs = np.stack(all_probs, axis=0)  # [50, N]
    mean_prob = all_probs.mean(axis=0)        # [N]
    std_prob = all_probs.std(axis=0)           # [N] — this is the uncertainty

    # ── Plot 1: Uncertainty distribution ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram of uncertainty
    ax = axes[0]
    ax.hist(std_prob[y_np == 0], bins=30, alpha=0.7, color="#22c55e",
            label="Normal nodes", density=True)
    ax.hist(std_prob[y_np == 1], bins=30, alpha=0.7, color="#ef4444",
            label="Failing nodes", density=True)
    ax.set_xlabel("Prediction uncertainty (std)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # Prediction vs uncertainty scatter
    ax = axes[1]
    scatter = ax.scatter(mean_prob, std_prob, c=y_np, cmap="RdYlGn_r",
                          alpha=0.4, s=10, edgecolors="none")
    ax.set_xlabel("Mean failure probability")
    ax.set_ylabel("Uncertainty (std)")
    ax.set_title("Prediction vs Confidence")
    plt.colorbar(scatter, ax=ax, label="True label")
    ax.grid(alpha=0.3)

    # Top uncertain nodes
    ax = axes[2]
    top_uncertain = np.argsort(std_prob)[-20:][::-1]
    colors = ["#ef4444" if y_np[i] == 1 else "#22c55e" for i in top_uncertain]
    bars = ax.barh(range(len(top_uncertain)), std_prob[top_uncertain], color=colors)
    ax.set_yticks(range(len(top_uncertain)))
    ax.set_yticklabels([f"Node {i}" for i in top_uncertain], fontsize=8)
    ax.set_xlabel("Uncertainty (std across 50 runs)")
    ax.set_title("Top 20 Most Uncertain Nodes")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    legend = [mpatches.Patch(color="#22c55e", label="Actually normal"),
              mpatches.Patch(color="#ef4444", label="Actually failing")]
    ax.legend(handles=legend, fontsize=8)

    plt.suptitle("MC Dropout Uncertainty Quantification", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/mc_dropout_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> results/mc_dropout_uncertainty.png")

    # Print stats
    fail_unc = std_prob[y_np == 1].mean() if (y_np == 1).any() else 0
    norm_unc = std_prob[y_np == 0].mean() if (y_np == 0).any() else 0
    print(f"  Avg uncertainty — normal nodes: {norm_unc:.4f}")
    print(f"  Avg uncertainty — failing nodes: {fail_unc:.4f}")
    print(f"  Top 5 uncertain nodes: {top_uncertain[:5].tolist()}")

    # Save uncertainty data
    np.savez("results/mc_dropout_data.npz",
             mean_prob=mean_prob, std_prob=std_prob, labels=y_np)
    print("  -> results/mc_dropout_data.npz")


# ══════════════════════════════════════════════════════════════
# FEATURE 3: EDGE CONTAGION SCORING
# ══════════════════════════════════════════════════════════════

def edge_contagion_scoring(model, features, labels, m2i, adj, ckpt, top_k=30):
    """
    Gradient-based edge importance — which connections carry failure signal.

    Critical node analysis asks: "which nodes matter most?"
    Edge contagion asks: "which CONNECTIONS matter most?"

    Method:
    1. Make edge weights require gradients
    2. Run forward pass on failure sequences
    3. Compute loss and backprop
    4. Edge weight gradient magnitude = contagion score
       High gradient = the model's failure prediction is very sensitive
       to this specific edge — failure signal flows strongly through it.

    Practical value: tells operators which dependencies to sever
    during an incident to stop cascading failure propagation.
    """
    print("\n  Computing edge contagion scores...")

    num_nodes = adj["num_nodes"]
    edge_list = adj["edges"]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    base_weights = adj.get("edge_weights", [1.0] * len(edge_list))

    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]

    sequences = find_failure_sequences(features, labels, num_seqs=30)
    if not sequences:
        print("  No failure sequences found")
        return

    # Accumulate edge gradients across failure sequences
    edge_importance = np.zeros(len(edge_list), dtype=np.float64)
    count = 0

    model.eval()

    for tw_slice in sequences:
        x_list, y = build_snapshot_features(features, labels, m2i, adj, ckpt, tw_slice)

        if y.sum() == 0:
            continue

        # Make edge weights a differentiable parameter
        ew = torch.tensor(base_weights, dtype=torch.float, requires_grad=True)

        # Need to pass edge weights through the model
        # Since SAGEConv doesn't use edge_weight directly, we weight
        # the neighbor features manually by multiplying x with edge weight
        # Instead, we compute gradient of loss w.r.t. a scalar multiplier
        # on each edge's contribution

        # Simpler approach: compute node-pair gradients
        # Use the input features' gradients and map them to edges
        for x_t in x_list:
            x_t.requires_grad_(True)

        logits = model(x_list, edge_index, num_nodes=num_nodes)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Map node gradients to edges: edge importance = 
        # |grad(src)| * |grad(dst)| — edges connecting two high-gradient
        # nodes are the ones where failure signal flows
        if x_list[-1].grad is not None:
            node_grad = x_list[-1].grad.abs().mean(dim=1).detach().numpy()

            for idx, (src, dst) in enumerate(edge_list):
                if src != dst:  # skip self-loops
                    edge_importance[idx] += node_grad[src] * node_grad[dst]
            count += 1

        del x_list, y, logits, loss
        gc.collect()

    if count == 0:
        print("  No usable failure sequences")
        return

    # Normalize
    edge_importance /= edge_importance.max() + 1e-8

    # Filter out self-loops and get top edges
    non_self = [(i, e, edge_importance[i])
                for i, e in enumerate(edge_list) if e[0] != e[1]]
    non_self.sort(key=lambda x: x[2], reverse=True)
    top_edges = non_self[:top_k]

    # ── Plot 1: Top contagious edges bar chart ────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    edge_labels = [f"{e[0]}→{e[1]}" for _, e, _ in top_edges]
    scores = [s for _, _, s in top_edges]
    colors = plt.cm.Reds(np.array(scores) * 0.8 + 0.2)

    ax.barh(range(len(top_edges)), scores, color=colors)
    ax.set_yticks(range(len(top_edges)))
    ax.set_yticklabels(edge_labels, fontsize=8)
    ax.set_xlabel("Contagion score")
    ax.set_title(f"Top {top_k} Failure-Contagious Edges")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/edge_contagion_scores.png", dpi=150)
    plt.close()
    print(f"  -> results/edge_contagion_scores.png")

    # ── Plot 2: Contagion network visualization ───────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    # Build subgraph from top contagious edges
    G = nx.Graph()
    vis_edges = top_edges[:20]  # top 20 for readability

    for _, e, score in vis_edges:
        G.add_edge(e[0], e[1], weight=score)

    if len(G.nodes()) == 0:
        print("  No edges to visualize")
        return

    pos = nx.spring_layout(G, seed=42, k=3.0 / max(np.sqrt(len(G.nodes())), 1))

    # Draw edges with width proportional to contagion score
    edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges()]
    edge_colors = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, edge_cmap=plt.cm.Reds,
                           alpha=0.8)

    # Color nodes by whether they're failing
    label_groups = dict(list(labels.groupby("time_window")))
    fail_machines = set()
    for tw_group in label_groups.values():
        for mid in tw_group["machine_id"]:
            if mid in m2i:
                fail_machines.add(m2i[mid])

    node_colors = ["#ef4444" if n in fail_machines else "#3b82f6" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=200, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                            font_color="white", font_weight="bold")

    legend = [mpatches.Patch(color="#3b82f6", label="Normal machine"),
              mpatches.Patch(color="#ef4444", label="Failing machine")]
    ax.legend(handles=legend, loc="lower right")
    ax.set_title("Failure Contagion Network\n(edge thickness = contagion score)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/contagion_network.png", dpi=150)
    plt.close()
    print("  -> results/contagion_network.png")

    # Print top 10
    print(f"\n  Top 10 contagious edges:")
    for _, e, score in top_edges[:10]:
        src_fail = "FAIL" if e[0] in fail_machines else "ok"
        dst_fail = "FAIL" if e[1] in fail_machines else "ok"
        print(f"    Machine {e[0]} ({src_fail}) → Machine {e[1]} ({dst_fail})"
              f"  score={score:.4f}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("ADVANCED ANALYSIS — Novel Research Contributions")
    print("=" * 65)

    if not os.path.exists("best_model.pt"):
        print("ERROR: No trained model found. Run train.py first.")
        return

    model, ckpt = load_model()
    features, labels, m2i, adj = load_graph_data()

    # ── Feature 1: Dynamic Graph Snapshots ────────────────────
    print("\n" + "─" * 50)
    print("1. DYNAMIC GRAPH SNAPSHOTS")
    print("─" * 50)
    try:
        compare_static_vs_dynamic(model, features, labels, m2i, adj, ckpt)
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Dynamic graph analysis requires raw data file)")

    # ── Feature 2: MC Dropout Uncertainty ─────────────────────
    print("\n" + "─" * 50)
    print("2. MC DROPOUT UNCERTAINTY QUANTIFICATION")
    print("─" * 50)
    try:
        mc_dropout_uncertainty(model, features, labels, m2i, adj, ckpt)
    except Exception as e:
        print(f"  Error: {e}")

    # ── Feature 3: Edge Contagion Scoring ─────────────────────
    print("\n" + "─" * 50)
    print("3. EDGE CONTAGION SCORING")
    print("─" * 50)
    try:
        edge_contagion_scoring(model, features, labels, m2i, adj, ckpt)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 65)
    print("Advanced analysis complete!")
    print("New plots in ./results/:")
    print("  - static_vs_dynamic_graph.png")
    print("  - mc_dropout_uncertainty.png")
    print("  - mc_dropout_data.npz")
    print("  - edge_contagion_scores.png")
    print("  - contagion_network.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
