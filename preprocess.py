"""
STEP 1: Preprocess Borg Traces
================================
    python preprocess.py

Reads borg_traces_data.csv → cleans → engineers features →
builds adjacency → saves to ./processed/

Output files:
  processed/machine_features.parquet  — (machine_id, time_window, 12 features)
  processed/failure_labels.parquet    — (machine_id, time_window) pairs about to fail
  processed/adjacency.json            — graph edges + node mapping
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config():
    """Load config.yaml or use defaults."""
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    return {}


def find_data_file(filename):
    """Try to locate the dataset file."""
    if os.path.exists(filename):
        return filename
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    for ext in [".csv", ".xlsx", ".xls", ""]:
        path = base + ext
        if os.path.exists(path):
            return path
    print(f"ERROR: Cannot find '{filename}'")
    print("Files in current directory:", os.listdir("."))
    return None


def load_data(filename):
    """Load dataset from CSV or Excel."""
    print(f"Loading {filename}...")
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, low_memory=False)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df):
    """Type conversions and basic cleaning."""
    print("Cleaning data...")

    # Numeric columns
    for col in ["time", "priority", "instance_index", "start_time", "end_time",
                 "average_usage", "maximum_usage", "random_sample_usage",
                 "assigned_memory", "page_cache_memory",
                 "cycles_per_instruction", "memory_accesses_per_instruction",
                 "sample_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ID columns → string
    for col in ["machine_id", "collection_id", "alloc_collection_id", "cluster"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Event type columns
    for col in ["instance_events_type", "collections_events_type"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Failed → binary
    if "failed" in df.columns:
        df["failed"] = df["failed"].map(
            lambda x: 1 if str(x).strip().lower() in ("1", "true", "yes", "1.0") else 0
        )

    # Drop rows with no machine_id
    if "machine_id" in df.columns:
        before = len(df)
        df = df[df["machine_id"].notna() & (df["machine_id"] != "nan") & (df["machine_id"] != "")]
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with no machine_id")

    print(f"  Clean: {df.shape[0]:,} rows")
    return df


def add_time_windows(df, window_sec=300):
    """Bucket timestamps into discrete time windows."""
    print(f"Creating time windows ({window_sec}s each)...")

    if "time" in df.columns and df["time"].notna().any():
        # Borg timestamps are in microseconds
        df["time_window"] = (df["time"] // (window_sec * 1_000_000)).astype("Int64")
    elif "start_time" in df.columns and df["start_time"].notna().any():
        df["time_window"] = (df["start_time"] // (window_sec * 1_000_000)).astype("Int64")
    else:
        print("  WARNING: No timestamp column found, using row index")
        df["time_window"] = (df.index // 1000)

    df = df.dropna(subset=["time_window"])
    df["time_window"] = df["time_window"].astype(np.int64)
    print(f"  {df['time_window'].nunique()} unique time windows")
    return df


def build_features(df):
    """
    Aggregate per (machine_id, time_window) → 12 node features.

    Features capture:
      - CPU load profile (mean, variability, peak)
      - Memory load profile (mean, variability, peak)
      - Total assigned memory
      - Task count (workload intensity)
      - Failure count and failure rate
      - Average task priority
      - Scheduling class diversity (workload heterogeneity)
      - Event count (scheduling churn)
    """
    print("Building machine-level features...")

    agg = {}
    if "average_usage" in df.columns:
        agg["average_usage"] = ["mean", "std", "max"]
    if "maximum_usage" in df.columns:
        agg["maximum_usage"] = ["mean", "std", "max"]
    if "assigned_memory" in df.columns:
        agg["assigned_memory"] = ["sum"]
    if "instance_index" in df.columns:
        agg["instance_index"] = ["count"]
    if "failed" in df.columns:
        agg["failed"] = ["sum", "mean"]
    if "priority" in df.columns:
        agg["priority"] = ["mean"]
    if "scheduling_class" in df.columns:
        agg["scheduling_class"] = ["nunique"]
    if "instance_events_type" in df.columns:
        agg["instance_events_type"] = ["count"]

    if not agg:
        raise ValueError("No usable columns for aggregation!")

    grouped = df.groupby(["machine_id", "time_window"]).agg(agg)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    grouped = grouped.reset_index().fillna(0)

    feat_cols = [c for c in grouped.columns if c not in ("machine_id", "time_window")]
    print(f"  {grouped.shape[0]:,} (machine, window) pairs")
    print(f"  {len(feat_cols)} features: {feat_cols}")
    return grouped


def build_labels(df, horizon=3):
    """
    Label machines as 'about to fail' if they fail within the
    next `horizon` time windows. This gives the model a prediction
    target that's ahead of the actual failure.
    """
    print(f"Building failure labels (horizon={horizon})...")

    if "failed" not in df.columns:
        if "instance_events_type" in df.columns:
            # Types 5=FAIL, 7=KILL, 8=LOST
            df["failed"] = df["instance_events_type"].isin([5, 7, 8]).astype(int)
        else:
            print("  ERROR: Cannot determine failures!")
            return pd.DataFrame(columns=["machine_id", "time_window", "label"])

    failures = (
        df[df["failed"] == 1]
        .groupby(["machine_id", "time_window"])
        .size()
        .reset_index(name="count")
    )

    if len(failures) == 0:
        print("  No failures found")
        return pd.DataFrame(columns=["machine_id", "time_window", "label"])

    # For each failure at time t, label windows t-1 through t-horizon
    frames = []
    for offset in range(1, horizon + 1):
        tmp = failures[["machine_id", "time_window"]].copy()
        tmp["time_window"] = tmp["time_window"] - offset
        tmp["label"] = 1
        frames.append(tmp)

    labels = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["machine_id", "time_window"]
    )
    print(f"  {len(labels):,} positive labels")
    return labels


def build_adjacency(df, max_nodes=3000):
    """
    Build machine graph edges from two types of relationships:
      1. Cluster co-location (physical proximity)
      2. Shared collections (workload dependency)

    Also computes edge weights:
      - Cluster edges get weight 1.0
      - Collection edges get weight 0.5
    This lets the GNN learn that physical co-location matters
    more than just sharing a job collection.
    """
    print("Building adjacency graph...")

    machines = df["machine_id"].unique()
    if max_nodes and len(machines) > max_nodes:
        print(f"  Subsampling {len(machines)} → {max_nodes} machines (by activity)")
        top = df["machine_id"].value_counts().head(max_nodes).index
        machines = top.values

    machine_to_idx = {m: i for i, m in enumerate(sorted(machines))}
    print(f"  {len(machine_to_idx)} nodes")

    edges = {}  # (src, dst) → weight

    # Cluster edges (weight=1.0)
    if "cluster" in df.columns:
        cluster_groups = (
            df[df["machine_id"].isin(machine_to_idx)]
            .groupby("cluster")["machine_id"].apply(set)
        )
        for _, m_set in cluster_groups.items():
            m_list = [machine_to_idx[m] for m in m_set if m in machine_to_idx]
            # Cap connections per cluster to avoid quadratic blowup
            for i in range(len(m_list)):
                for j in range(i + 1, min(i + 20, len(m_list))):
                    edges[(m_list[i], m_list[j])] = 1.0
                    edges[(m_list[j], m_list[i])] = 1.0
        print(f"  Cluster edges: {len(edges)}")

    # Collection edges (weight=0.5) — only if we need more connectivity
    if "collection_id" in df.columns and len(edges) < 2000:
        coll_groups = (
            df[df["machine_id"].isin(machine_to_idx)]
            .groupby("collection_id")["machine_id"].apply(set)
        )
        for _, m_set in coll_groups.items():
            m_list = [machine_to_idx[m] for m in m_set if m in machine_to_idx]
            for i in range(len(m_list)):
                for j in range(i + 1, min(i + 10, len(m_list))):
                    key = (m_list[i], m_list[j])
                    if key not in edges:
                        edges[key] = 0.5
                    key_r = (m_list[j], m_list[i])
                    if key_r not in edges:
                        edges[key_r] = 0.5
        print(f"  + Collection edges → {len(edges)} total")

    # Self-loops (weight=1.0)
    for i in range(len(machine_to_idx)):
        edges[(i, i)] = 1.0

    edge_list = sorted(edges.keys())
    edge_weights = [edges[e] for e in edge_list]

    print(f"  Final: {len(edge_list)} edges (incl self-loops)")

    return machine_to_idx, [[e[0], e[1]] for e in edge_list], edge_weights


def main():
    config = load_config()
    data_cfg = config.get("data", {})

    filename = data_cfg.get("filename", "borg_traces_data.csv")
    tw_sec = data_cfg.get("time_window_sec", 300)
    horizon = data_cfg.get("prediction_horizon", 3)
    max_nodes = data_cfg.get("max_nodes", 3000)

    path = find_data_file(filename)
    if path is None:
        return

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data(path)
    df = clean_data(df)
    df = add_time_windows(df, tw_sec)

    features = build_features(df)
    labels = build_labels(df, horizon)
    machine_to_idx, edge_list, edge_weights = build_adjacency(df, max_nodes)

    # Save
    os.makedirs("processed", exist_ok=True)

    features.to_parquet("processed/machine_features.parquet", index=False)
    labels.to_parquet("processed/failure_labels.parquet", index=False)

    with open("processed/adjacency.json", "w") as f:
        json.dump({
            "machine_to_idx": {str(k): v for k, v in machine_to_idx.items()},
            "edges": edge_list,
            "edge_weights": edge_weights,
            "num_nodes": len(machine_to_idx),
        }, f)

    # Summary
    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]
    total = features[["machine_id", "time_window"]].drop_duplicates().shape[0]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Nodes (machines):    {len(machine_to_idx):,}")
    print(f"  Time windows:        {features['time_window'].nunique():,}")
    print(f"  Features per node:   {len(feat_cols)}")
    print(f"  Edges:               {len(edge_list):,}")
    print(f"  Positive labels:     {len(labels):,} / {total:,} ({100*len(labels)/max(total,1):.2f}%)")
    print(f"\n✓ Saved to ./processed/")
    print(f"  Next: python train.py")


if __name__ == "__main__":
    main()
