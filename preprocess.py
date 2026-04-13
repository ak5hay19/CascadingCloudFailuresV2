"""
STEP 1: Preprocess Borg Traces
================================
    python preprocess.py

Reads borg_traces_data.csv → cleans → engineers features →
builds static adjacency + per-window membership data → saves to ./processed/

Output files:
  processed/machine_features.parquet  — (machine_id, time_window, 13 features)
  processed/failure_labels.parquet    — (machine_id, time_window) pairs about to fail
  processed/adjacency.json            — static graph edges + node mapping (fallback)
  processed/window_membership.parquet — per-window cluster/collection membership (for dynamic edges)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    return {}


def find_data_file(filename):
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
    print(f"Loading {filename}...")
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean_data(df):
    print("Cleaning data...")
    for col in ["time", "priority", "instance_index", "start_time", "end_time",
                 "average_usage", "maximum_usage", "random_sample_usage",
                 "assigned_memory", "page_cache_memory",
                 "cycles_per_instruction", "memory_accesses_per_instruction",
                 "sample_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["machine_id", "collection_id", "alloc_collection_id", "cluster"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in ["instance_events_type", "collections_events_type"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "failed" in df.columns:
        df["failed"] = df["failed"].map(
            lambda x: 1 if str(x).strip().lower() in ("1", "true", "yes", "1.0") else 0
        )

    if "machine_id" in df.columns:
        before = len(df)
        df = df[df["machine_id"].notna() & (df["machine_id"] != "nan") & (df["machine_id"] != "")]
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with no machine_id")

    print(f"  Clean: {df.shape[0]:,} rows")
    return df


def add_time_windows(df, window_sec=300):
    print(f"Creating time windows ({window_sec}s each)...")
    if "time" in df.columns and df["time"].notna().any():
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
    print(f"Building failure labels (horizon={horizon})...")
    if "failed" not in df.columns:
        if "instance_events_type" in df.columns:
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


def build_static_adjacency(df):
    """Build static adjacency (used as fallback and for eval)."""
    print("Building static adjacency graph...")

    machines = df["machine_id"].unique()
    machine_to_idx = {m: i for i, m in enumerate(sorted(machines))}
    print(f"  {len(machine_to_idx)} nodes")

    edges = {}

    if "cluster" in df.columns:
        cluster_groups = (
            df[df["machine_id"].isin(machine_to_idx)]
            .groupby("cluster")["machine_id"].apply(set)
        )
        for _, m_set in cluster_groups.items():
            m_list = [machine_to_idx[m] for m in m_set if m in machine_to_idx]
            for i in range(len(m_list)):
                for j in range(i + 1, min(i + 20, len(m_list))):
                    edges[(m_list[i], m_list[j])] = 1.0
                    edges[(m_list[j], m_list[i])] = 1.0
        print(f"  Cluster edges: {len(edges)}")

    if "collection_id" in df.columns:
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
        print(f"  + Collection edges -> {len(edges)} total")

    for i in range(len(machine_to_idx)):
        edges[(i, i)] = 1.0

    edge_list = sorted(edges.keys())
    edge_weights = [edges[e] for e in edge_list]
    print(f"  Final: {len(edge_list)} edges (incl self-loops)")

    return machine_to_idx, [[e[0], e[1]] for e in edge_list], edge_weights


def build_window_membership(df):
    """
    Save per-window cluster/collection membership for dynamic edge building.

    Instead of building all per-window edge lists during preprocessing
    (which would be a huge file), we save the compact membership data
    and let the data loader build edges on-the-fly per window.

    This is ~20MB vs ~500MB+ for pre-built per-window edge lists.
    """
    print("Building per-window membership data for dynamic edges...")

    cols = ["machine_id", "time_window"]
    if "cluster" in df.columns:
        cols.append("cluster")
    if "collection_id" in df.columns:
        cols.append("collection_id")

    membership = (
        df[cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"  {len(membership):,} membership records")
    return membership


def main():
    config = load_config()
    data_cfg = config.get("data", {})

    filename = data_cfg.get("filename", "borg_traces_data.csv")
    tw_sec = data_cfg.get("time_window_sec", 300)
    horizon = data_cfg.get("prediction_horizon", 3)

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
    machine_to_idx, edge_list, edge_weights = build_static_adjacency(df)
    membership = build_window_membership(df)

    # Save
    os.makedirs("processed", exist_ok=True)

    features.to_parquet("processed/machine_features.parquet", index=False)
    labels.to_parquet("processed/failure_labels.parquet", index=False)
    membership.to_parquet("processed/window_membership.parquet", index=False)

    with open("processed/adjacency.json", "w") as f:
        json.dump({
            "machine_to_idx": {str(k): v for k, v in machine_to_idx.items()},
            "edges": edge_list,
            "edge_weights": edge_weights,
            "num_nodes": len(machine_to_idx),
        }, f)

    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]
    total = features[["machine_id", "time_window"]].drop_duplicates().shape[0]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Nodes (machines):    {len(machine_to_idx):,}")
    print(f"  Time windows:        {features['time_window'].nunique():,}")
    print(f"  Features per node:   {len(feat_cols)}")
    print(f"  Static edges:        {len(edge_list):,}")
    print(f"  Membership records:  {len(membership):,} (for dynamic edges)")
    print(f"  Positive labels:     {len(labels):,} / {total:,} ({100*len(labels)/max(total,1):.2f}%)")
    print(f"\n  Saved to ./processed/")
    print(f"  Next: python train.py")


if __name__ == "__main__":
    main()
