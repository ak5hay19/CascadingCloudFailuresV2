"""
STEP 1: Preprocess Borg Traces
================================
    python preprocess.py

Reads borg_traces_data.csv -> cleans -> engineers features ->
builds static adjacency + per-window membership data -> saves to ./processed/

Output files:
  processed/machine_features.parquet  -- (machine_id, time_window, 13 features)
  processed/failure_labels.parquet    -- (machine_id, time_window) pairs about to fail
  processed/adjacency.json           -- static graph edges + node mapping (fallback)
  processed/window_membership.parquet -- per-window cluster/collection membership
  processed/metadata.json            -- global stats incl. failure ratio

Optimizations:
  - All feature columns downcast to float32 to save VRAM/RAM
  - Adjacency building uses vectorized NumPy (no nested Python loops)
  - Metadata includes global failure ratio for dynamic focal_alpha
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
    numeric_cols = [
        "time", "priority", "instance_index", "start_time", "end_time",
        "average_usage", "maximum_usage", "random_sample_usage",
        "assigned_memory", "page_cache_memory",
        "cycles_per_instruction", "memory_accesses_per_instruction",
        "sample_rate",
    ]
    for col in numeric_cols:
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
    """Build machine-level features, downcast to float32 for VRAM savings."""
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

    # Downcast all feature columns to float32 to save VRAM/RAM
    feat_cols = [c for c in grouped.columns if c not in ("machine_id", "time_window")]
    for col in feat_cols:
        grouped[col] = grouped[col].astype(np.float32)

    print(f"  {grouped.shape[0]:,} (machine, window) pairs")
    print(f"  {len(feat_cols)} features: {feat_cols}")
    print(f"  All features downcast to float32")
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


def build_static_adjacency_vectorized(df):
    """
    Build static adjacency using vectorized NumPy operations.
    Avoids nested Python loops for edge construction.
    """
    print("Building static adjacency graph (vectorized)...")

    machines = df["machine_id"].unique()
    machine_to_idx = {m: i for i, m in enumerate(sorted(machines))}
    num_nodes = len(machine_to_idx)
    print(f"  {num_nodes} nodes")

    all_src = []
    all_dst = []
    all_wgt = []

    def _add_edges_vectorized(groups_series, k_max, weight):
        """Vectorized edge building for a groupby result."""
        for _, m_set in groups_series.items():
            m_list = np.array([machine_to_idx[m] for m in m_set if m in machine_to_idx],
                              dtype=np.int64)
            na = len(m_list)
            if na < 2:
                continue
            if na <= k_max * 2:
                # Small group: full pairwise (vectorized)
                ii, jj = np.triu_indices(na, k=1)
                src = m_list[ii]
                dst = m_list[jj]
            else:
                # Large group: k-nearest neighbors by index (vectorized)
                K = min(k_max, na - 1)
                pos = np.arange(na)
                j_idx = pos[:, None] + np.arange(1, K + 1)[None, :]
                valid = j_idx < na
                i_val = np.broadcast_to(pos[:, None], (na, K))[valid]
                j_val = j_idx[valid]
                src = m_list[i_val]
                dst = m_list[j_val]

            # Bidirectional
            all_src.append(src)
            all_src.append(dst)
            all_dst.append(dst)
            all_dst.append(src)
            all_wgt.append(np.full(len(src) * 2, weight, dtype=np.float32))

    if "cluster" in df.columns:
        cluster_groups = (
            df[df["machine_id"].isin(machine_to_idx)]
            .groupby("cluster")["machine_id"].apply(set)
        )
        _add_edges_vectorized(cluster_groups, k_max=20, weight=1.0)
        print(f"  Cluster edges added")

    if "collection_id" in df.columns:
        coll_groups = (
            df[df["machine_id"].isin(machine_to_idx)]
            .groupby("collection_id")["machine_id"].apply(set)
        )
        _add_edges_vectorized(coll_groups, k_max=10, weight=0.5)
        print(f"  Collection edges added")

    # Self-loops
    self_nodes = np.arange(num_nodes, dtype=np.int64)
    all_src.append(self_nodes)
    all_dst.append(self_nodes)
    all_wgt.append(np.ones(num_nodes, dtype=np.float32))

    # Concatenate and deduplicate
    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    edges = np.stack([src, dst], axis=0)

    # Deduplicate edges (keep first occurrence)
    _, unique_idx = np.unique(edges, axis=1, return_index=True)
    unique_idx.sort()
    edges = edges[:, unique_idx]
    wgt = np.concatenate(all_wgt)[unique_idx]

    edge_list = edges.T.tolist()  # [[src, dst], ...]
    edge_weights = wgt.tolist()

    print(f"  Final: {len(edge_list)} edges (incl self-loops)")
    return machine_to_idx, edge_list, edge_weights


def build_window_membership(df):
    """
    Save per-window cluster/collection membership for dynamic edge building.
    Compact format (~20MB) vs pre-built edge lists (~500MB+).
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


def compute_metadata(features, labels):
    """Compute global statistics including failure ratio for dynamic focal_alpha."""
    total_pairs = features[["machine_id", "time_window"]].drop_duplicates().shape[0]
    num_positive = len(labels)
    failure_ratio = num_positive / max(total_pairs, 1)

    # Dynamic focal_alpha: higher alpha = more weight on rare failures
    # For ~0.0001 failure rate, alpha should be ~0.99
    dynamic_alpha = float(max(0.5, min(0.99, 1.0 - failure_ratio)))

    metadata = {
        "total_machine_window_pairs": int(total_pairs),
        "num_positive_labels": int(num_positive),
        "failure_ratio": float(failure_ratio),
        "recommended_focal_alpha": dynamic_alpha,
        "num_machines": int(features["machine_id"].nunique()),
        "num_time_windows": int(features["time_window"].nunique()),
        "num_features": int(len([c for c in features.columns
                                  if c not in ("machine_id", "time_window")])),
    }
    return metadata


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
    machine_to_idx, edge_list, edge_weights = build_static_adjacency_vectorized(df)
    membership = build_window_membership(df)
    metadata = compute_metadata(features, labels)

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

    with open("processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    feat_cols = [c for c in features.columns if c not in ("machine_id", "time_window")]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Nodes (machines):    {metadata['num_machines']:,}")
    print(f"  Time windows:        {metadata['num_time_windows']:,}")
    print(f"  Features per node:   {metadata['num_features']}")
    print(f"  Static edges:        {len(edge_list):,}")
    print(f"  Membership records:  {len(membership):,} (for dynamic edges)")
    print(f"  Positive labels:     {metadata['num_positive_labels']:,} / "
          f"{metadata['total_machine_window_pairs']:,} "
          f"({100*metadata['failure_ratio']:.4f}%)")
    print(f"  Failure ratio:       {metadata['failure_ratio']:.6f}")
    print(f"  Recommended alpha:   {metadata['recommended_focal_alpha']:.3f}")
    print(f"\n  Saved to ./processed/")
    print(f"  Metadata: ./processed/metadata.json")
    print(f"  Next: python train.py")


if __name__ == "__main__":
    main()
