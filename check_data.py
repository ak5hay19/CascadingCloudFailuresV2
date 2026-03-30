"""
Quick sanity check — run after preprocess.py to verify data looks correct.
    python check_data.py
"""

import os
import json
import pandas as pd
import torch


def main():
    print("=" * 50)
    print("DATA SANITY CHECK")
    print("=" * 50)

    # Check files exist
    files = [
        "processed/machine_features.parquet",
        "processed/failure_labels.parquet",
        "processed/adjacency.json",
    ]
    for f in files:
        exists = "✓" if os.path.exists(f) else "✗ MISSING"
        print(f"  {exists}  {f}")

    if not all(os.path.exists(f) for f in files):
        print("\nRun preprocess.py first!")
        return

    # Features
    feat = pd.read_parquet("processed/machine_features.parquet")
    print(f"\nFeatures: {feat.shape[0]:,} rows × {feat.shape[1]} cols")
    print(f"  Machines: {feat['machine_id'].nunique()}")
    print(f"  Time windows: {feat['time_window'].nunique()}")
    feat_cols = [c for c in feat.columns if c not in ("machine_id", "time_window")]
    print(f"  Feature cols ({len(feat_cols)}): {feat_cols}")

    # Labels
    labels = pd.read_parquet("processed/failure_labels.parquet")
    print(f"\nLabels: {len(labels):,} positive samples")
    if len(labels) > 0:
        print(f"  Unique failing machines: {labels['machine_id'].nunique()}")
        total = feat[["machine_id", "time_window"]].drop_duplicates().shape[0]
        print(f"  Failure rate: {100*len(labels)/max(total,1):.2f}%")

    # Adjacency
    with open("processed/adjacency.json") as f:
        adj = json.load(f)
    print(f"\nGraph:")
    print(f"  Nodes: {adj['num_nodes']}")
    print(f"  Edges: {len(adj['edges'])}")
    has_weights = "edge_weights" in adj
    print(f"  Edge weights: {'yes' if has_weights else 'no'}")

    # CUDA check
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem:.1f} GB")

    print("\n✓ All good! Run: python train.py")


if __name__ == "__main__":
    main()
