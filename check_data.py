"""
Quick sanity check — run after preprocess.py
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

    files = [
        "processed/machine_features.parquet",
        "processed/failure_labels.parquet",
        "processed/adjacency.json",
        "processed/window_membership.parquet",
    ]
    for f in files:
        exists = "ok" if os.path.exists(f) else "MISSING"
        print(f"  [{exists}] {f}")

    if not os.path.exists("processed/machine_features.parquet"):
        print("\nRun preprocess.py first!")
        return

    feat = pd.read_parquet("processed/machine_features.parquet")
    print(f"\nFeatures: {feat.shape[0]:,} rows x {feat.shape[1]} cols")
    print(f"  Machines: {feat['machine_id'].nunique()}")
    print(f"  Time windows: {feat['time_window'].nunique()}")

    labels = pd.read_parquet("processed/failure_labels.parquet")
    print(f"\nLabels: {len(labels):,} positive samples")

    with open("processed/adjacency.json") as f:
        adj = json.load(f)
    print(f"\nStatic graph: {adj['num_nodes']} nodes, {len(adj['edges'])} edges")

    if os.path.exists("processed/window_membership.parquet"):
        mem = pd.read_parquet("processed/window_membership.parquet")
        print(f"Membership: {len(mem):,} records (for dynamic edges)")

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")

    print("\nAll good! Run: python train.py")


if __name__ == "__main__":
    main()
