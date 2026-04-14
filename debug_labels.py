import pandas as pd
import numpy as np
import json

labels = pd.read_parquet('processed/failure_labels.parquet')
feats = pd.read_parquet('processed/machine_features.parquet')

with open('processed/adjacency.json') as f:
    adj = json.load(f)

m2i = adj['machine_to_idx']
num_nodes = adj['num_nodes']

print(f"=== PREPROCESSED DATA ===")
print(f"Total feature rows (machine,window pairs): {len(feats)}")
print(f"Unique machines in features: {feats['machine_id'].nunique()}")
print(f"Unique windows in features: {feats['time_window'].nunique()}")
print(f"Total label=1 rows: {len(labels)}")
print(f"Unique machines in labels: {labels['machine_id'].nunique()}")
print(f"Num nodes in graph: {num_nodes}")

# What the model ACTUALLY sees per window
merged = feats[['machine_id','time_window']].merge(
    labels, on=['machine_id','time_window'], how='left')
merged['label'] = merged['label'].fillna(0)
pos = (merged['label']==1).sum()
total = len(merged)
print(f"\n=== ACTUAL LABEL DISTRIBUTION (feature rows) ===")
print(f"Positive: {pos} / {total} = {100*pos/total:.1f}%")

# But the model has num_nodes slots per window!
n_windows = feats['time_window'].nunique()
print(f"\n=== WHAT THE MODEL SEES (before masking) ===")
print(f"Total node slots per window: {num_nodes}")
print(f"Avg active nodes per window: {len(feats) / n_windows:.1f}")
print(f"Total node slots across all windows: {num_nodes * n_windows}")
print(f"Positive labels: {len(labels)}")
print(f"Effective positive rate WITHOUT mask: {100*len(labels)/(num_nodes * n_windows):.4f}%")

# Per-window breakdown for first 10 windows
windows = sorted(feats['time_window'].unique())[:10]
print(f"\n=== PER-WINDOW BREAKDOWN (first 10) ===")
for tw in windows:
    n_feat = len(feats[feats['time_window']==tw])
    n_label = len(labels[labels['time_window']==tw])
    rate = 100*n_label/max(n_feat,1)
    print(f"  window {tw}: {n_feat} active nodes, {n_label} failing, rate={rate:.1f}%")

# Check: are label machines even in the graph?
labels['machine_id'] = labels['machine_id'].astype(str)
in_graph = labels['machine_id'].isin(m2i).sum()
print(f"\n=== LABEL-GRAPH ALIGNMENT ===")
print(f"Label machines found in graph: {in_graph}/{len(labels)}")

# Check: are label machines in features for same window?
label_keys = set(zip(labels['machine_id'].astype(str), labels['time_window']))
feat_keys = set(zip(feats['machine_id'].astype(str), feats['time_window']))
overlap = len(label_keys & feat_keys)
print(f"Label (machine,window) pairs also in features: {overlap}/{len(label_keys)}")
print(f"  -> {100*overlap/max(len(label_keys),1):.1f}% of labels have matching features")
