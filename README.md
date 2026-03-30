# Modelling Cascading Resource Failures in Cloud Infrastructure using Spatio-Temporal Graph Deep Learning

> **Course:** Interdisciplinary Deep Learning on Graphs (CSE-AIML)
>
> **Done by:** Akshay P Shetti, Tarun S, Aadithyaa Kumar, Adarsh R Menon

---

## Table of Contents

1. [What This Project Is About](#what-this-project-is-about)
2. [Why Graphs?](#why-graphs)
3. [Architecture Overview](#architecture-overview)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [How to Run](#how-to-run)
8. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Step 1: Preprocessing](#step-1-preprocessing-preprocesspy)
   - [Step 2: Training](#step-2-training-trainpy)
   - [Step 3: Evaluation](#step-3-evaluation-evaluatepy)
9. [Model Architecture — In Detail](#model-architecture--in-detail)
10. [Syllabus Topics Covered](#syllabus-topics-covered)
11. [Hardware Requirements](#hardware-requirements)
12. [Output Files](#output-files)
13. [References](#references)

---

## What This Project Is About

Cloud data centers run thousands of machines. When one machine gets overloaded or fails, it doesn't just affect that machine — tasks get rescheduled to neighbors, load shifts, and failures can **cascade** across the infrastructure like dominoes.

Traditional ML treats each machine independently: "is machine 47's CPU high? then it might fail." But it completely ignores that machine 47 shares a cluster with machines 12, 33, and 89 — and machine 12 just went down, meaning machine 47 is about to get flooded with rescheduled tasks.

This project models the cloud infrastructure as a **graph** (machines = nodes, relationships = edges) and uses a **Spatio-Temporal Graph Neural Network** to:

1. **Learn spatial patterns** — how a machine's state is influenced by its neighbors (GCN)
2. **Learn temporal patterns** — how resource usage evolves over time (GRU)
3. **Predict cascading failures** before they happen (15 minutes ahead)
4. **Identify critical nodes** — which machines are structurally most dangerous if they fail

---

## Why Graphs?

Cloud infrastructure is inherently graph-structured:

- Machines in the **same cluster** share physical resources (network, power, cooling)
- Machines running tasks from the **same job collection** have workload dependencies
- When machine A fails, its tasks migrate to connected machines B and C, increasing their load

A standard neural network sees each machine as an isolated row in a table. A **Graph Neural Network** sees the connections. Each node aggregates information from its neighbors, so machine 47's prediction is informed by the current state of every machine it's connected to.

This is exactly the **message passing** framework from the course:

```
h_i = σ(W · AGGREGATE({h_j : j ∈ N(i) ∪ {i}}))
```

Each node updates its representation by aggregating neighbor features — which is what GCNConv does under the hood.

---

## Architecture Overview

```
Input: 6 consecutive graph snapshots (30 minutes of history)
       Each snapshot: 500 nodes × 12 features + edge structure

                    ┌─────────────────────┐
  Snapshot t=1 ───▶ │  2-layer GCN        │ ───▶ h₁  [500, 48]
  Snapshot t=2 ───▶ │  (spatial encoder)   │ ───▶ h₂  [500, 48]
  Snapshot t=3 ───▶ │  with residual       │ ───▶ h₃  [500, 48]
  Snapshot t=4 ───▶ │  connections         │ ───▶ h₄  [500, 48]
  Snapshot t=5 ───▶ │  + edge weights      │ ───▶ h₅  [500, 48]
  Snapshot t=6 ───▶ │                      │ ───▶ h₆  [500, 48]
                    └─────────────────────┘
                              │
                    Stack: H = [h₁...h₆]  →  [500, 6, 48]
                              │
                    ┌─────────────────────┐
                    │  GRU                 │
                    │  (temporal encoder)   │
                    │  last hidden state   │
                    └─────────────────────┘
                              │
                         z  [500, 48]
                              │
                    ┌─────────────────────┐
                    │  MLP Classifier      │
                    │  48 → 24 → 2        │
                    └─────────────────────┘
                              │
                    logits  [500, 2]  →  Normal / Failing per node
```

---

## Dataset

**Source:** Google Borg Cluster Trace (processed subset)

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| Columns | 34 |
| Unique machines | 4,893 |
| Clusters | 8 |
| Failure rate | ~22.8% (1,143 failed / 3,857 normal) |
| Event types | 8 (SUBMIT, SCHEDULE, EVICT, FAIL, FINISH, KILL, etc.) |

**Key columns:**

- `machine_id` — unique machine identifier
- `time` — timestamp in microseconds
- `average_usage`, `maximum_usage` — CPU/memory resource usage
- `assigned_memory`, `page_cache_memory` — memory allocation
- `priority`, `scheduling_class` — task scheduling metadata
- `cluster`, `collection_id` — infrastructure grouping
- `failed` — binary label (0 = normal, 1 = failed)

---

## Project Structure

```
project/
├── config.yaml          # All hyperparameters
├── requirements.txt     # Python dependencies
├── preprocess.py        # Step 1: CSV → features + labels + graph
├── model.py             # ST-GNN architecture (GCN + GRU + Focal Loss)
├── train.py             # Step 2: Training loop with VRAM optimization
├── evaluate.py          # Step 3: Plots and analysis
├── check_data.py        # Sanity check after preprocessing
├── processed/           # (generated) preprocessed data
│   ├── machine_features.parquet
│   ├── failure_labels.parquet
│   ├── adjacency.json
│   └── test_results.npz
├── results/             # (generated) evaluation plots
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   ├── critical_nodes.png
│   ├── failure_propagation.png
│   └── embedding_tsne.png
└── best_model.pt        # (generated) saved model checkpoint
```

---

## Setup & Installation

**Prerequisites:** Python 3.9+, pip, CUDA-compatible GPU (optional but recommended)

```bash
# Clone or copy the project files
cd project/

# Install dependencies
pip install -r requirements.txt

# For PyTorch with CUDA (if not already installed):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is detected:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## How to Run

Place `borg_traces_data.csv` in the project folder, then run these three commands in order:

```bash
# Step 1: Preprocess the raw data
python preprocess.py

# (Optional) Verify preprocessing worked
python check_data.py

# Step 2: Train the model
python train.py

# Step 3: Generate all evaluation plots
python evaluate.py
```

That's it. Results will be in `./results/`.

---

## Pipeline Walkthrough

### Step 1: Preprocessing (`preprocess.py`)

**What it does:** Takes the raw Borg CSV and produces three files that the training script needs.

**1a. Data Cleaning**
- Converts columns to correct types (numeric, string, binary)
- Drops rows with missing machine IDs
- Converts `failed` column to binary 0/1

**1b. Time Windowing**
- Borg timestamps are in microseconds
- We bucket them into **5-minute windows** (300 seconds)
- This turns continuous time into discrete steps the model can iterate over

**1c. Feature Engineering — 12 features per machine per time window:**

| # | Feature | What it captures |
|---|---------|------------------|
| 1 | `average_usage_mean` | Typical CPU load |
| 2 | `average_usage_std` | CPU volatility (spiky = unstable) |
| 3 | `average_usage_max` | Peak CPU stress |
| 4 | `maximum_usage_mean` | Typical memory load |
| 5 | `maximum_usage_std` | Memory volatility |
| 6 | `maximum_usage_max` | Peak memory stress |
| 7 | `assigned_memory_sum` | Total memory allocated |
| 8 | `instance_index_count` | Number of active tasks (workload intensity) |
| 9 | `failed_sum` | Count of failed tasks |
| 10 | `failed_mean` | Failure rate |
| 11 | `priority_mean` | Average task priority |
| 12 | `scheduling_class_nunique` | Scheduling diversity (heterogeneous workload) |

**1d. Failure Labels**
- A machine is labeled "about to fail" at time `t` if it has any failures in windows `t+1`, `t+2`, or `t+3` (prediction horizon = 3 windows = 15 minutes)
- This means the model learns to predict failures **before they happen**, not at the moment of failure

**1e. Graph Adjacency (Weighted)**
- **Cluster edges (weight 1.0):** Machines in the same cluster share physical infrastructure — strongest dependency
- **Collection edges (weight 0.5):** Machines running tasks from the same job collection share workload — weaker but real dependency
- **Self-loops (weight 1.0):** Every node connects to itself so its own features are preserved during message passing
- Edge weights let the GCN learn that cluster co-location is a stronger signal than shared collections

**Output:**
```
processed/machine_features.parquet  — node features per time window
processed/failure_labels.parquet    — which (machine, window) pairs are about to fail
processed/adjacency.json            — graph edges + edge weights + node mapping
```

---

### Step 2: Training (`train.py`)

**Data Loading — On-the-fly (not pre-built)**
- Old approach: build all graph sequences into a giant `.pt` file (700MB+ in RAM)
- New approach: store compact parquet files, build each sequence when needed (~50MB)
- This is why the old code was crashing — it tried to hold everything in memory at once

**Node Subsampling**
- Full graph has ~4,900 nodes, but we only train on 500
- Selection priority: (1) machines that actually have failures, (2) most active machines, (3) random fill
- This makes training 10x faster without losing signal — the failure-relevant machines are all kept

**Feature Normalization**
- Z-score normalization: `x = (x - mean) / std`
- Computed from a sample of 100 time windows
- Ensures CPU usage (0-1 range) and memory counts (0-1000+ range) are on comparable scales

**Training Loop**
- Mini-batch gradient accumulation (batch_size=8): process 8 sequences, accumulate gradients, then update weights
- This smooths training and keeps VRAM usage bounded
- Gradient clipping at 1.0 to prevent exploding gradients (common with GRU)
- Adam optimizer with weight decay (L2 regularization)
- ReduceLROnPlateau: halves learning rate if validation F1 doesn't improve for 4 epochs
- Early stopping: stops training if no improvement for 10 epochs

**Loss Function: Focal Loss**
- Dataset is imbalanced (~78% normal, ~22% failing)
- Standard cross-entropy would let the model get 78% accuracy by always predicting "normal"
- Focal Loss with alpha=0.75, gamma=2.0:
  - `alpha=0.75`: gives 3x more weight to the failing class
  - `gamma=2.0`: down-weights easy (high-confidence) predictions, forces focus on hard cases

**Temporal Split**
- Train: first 70% of sequences (earliest time windows)
- Validation: next 15%
- Test: last 15% (most recent time windows)
- This is a **temporal split**, not random — prevents data leakage from future to past

**Output:**
```
best_model.pt               — model weights + config + normalization stats
processed/test_results.npz  — test predictions, labels, probabilities
```

---

### Step 3: Evaluation (`evaluate.py`)

Generates five visualizations in `./results/`:

**1. Confusion Matrix** (`confusion_matrix.png`)
- Shows true positives, false positives, true negatives, false negatives
- Tells you: "of the nodes the model flagged as failing, how many actually failed?"

**2. ROC & Precision-Recall Curves** (`roc_pr_curves.png`)
- ROC-AUC: overall discrimination ability (how well the model separates normal from failing)
- PR-AUC: more informative for imbalanced data — precision at each recall threshold

**3. Critical Node Identification** (`critical_nodes.png`)
- Uses gradient-based analysis: computes `d(loss)/d(input_features)` for failure samples
- Nodes with high gradient magnitude are the ones whose features most influence failure predictions
- These are the **structurally critical machines** — if they fail, cascading effects are worst
- This is the **GNN Interpretability** component from the syllabus

**4. Failure Propagation** (`failure_propagation.png`)
- Visualizes the graph across 6 time windows
- Red nodes = failing, green = normal
- Shows how failures spread from a few machines to their neighbors over time
- Demonstrates the cascading effect that motivates using a graph model

**5. t-SNE Embeddings** (`embedding_tsne.png`)
- Projects the 48-dimensional learned node embeddings into 2D
- If the model learned meaningful representations, normal and failing nodes form separable clusters
- This is the **node embedding / representation learning** concept from the syllabus

---

## Model Architecture — In Detail

### Why GCN (not GAT, GraphSAGE, or GIN)?

| Model | Mechanism | Speed | Our choice? |
|-------|-----------|-------|------------|
| **GCN** | Mean aggregation with learned weights | Fast | **Yes** — best speed/quality for 500 nodes |
| GAT | Attention-weighted aggregation | 3x slower | No — attention overhead not worth it for this graph size |
| GraphSAGE | Sampled neighborhood aggregation | Medium | No — designed for very large graphs (millions of nodes) |
| GIN | Sum aggregation (most expressive) | Medium | No — overkill for binary classification |

### Why 2 GCN Layers (not 1 or 3+)?

- **1 layer:** Each node sees only direct neighbors (1-hop). Machine C can't see that machine A (2 hops away) just failed.
- **2 layers:** Each node sees neighbors-of-neighbors (2-hop). Machine C now knows about machine A's failure through machine B.
- **3+ layers:** Over-smoothing — all node representations converge to the same vector. The model loses the ability to distinguish individual machines.

### Why GRU (not LSTM or Temporal GNN)?

- **GRU vs LSTM:** GRU has 2 gates (reset, update) vs LSTM's 3 (forget, input, output). For short sequences (length 6), the extra gate doesn't help but does slow training.
- **GRU vs Temporal GNN (DCRNN/A3TGCN):** Temporal GNNs run graph convolutions *inside* each recurrent step — for T=6 and 2 GCN layers, that's 12 GCN calls. Our decoupled approach does 6 GCN calls + 1 GRU pass. Since our graph structure is static across the sequence (same machines, same edges), the coupling doesn't add value.

### Why Residual Connections?

Without residuals, the 2nd GCN layer's output is purely aggregated neighbor information — the node's own original features are diluted. The residual connection (`x = x + residual`) preserves the node's own CPU/memory stats so the classifier can still see them directly.

### Why Focal Loss (not Weighted Cross-Entropy)?

Both handle class imbalance, but Focal Loss is **adaptive**. Weighted CE gives a fixed 3x multiplier to failing nodes. Focal Loss gives a *variable* multiplier based on confidence — easy predictions (the model is already sure) get almost no gradient, while hard predictions (ambiguous cases near the decision boundary) get amplified. This focuses training on the cases that matter.

---

## Syllabus Topics Covered

| Syllabus Topic | Where in the project |
|----------------|---------------------|
| Machine learning on non-Euclidean data | The entire project — cloud infrastructure is a graph, not a grid |
| Graph Convolution Networks (GCN) | `model.py` → `SpatialEncoder` using `GCNConv` |
| Stacking GNN layers | 2-layer GCN with residual connections |
| Dynamic graphs spatial-temporal GNN | Full ST-GNN architecture (GCN → GRU → MLP) |
| GNN layer optimization | LayerNorm, dropout, residual connections in `SpatialEncoder` |
| Generalized neighborhood aggregation | GCN's weighted mean-aggregation with edge weights |
| GNN Interpretability | Gradient-based critical node identification in `evaluate.py` |
| Anomaly detection (interdisciplinary application) | Failure prediction is anomaly detection on dynamic graphs |
| Node embeddings / representation learning | t-SNE visualization of learned 48-dim embeddings |
| Encoder perspective | GCN+GRU as encoder, MLP as decoder for node classification |
| Graph datasets and learning tasks | Borg trace → graph construction → node classification task |
| Loss functions | Focal Loss for imbalanced node classification |

---

## Hardware Requirements

| Component | Minimum | Recommended (tested) |
|-----------|---------|---------------------|
| GPU | Any CUDA GPU with 4GB+ VRAM | NVIDIA RTX 4060 Mobile (8GB) |
| RAM | 8 GB | 16 GB |
| CPU | Any modern CPU | Works on CPU too (just slower) |
| Disk | 500 MB free | 1 GB free |

**VRAM optimizations in the code:**
- On-the-fly sequence generation (no giant .pt file in memory)
- 500-node subsampling (from ~4,900)
- Mini-batch gradient accumulation (batch_size=8)
- `torch.cuda.empty_cache()` every 50 sequences
- hidden_dim=48 (not 64/128)

---

## Output Files

After running all three steps, you'll have:

```
processed/
├── machine_features.parquet    # 12 features per (machine, time_window)
├── failure_labels.parquet      # positive labels for failure prediction
├── adjacency.json              # graph edges with weights
└── test_results.npz            # test set predictions

results/
├── confusion_matrix.png        # TP/FP/TN/FN breakdown
├── roc_pr_curves.png           # ROC-AUC and PR-AUC curves
├── critical_nodes.png          # top-20 critical machines by gradient importance
├── failure_propagation.png     # graph visualization showing failure spread
└── embedding_tsne.png          # 2D projection of learned node representations

best_model.pt                   # trained model checkpoint
```

---

## References

1. Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
2. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
3. Google Cluster Data. *Borg Cluster Traces.* https://github.com/google/cluster-data
4. Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* (GRU paper)
5. Modeling and Analysis of Cascading Failures in Cloud Computing Systems — ScienceDirect
6. A Graph Neural Network-Based Approach With Dynamic Multiqueue Optimization Scheduling (DMQOS) — Wiley Online Library

---

*Built for the Interdisciplinary Deep Learning on Graphs course project, PES University.*
