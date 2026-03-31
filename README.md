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
9. [Model Architecture — In Detail](#model-architecture--in-detail)
10. [Syllabus Topics Covered](#syllabus-topics-covered)
11. [Hardware Requirements](#hardware-requirements)
12. [Output Files](#output-files)
13. [References](#references)

---

## What This Project Is About

Cloud data centers run thousands of interconnected machines. When one machine gets overloaded or fails, it doesn't just affect that machine — tasks get rescheduled to neighbors, load shifts, and failures **cascade** across the infrastructure like dominoes.

Traditional ML treats each machine as an independent data point. It looks at machine 47's CPU usage and asks "will it fail?" — but completely ignores that machine 47 shares a cluster with machines 12, 33, and 89, and that machine 12 just went down, meaning machine 47 is about to get flooded with rescheduled tasks.

This project models cloud infrastructure as a **graph** and uses a **Spatio-Temporal GNN** to:

1. **Learn spatial patterns** — how a machine's state is influenced by its neighbors (GraphSAGE)
2. **Learn temporal patterns** — how resource usage evolves over time (GRU)
3. **Predict cascading failures** 15 minutes before they happen
4. **Identify critical nodes** — which machines are structurally most dangerous

---

## Why Graphs?

Cloud infrastructure is inherently graph-structured:

- Machines in the **same cluster** share physical resources (network, power, cooling)
- Machines running tasks from the **same collection** have workload dependencies
- When machine A fails, its tasks migrate to connected machines B and C, increasing their load

A standard neural network sees each machine as an isolated row in a table. A Graph Neural Network sees the **connections** and aggregates information from neighbors — so machine 47's failure prediction is informed by every machine it's connected to.

---

## Architecture Overview

```
Input: 6 consecutive graph snapshots (30 min of history)
       Each snapshot: ~4,900 nodes x 12 features + graph edges

                    ┌──────────────────────────┐
  Snapshot t=1 ──> │  2-layer GraphSAGE         │ ──> h1  [N, 48]
  Snapshot t=2 ──> │  (samples 15 neighbors/node │ ──> h2  [N, 48]
  Snapshot t=3 ──> │   per layer — scalable to   │ ──> h3  [N, 48]
  Snapshot t=4 ──> │   any graph size)            │ ──> h4  [N, 48]
  Snapshot t=5 ──> │  + residual connections      │ ──> h5  [N, 48]
  Snapshot t=6 ──> │                              │ ──> h6  [N, 48]
                    └──────────────────────────┘
                                │
                    Stack: H = [h1...h6]  →  [N, 6, 48]
                                │
                    ┌──────────────────────────┐
                    │  GRU                      │
                    │  (temporal encoder)        │
                    │  take last hidden state    │
                    └──────────────────────────┘
                                │
                           z  [N, 48]
                                │
                    ┌──────────────────────────┐
                    │  MLP Classifier            │
                    │  48 → 24 → 2              │
                    └──────────────────────────┘
                                │
                    logits  [N, 2]  →  Normal / Failing per node
```

---

## Dataset

**Source:** Google Borg Cluster Trace (processed subset)

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| Columns | 34 |
| Unique machines | ~4,893 |
| Clusters | 8 |
| Failure rate | ~22.8% (1,143 failed / 3,857 normal) |
| Event types | 8 (SUBMIT, SCHEDULE, EVICT, FAIL, FINISH, KILL, etc.) |

**Key columns:** `machine_id`, `time` (microseconds), `average_usage` / `maximum_usage` (CPU/memory), `assigned_memory`, `priority`, `scheduling_class`, `cluster`, `collection_id`, `failed` (binary)

---

## Project Structure

```
project/
├── config.yaml          # All hyperparameters
├── requirements.txt     # Python dependencies
├── preprocess.py        # Step 1: CSV → features + labels + graph
├── model.py             # GraphSAGE + GRU + Focal Loss
├── train.py             # Step 2: Training with neighbor sampling
├── evaluate.py          # Step 3: All plots and analysis
├── check_data.py        # Sanity check after preprocessing
├── processed/           # (generated)
│   ├── machine_features.parquet
│   ├── failure_labels.parquet
│   ├── adjacency.json
│   └── test_results.npz
├── results/             # (generated)
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   ├── critical_nodes.png
│   ├── failure_propagation.png
│   └── embedding_tsne.png
└── best_model.pt        # (generated) model checkpoint
```

---

## Setup & Installation

```bash
# Python 3.9+ required
pip install -r requirements.txt

# If PyTorch CUDA isn't set up:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## How to Run

Place `borg_traces_data.csv` in the project folder, then:

```bash
python preprocess.py     # Step 1: preprocess raw data
python check_data.py     # (optional) verify everything looks right
python train.py          # Step 2: train the model
python evaluate.py       # Step 3: generate all plots
```

Results will be in `./results/`.

---

## Pipeline Walkthrough

### Step 1: Preprocessing (`preprocess.py`)

Takes the raw Borg CSV and produces three files.

**Time windowing:** Borg timestamps (microseconds) are bucketed into 5-minute windows, giving discrete timesteps.

**12 features per machine per time window:**

| Feature | What it captures |
|---------|------------------|
| `average_usage_mean/std/max` | CPU load profile (mean, volatility, peak) |
| `maximum_usage_mean/std/max` | Memory load profile |
| `assigned_memory_sum` | Total memory allocated |
| `instance_index_count` | Number of active tasks |
| `failed_sum` / `failed_mean` | Failure count and rate |
| `priority_mean` | Average task priority |
| `scheduling_class_nunique` | Workload diversity |
| `instance_events_type_count` | Scheduling event count |

**Failure labels:** A machine is labeled "about to fail" at time t if it fails within windows t+1 to t+3 (15-minute prediction horizon).

**Weighted adjacency:** Cluster co-location edges (weight 1.0) + shared collection edges (weight 0.5) + self-loops. Edge weights encode that physical co-location is a stronger dependency than shared workloads.

### Step 2: Training (`train.py`)

**Full graph, no subsampling.** Unlike the earlier version that subsampled to 500 nodes (losing information), we now use ALL ~4,900 machines. GraphSAGE's neighbor sampling makes this possible — each node samples 15 neighbors per layer, keeping computation bounded regardless of graph size.

**On-the-fly sequence building:** Sequences are built from compact parquet files when needed (~50MB), instead of a pre-built .pt file (~700MB). This is why the old code was crashing.

**Training details:** Mini-batch gradient accumulation (batch=8), Adam with weight decay, ReduceLROnPlateau scheduler, Focal Loss for class imbalance, early stopping (patience=10), temporal train/val/test split (70/15/15).

### Step 3: Evaluation (`evaluate.py`)

Generates five visualizations:

1. **Confusion Matrix** — TP/FP/TN/FN breakdown
2. **ROC & PR Curves** — discrimination ability and precision-recall tradeoff
3. **Critical Nodes** — gradient-based importance scores showing which machines most influence failure predictions (GNN interpretability)
4. **Failure Propagation** — graph visualization showing failures spreading across nodes over time
5. **t-SNE Embeddings** — 2D projection of learned 48-dim node representations

---

## Model Architecture — In Detail

### Why GraphSAGE?

The core scaling problem: our graph has ~4,900 nodes. GCN does **full neighborhood aggregation** — every node aggregates ALL its neighbors, every layer, every timestep. At 4,900 nodes with dense cluster connectivity, this blows up VRAM (this is exactly why the old GCN-based code crashed).

GraphSAGE solves this by **sampling** a fixed number of neighbors (k=15) per node per layer. Whether a node has 5 or 500 neighbors, it only aggregates 15. This makes computation O(N × k) instead of O(|E|), and crucially, it's bounded — VRAM usage doesn't depend on graph density.

| Model | Aggregation | Scales to full graph? | Our use case |
|-------|------------|----------------------|-------------|
| GCN | All neighbors | No — crashed at 3,000 nodes | Too expensive |
| **GraphSAGE** | **Sample k neighbors** | **Yes — runs all 4,900 nodes** | **What we use** |
| GAT | Attention over all neighbors | Memory-heavy | Overkill — we encode importance via edge weights |

During training, the random neighbor sampling also acts as a **regularizer** (like dropout on the graph structure), reducing overfitting. During evaluation, we use the full neighborhood for deterministic predictions.

### Why 2 Layers?

- **1 layer:** Each node sees direct neighbors only (1-hop). Machine C can't detect that machine A (2 hops away) failed.
- **2 layers:** Each node sees neighbors-of-neighbors (2-hop). Failure signal from A propagates through B to reach C — exactly what cascading failure detection needs.
- **3+ layers:** Over-smoothing — all node representations converge to the same vector, losing discriminative power.

### Why GRU (not LSTM)?

GRU has 2 gates vs LSTM's 3. For short sequences (length 6), the extra gate doesn't help. GRU trains faster and uses less memory.

### Why Not a Temporal GNN (DCRNN, A3TGCN)?

Temporal GNNs run graph convolutions *inside* each recurrent step. For T=6 and 2 SAGE layers, that's 12 graph convolution calls per forward pass. Our decoupled approach (SAGE first, then GRU) does 6 SAGE calls + 1 GRU pass. Since our graph structure doesn't change across the sequence (same machines, same edges), the coupling adds cost without benefit.

### Why Focal Loss?

~78% of nodes are normal, ~22% failing. Standard cross-entropy lets the model get 78% accuracy by always predicting "normal." Focal Loss (alpha=0.75, gamma=2.0) gives 3x weight to the failing class and down-weights easy predictions, forcing the model to focus on hard cases near the decision boundary.

### Residual Connections

Without residuals, the 2nd SAGE layer's output is purely aggregated neighbor information — the node's own features get diluted. The residual connection (`x = x + residual`) preserves each node's own CPU/memory stats so the classifier can still see them directly.

---

## Syllabus Topics Covered

| Syllabus Topic | Where in the project |
|----------------|---------------------|
| ML on non-Euclidean data | Entire project — cloud infra is a graph |
| Graph SAGE | `model.py` → `SpatialEncoder` with `SAGEConv` |
| Generalized neighborhood aggregation | SAGE's sample-and-aggregate framework |
| Stacking GNN layers | 2-layer SAGE with residual connections |
| Dynamic graphs / spatial-temporal GNN | GCN→GRU architecture for temporal graph data |
| GNN layer optimization | LayerNorm, dropout, residual connections |
| GNN Interpretability | Gradient-based critical node identification |
| Anomaly detection | Failure prediction as graph anomaly detection |
| Node embeddings | t-SNE visualization of learned representations |
| Encoder-decoder perspective | SAGE+GRU encoder, MLP decoder |
| Loss functions | Focal Loss for imbalanced node classification |
| Setting up graph datasets | Borg trace → weighted graph construction |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA 4GB+ | RTX 4060 Mobile 8GB |
| RAM | 8 GB | 16 GB |
| Disk | 500 MB | 1 GB |

Works on CPU too (just slower). GraphSAGE neighbor sampling keeps VRAM bounded at ~17MB per forward pass for 4,900 nodes — well within any modern GPU.

---

## Output Files

```
processed/                          # After preprocess.py
├── machine_features.parquet        # Node features
├── failure_labels.parquet          # Failure labels
├── adjacency.json                  # Weighted graph
└── test_results.npz                # Test predictions (after train.py)

results/                            # After evaluate.py
├── confusion_matrix.png
├── roc_pr_curves.png
├── critical_nodes.png
├── failure_propagation.png
└── embedding_tsne.png

best_model.pt                       # Trained model (after train.py)
```

---

## References

1. Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. (GraphSAGE)
2. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
3. Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder.* (GRU)
4. Google Cluster Data — Borg Traces. https://github.com/google/cluster-data
5. Modeling and Analysis of Cascading Failures in Cloud Computing — ScienceDirect
6. GNN-Based Dynamic Multiqueue Optimization for Cloud Fault Tolerance — Wiley

---

*Built for the Interdisciplinary Deep Learning on Graphs course, PES University.*
