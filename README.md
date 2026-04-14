# Modelling Cascading Resource Failures in Cloud Infrastructure using Spatio-Temporal Graph Deep Learning

> **Course:** Interdisciplinary Deep Learning on Graphs (CSE-AIML)
>
> **Done by:** Akshay P Shetti, Tarun S, Aadithyaa Kumar, Adarsh R Menon

---

## What This Project Is About

Cloud data centers run thousands of interconnected machines. When one machine fails, tasks get rescheduled to neighbors, load shifts, and failures **cascade** across the infrastructure. Traditional ML treats each machine independently — our approach models the infrastructure as a **dynamic graph** and uses a **Spatio-Temporal GNN** to predict cascading failures before they happen.

**Three novel contributions:**

1. **Leak-Proof Ghost Injection** — To solve extreme class imbalance without causing a "mask leak" (where the model memorizes that being in the loss mask means you are failing), we inject "Decoy Ghost" nodes (normal machines with 0 features) at a 10:1 ratio into the evaluation mask.
2. **Dual-Stage Thermal Safety** — Custom pipeline engineering for laptop GPUs. Implements precise inter-batch (1s) and inter-epoch (15s) cooldown sleeps, stabilizing an RTX 4060 Mobile at 80-87°C GPU and 94-95°C CPU (down from thermal throttling at 89°C) on an ASUS G14.
3. **Edge contagion scoring (with Min-Max Scaling)** — identifies which specific connections between machines carry the most failure signal. Scores are scaled between 1.0 and 5.0 for clear interpretability, telling operators which dependencies to sever during incidents.

---

## Final Results

This table represents the model's final testing metrics against the optimized dynamically scaled predictive range using `eval_threshold = 0.1`.

| Metric | Value |
|--------|-------|
| **AUROC** | 0.745 |
| **Recall (Fail Class)** | 1.00 (99.8%) |
| **Precision (Fail Class)**| 0.09 |
| **F1-Score (Weighted)** | 0.64 |

---

## Architecture Overview

```
Input: 6 consecutive DYNAMIC graph snapshots (30 min history)
       Each snapshot: ~4,900 nodes × 25 features + PER-WINDOW edges

                    ┌──────────────────────────┐
  Snapshot t=1 ──> │  2-layer GraphSAGE         │ ──> h1  [N, 32]
  (edges_t1)  ──>  │  (samples 15 neighbors/node │
  Snapshot t=2 ──> │   per layer)                │ ──> h2  [N, 32]
  (edges_t2)  ──>  │                             │
  ...              │  edges CHANGE per timestep   │
  Snapshot t=6 ──> │  + residual connections      │ ──> h6  [N, 32]
  (edges_t6)  ──>  │                              │
                    └──────────────────────────┘
                                │
                    Stack: H = [h1...h6]  →  [N, 6, 32]
                                │
                    ┌──────────────────────────┐
                    │  GRU (temporal encoder)    │
                    │  last hidden state         │
                    └──────────────────────────┘
                                │
                           z  [N, 32]
                                │
                    ┌──────────────────────────┐
                    │  MLP: 32 → 16 → 2        │
                    └──────────────────────────┘
                                │
                    logits  [N, 2]  →  Normal / Failing per node
```

**Key difference from standard ST-GNNs:** the `edge_index` is a **list of 6 different tensors** — one per timestep — not a single shared tensor. The model sees the graph topology evolve over time.

---

## Dataset

**Source:** Google Borg Cluster Trace 2019

| Property | Value |
|----------|-------|
| **Base File** | `borg_traces_half.csv` |
| **Columns** | Reduced to 25 safe predictive features (no leaky columns) |
| **Unique machines** | ~4,893 |
| **Clusters** | 8 |
| **Time windows** | ~8,900 (5-min each) |
| **Failure rate** | ~70.5% (Mildly imbalanced) |

---

## Project Structure

```
project/
├── config.yaml          # All hyperparameters
├── requirements.txt     # Python dependencies
├── preprocess.py        # Step 1: CSV → features + labels + graph + membership
├── model.py             # Dynamic GraphSAGE + GRU + Focal Loss
├── train.py             # Step 2: Training with dynamic per-window edges
├── evaluate.py          # Step 3: All plots + MC Dropout + Edge Contagion
├── debug_labels.py      # Label distribution inspector
├── processed/           # (generated)
│   ├── machine_features.parquet
│   ├── failure_labels.parquet
│   ├── adjacency.json
│   ├── metadata.json
│   ├── window_membership.parquet   ← NEW: for dynamic edges
│   └── test_results.npz
├── results/             # (generated)
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   ├── critical_nodes.png
│   ├── failure_propagation.png
│   ├── embedding_tsne.png
│   ├── mc_dropout_uncertainty.png  ← NEW
│   ├── edge_contagion_scores.png   ← NEW
│   └── contagion_network.png       ← NEW
└── best_model.pt
```

---

## How to Run

```bash
pip install -r requirements.txt

python preprocess.py     # Step 1: preprocess + build membership data
python train.py          # Step 2: train with dynamic edges
python evaluate.py       # Step 3: all plots + novel analysis
```

---

## Pipeline Walkthrough

### Step 1: Preprocessing (`preprocess.py`)

- **Time windowing:** Borg microsecond timestamps → 5-minute buckets
- **25 predictive features:** Cleaned list (dropped leaky counts such as `failed_sum`), mapping task, event, and load values per window 
- **Failure labels:** Machines labeled "about to fail" if they fail within the next 3 windows (15-minute prediction horizon)
- **Static adjacency:** Cluster edges (weight 1.0) + collection edges (weight 0.5) + self-loops
- **Window membership data (NEW):** Per-window records of which machines are in which clusters/collections — this is what enables dynamic edge building during training

### Step 2: Training (`train.py`)

**Dynamic graph topology (novel contribution #1):**
- For each time window in a sequence, edges are built on-the-fly from the membership data
- Only machines ACTIVE in that window get connected
- This means the graph structure evolves: machines joining a collection create new edges, machines leaving lose them
- The model forward pass accepts a LIST of edge tensors — one per timestep

**GraphSAGE neighbor sampling:**
- Samples 15 neighbors per node per layer during training
- Keeps VRAM bounded at ~17MB per forward pass regardless of graph density
- Random sampling acts as graph-structure dropout (regularizer)
- Full neighborhood used during eval for deterministic predictions

**Training details:**
- Threshold Sweep applied across iterations for matching operational risk bounds (`eval_threshold = 0.1` selected).
- Stabilized thermal output running natively on ASUS G14 laptop GPU limiting temp ranges.
- Focal Loss for class imbalance.
- Mini-batch gradient accumulation (batch=2 limit).
- Temporal train/val/test split (70/15/15)

### Step 3: Evaluation (`evaluate.py`)

**Standard analysis:**
- Confusion matrix, ROC & PR curves
- Gradient-based critical node identification (GNN interpretability)
- Failure propagation visualization across time
- t-SNE of learned 48-dim node embeddings

**MC Dropout Uncertainty (novel contribution #2):**
- Runs 50 forward passes with dropout active (`model.train()` during inference)
- Each pass gives slightly different predictions (different neurons dropped)
- Mean across 50 passes = robust prediction
- Std across 50 passes = uncertainty per node
- No retraining needed — same model, different inference mode
- Output: uncertainty distribution, prediction-vs-confidence scatter, top uncertain nodes

**Edge Contagion Scoring (novel contribution #3):**
- Computes gradient of loss w.r.t. input features for failure sequences
- Maps node gradients to edges: `edge_score = |grad(src)| × |grad(dst)|`
- Edges connecting two high-gradient nodes are where failure signal propagates
- Output: ranked contagious edges, contagion network visualization
- Practical value: tells operators which dependencies to break during incidents

---

## Model Architecture — Why Each Choice

### Why GraphSAGE?

| Model | Aggregation | Scales? | Our choice |
|-------|------------|---------|------------|
| GCN | All neighbors | No — crashed at 3K nodes | Too expensive |
| **GraphSAGE** | **Sample k=15** | **Yes — all 4,900 nodes** | **What we use** |
| GAT | Attention (all neighbors) | Memory-heavy | Overkill — we encode importance via edge weights |

### Why Dynamic Edges?

Static ST-GNNs assume the graph doesn't change across the sequence. In reality, machines join and leave job collections every few minutes. Dynamic edges capture this: a machine that was isolated at t=1 but joined a busy collection at t=3 now has edges to its new neighbors at t=3 onward. The model sees the topology evolve, not just the features.

### Why 2 Layers?

1 layer = 1-hop (direct neighbors only). 2 layers = 2-hop (neighbors-of-neighbors). Cascading failures need 2-hop: if A fails → B is affected → C should see A's failure through B. 3+ layers = over-smoothing.

### Why GRU?

2 gates vs LSTM's 3. For short sequences (T=6), the extra gate doesn't help. GRU trains faster and uses less memory.

### Why Focal Loss?

~78% normal vs ~22% failing. Standard CE lets the model predict "normal" always and get 78% accuracy. Focal Loss (alpha=0.75, gamma=2.0) gives 3x weight to failures and down-weights easy predictions adaptively.

---

## Syllabus Topics Covered

| Syllabus Topic | Where in the project |
|----------------|---------------------|
| ML on non-Euclidean data | Entire project — cloud infra is a graph |
| GraphSAGE | `model.py` → `SpatialEncoder` with `SAGEConv` |
| Generalized neighborhood aggregation | SAGE's sample-and-aggregate framework |
| Stacking GNN layers | 2-layer SAGE with residual connections |
| **Dynamic graphs / spatial-temporal GNN** | **Per-timestep edge_index — truly dynamic graph** |
| GNN layer optimization | LayerNorm, dropout, residual connections |
| GNN Interpretability | Gradient-based critical nodes + edge contagion |
| Anomaly detection | Failure prediction as graph anomaly detection |
| Node embeddings | t-SNE visualization of learned representations |
| Encoder-decoder perspective | SAGE+GRU encoder, MLP decoder |
| Loss functions | Focal Loss for imbalanced node classification |
| Setting up graph datasets | Borg trace → dynamic weighted graph construction |

---

## Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | CUDA 4GB+ | RTX 4060 Mobile 8GB |
| RAM | 8 GB | 16 GB |
| CPU | Any modern | Ryzen 9 8945HS |

VRAM usage: ~17MB per forward pass (GraphSAGE sampling). Dynamic edges are built on CPU, only the edge tensor goes to GPU.

---

## Output Files

```
results/
├── confusion_matrix.png            # Standard
├── roc_pr_curves.png
├── critical_nodes.png
├── failure_propagation.png
├── embedding_tsne.png
├── mc_dropout_uncertainty.png      # Novel
├── edge_contagion_scores.png       # Novel
└── contagion_network.png           # Novel
```

---

## References

1. Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS. (GraphSAGE)
2. Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML. (MC Dropout)
3. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
4. Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder.* (GRU)
5. Google Cluster Data — Borg Traces 2019. https://github.com/google/cluster-data
6. Modeling and Analysis of Cascading Failures in Cloud Computing — ScienceDirect
7. GNN-Based Dynamic Multiqueue Optimization for Cloud Fault Tolerance — Wiley

---

*Built for the Interdisciplinary Deep Learning on Graphs course, PES University.*
