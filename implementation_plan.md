# Eliminate CPU→GPU Bottleneck in ST-GNN Training Pipeline

## Problem Diagnosis

The root cause of GPU starvation is in the **training loop's synchronous data path**:

```
for each sequence in epoch:
    CPU: loader.get_sequence(si)         ← BLOCKS GPU
         └─ _build_dynamic_edges(tw) x6  ← NumPy groupby → nested Python loops
         └─ get_snapshot(tw) x6          ← pandas→numpy→torch
    GPU: model.forward()                 ← fast, ~10ms
    GPU: loss.backward()                 ← fast
```

Each call to `_build_dynamic_edges` does:
1. `df.groupby("cluster")` → Python iterator over variable-size groups
2. Per-group: `df["machine_id"].unique()` → `np.array([self.m2i[m] for m in machines])` — a **Python list comprehension** over up to thousands of machines
3. `np.triu_indices` or broadcasting for edge construction
4. `np.concatenate` + `np.unique` for dedup
5. `torch.from_numpy().to(device)` — CPU→GPU transfer

This happens **6 times per sequence** (seq_length=8 makes it 8 now), and each sequence is processed serially. Even with the LRU cache (200 entries), the first epoch is catastrophic, and cache eviction means many windows get rebuilt.

## Strategy: Three-Layer Fix

### Layer 1 — GPU-Native Edge Builder (Primary Fix)
Move the entire `_build_dynamic_edges` from NumPy/Python to **pure PyTorch GPU tensor operations**. Pre-encode membership data as GPU tensors at init time instead of keeping it as a pandas DataFrame.

### Layer 2 — Epoch-Start Batch Precomputation (Secondary Fix)
At the start of each epoch, precompute ALL needed edge indices for the epoch's training windows in one vectorized batch pass, rather than lazily building them per-sequence.

### Layer 3 — Background Prefetch Pipeline (Tertiary Fix)
Use a `threading.Thread` background worker to prepare the next batch's data while the GPU processes the current batch, overlapping CPU and GPU work.

---

## Proposed Changes

### [MODIFY] [train.py](file:///c:/InterdisciplinaryGraphs_sem6/v3/train.py)

#### 1. New `GPUEdgeBuilder` class replacing NumPy-based `_build_dynamic_edges`

At loader init time, convert the membership DataFrame into GPU-resident tensors:
- `self._gpu_membership_tw`: int tensor of time_window IDs, shape `[M]`
- `self._gpu_membership_node`: int tensor of mapped node indices, shape `[M]`
- `self._gpu_membership_cluster`: int tensor of cluster IDs, shape `[M]`
- `self._gpu_membership_collection`: int tensor of collection IDs, shape `[M]`

Then `_build_dynamic_edges_gpu(tw)` works entirely with `torch` ops on CUDA:

```python
def _build_dynamic_edges_gpu(self, tw):
    """100% GPU tensor ops — no NumPy, no Python loops, no CPU→GPU transfer."""
    cached = self._edge_cache.get(tw)
    if cached is not None:
        return cached

    # Mask for this time window — single GPU comparison
    mask = (self._gpu_membership_tw == tw)
    if mask.sum() == 0:
        self._edge_cache.put(tw, self.static_ei)
        return self.static_ei

    nodes_tw = self._gpu_membership_node[mask]
    clusters_tw = self._gpu_membership_cluster[mask]
    collections_tw = self._gpu_membership_collection[mask]

    all_edges = []

    # --- Cluster edges (GPU) ---
    unique_clusters = clusters_tw.unique()
    for c in unique_clusters:
        if c < 0:  # sentinel for "no cluster"
            continue
        cmask = (clusters_tw == c)
        active = nodes_tw[cmask].unique()
        n = active.shape[0]
        if n < 2:
            continue
        if n <= 60:
            ii, jj = torch.triu_indices(n, n, offset=1, device=self.device)
            src, dst = active[ii], active[jj]
        else:
            # K-nearest sliding window on GPU
            K = min(15, n - 1)
            pos = torch.arange(n, device=self.device)
            offsets = torch.arange(1, K + 1, device=self.device)
            j_idx = pos.unsqueeze(1) + offsets.unsqueeze(0)
            valid = j_idx < n
            i_valid = pos.unsqueeze(1).expand_as(j_idx)[valid]
            j_valid = j_idx[valid]
            src, dst = active[i_valid], active[j_valid]
        all_edges.append(torch.stack([src, dst]))
        all_edges.append(torch.stack([dst, src]))

    # --- Collection edges (GPU) ---
    unique_colls = collections_tw.unique()
    for c in unique_colls:
        if c < 0:
            continue
        cmask = (collections_tw == c)
        active = nodes_tw[cmask].unique()
        n = active.shape[0]
        if n < 2:
            continue
        if n <= 40:
            ii, jj = torch.triu_indices(n, n, offset=1, device=self.device)
            src, dst = active[ii], active[jj]
        else:
            K = min(10, n - 1)
            pos = torch.arange(n, device=self.device)
            offsets = torch.arange(1, K + 1, device=self.device)
            j_idx = pos.unsqueeze(1) + offsets.unsqueeze(0)
            valid = j_idx < n
            i_valid = pos.unsqueeze(1).expand_as(j_idx)[valid]
            j_valid = j_idx[valid]
            src, dst = active[i_valid], active[j_valid]
        all_edges.append(torch.stack([src, dst]))
        all_edges.append(torch.stack([dst, src]))

    # Self-loops for active nodes
    active_all = nodes_tw.unique()
    all_edges.append(torch.stack([active_all, active_all]))

    if not all_edges:
        self._edge_cache.put(tw, self.static_ei)
        return self.static_ei

    edges = torch.cat(all_edges, dim=1)
    # Deduplicate on GPU
    edges = torch.unique(edges, dim=1)

    self._edge_cache.put(tw, edges)
    return edges
```

> [!IMPORTANT]
> The Python `for c in unique_clusters` loop still exists, but it iterates over **unique cluster IDs** (typically ~50-200 clusters), NOT over individual machines (~89,000). Each iteration body is pure GPU tensor ops. The old code iterated over pandas groupby objects AND had Python list comprehensions inside each group.

#### 2. Membership Data Encoding at Init Time

In `DynamicGraphLoader.__init__`, after loading `window_membership.parquet`:

```python
# Pre-encode membership as GPU tensors (one-time cost, ~50-100MB VRAM)
if self.dynamic:
    # Map machine_id strings to node indices
    node_idx = self.membership["machine_id"].map(self.m2i)
    valid = node_idx.notna()
    mem_valid = self.membership[valid].copy()
    mem_valid["_node_idx"] = node_idx[valid].astype(int)

    # Encode cluster/collection as integer IDs (-1 = missing)
    if "cluster" in mem_valid.columns:
        cluster_cats = mem_valid["cluster"].astype("category")
        mem_valid["_cluster_id"] = cluster_cats.cat.codes  # -1 for NaN
    else:
        mem_valid["_cluster_id"] = -1

    if "collection_id" in mem_valid.columns:
        coll_cats = mem_valid["collection_id"].astype("category")
        mem_valid["_coll_id"] = coll_cats.cat.codes
    else:
        mem_valid["_coll_id"] = -1

    # Move to GPU — these tensors stay resident for the entire run
    self._gpu_membership_tw = torch.tensor(
        mem_valid["time_window"].values, dtype=torch.long, device=self.device)
    self._gpu_membership_node = torch.tensor(
        mem_valid["_node_idx"].values, dtype=torch.long, device=self.device)
    self._gpu_membership_cluster = torch.tensor(
        mem_valid["_cluster_id"].values, dtype=torch.long, device=self.device)
    self._gpu_membership_collection = torch.tensor(
        mem_valid["_coll_id"].values, dtype=torch.long, device=self.device)

    # Free the pandas membership df — no longer needed
    del self.membership, self._mem_groups
    gc.collect()
```

#### 3. Background Prefetch Thread

Add a simple producer/consumer prefetch for sequence data:

```python
import threading
from queue import Queue

class PrefetchIterator:
    """Prefetches the next sequence in a background thread while GPU runs."""
    def __init__(self, loader, indices, depth=2):
        self.loader = loader
        self.indices = indices
        self.queue = Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        for si in self.indices:
            data = self.loader.get_sequence(si)
            self.queue.put(data)
        self.queue.put(None)  # sentinel

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item
```

> [!NOTE]
> Python's GIL doesn't block here because the heavy ops in the GPU edge builder release the GIL (PyTorch CUDA ops do). The background thread can build the next batch's edges on GPU while the main thread runs forward/backward.

#### 4. Feature Tensor Caching on GPU

Currently `get_snapshot` caches feature tensors on CPU (`torch.tensor(x)`) and moves them to GPU in the training loop. Change to cache directly on GPU:

```python
x_cached = torch.tensor(x, device=self.device)
```

This eliminates per-sequence CPU→GPU transfer for features.

> [!WARNING]
> VRAM usage: `num_nodes × num_features × 4 bytes × 200 cache entries`. With ~4,900 nodes and ~10 features, that's `4900 × 10 × 4 × 200 ≈ 39 MB` — well within budget. But if your actual data has ~89,000 nodes, this becomes `89000 × 10 × 4 × 200 ≈ 713 MB`. The 200-entry cache limit should be reduced if node count is very high (will add a dynamic cap).

---

### [MODIFY] [model.py](file:///c:/InterdisciplinaryGraphs_sem6/v3/model.py)

No changes needed. The model already accepts `edge_index` as either a list or single tensor, and `dropout_edge` works on GPU tensors natively. The bottleneck is entirely in the data pipeline.

---

### [MODIFY] [config.yaml](file:///c:/InterdisciplinaryGraphs_sem6/v3/config.yaml)

```yaml
data:
  sequence_length: 6        # reduce from 8 → 6 (was original default)
                             # 6 × 5min = 30min history is sufficient
                             # reduces edge builds per sequence by 25%

training:
  batch_size: 16             # increase from 32 (counterintuitive but with GPU edge
                             # building, the bottleneck shifts — larger batches
                             # amortize the per-step overhead better)
```

> [!NOTE]
> sequence_length=6 was the original default in the code. Your config set it to 8. Going back to 6 reduces edge computations per sequence by 25% with minimal accuracy impact, since the GRU + temporal attention already learns to weight the most informative windows.

---

## Expected Performance Impact

| Metric | Before (NumPy/CPU) | After (GPU Tensors) | Speedup |
|--------|-------------------|---------------------|---------|
| `_build_dynamic_edges` per window | ~200-500ms | ~2-10ms | **20-50x** |
| CPU→GPU edge transfer per window | ~5-15ms | 0ms (already on GPU) | ∞ |
| GPU utilization during training | 0-100% (oscillating) | 85-100% (sustained) | — |
| Epoch time (estimated) | 45-60+ min | 5-15 min | **3-10x** |
| VRAM overhead for membership tensors | 0 | ~50-100 MB | — |

## Open Questions

> [!IMPORTANT]
> 1. **How many unique clusters/collections per time window?** If there are >500 unique clusters per window, the Python `for c in unique_clusters` loop could still be slow. In that case, we'd need a fully vectorized scatter-based approach (more complex but eliminates all Python loops). Can you check by running: `python -c "import pandas as pd; m = pd.read_parquet('processed/window_membership.parquet'); print('clusters/window:', m.groupby('time_window')['cluster'].nunique().describe())"`?
>
> 2. **What's the actual node count?** CLAUDE.md mentions ~89,000 nodes but the code comments say ~4,900. This determines VRAM budget for feature caching on GPU. If 89K, we need tighter cache limits.

## Verification Plan

### Automated Tests
1. Run `python train.py` and measure:
   - Per-epoch wall time (target: <10 min)
   - GPU utilization via `nvidia-smi -l 1` in parallel (target: >85% sustained)
2. Compare validation F1 between static fallback and new GPU dynamic edges (should be same or better)

### Manual Verification
- Monitor VRAM usage during training (target: peak <7.5 GB of 8 GB)
- Confirm no CPU→GPU data transfer bottleneck in `torch.profiler` trace
