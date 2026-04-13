# Project: ST-GNN for Cascading Cloud Failures

## Project Context
This project models cascading resource failures in cloud infrastructure using Spatio-Temporal Graph Deep Learning. The architecture utilizes a 2-layer GraphSAGE spatial encoder and a GRU temporal encoder to process dynamic per-window topologies.

## Hardware & Environment Profile
**Target OS:** Windows 11 (Run natively in PowerShell, NO WSL)
**Target Machine:** ASUS ROG G14 2024
- **CPU:** AMD Ryzen 9 8945HS
- **GPU:** NVIDIA RTX 4060 Mobile (8GB VRAM)
- **RAM:** 16GB
- **Storage:** 512GB SSD

## STRICT BUDGETARY CONSTRAINTS
- **Remaining Budget:** $1.00 USD.
- **Protocol:** You are forbidden from executing more than 2 consecutive turns without a manual cost check.
- **Alert:** If a single request is estimated to exceed 100k input tokens, you MUST ask for permission before proceeding.
- **Reporting:** Run the `/cost` command after every code modification and report the remaining session balance.


**CRITICAL Execution Directives:**
1. **Windows Compatibility:** All code MUST run natively in Windows PowerShell. Do not assume Linux environments. Explicitly bypass or remove `torch.compile` as the Triton backend is not natively supported on Windows.
2. **Time Constraint (30-60 mins max):** The entire training script MUST finish within 30 to 60 minutes natively on Windows. Optimize `train.py` and `config.yaml` to achieve this (e.g., utilize the halved dataset, optimize batch sizes).
3. **GPU & VRAM Limits (8GB):** All training and heavy tensor operations MUST be explicitly routed to the GPU (`cuda`). Maintain strict VRAM boundaries utilizing PyG's `dropout_edge` and gradient accumulation. **CRITICAL: Do not exceed `batch_size: 2` or `hidden_dim: 32` for this GNN architecture — exceeding these limits causes OOM crashes on the RTX 4060 Laptop (8GB VRAM).**
4. **RAM Limits (16GB):** Enforce strict garbage collection (`gc.collect()`) and bounded LRU caching (max 200 windows). Do not load full historical edge lists into memory simultaneously.
5. **STRICT TOKEN & FILE LIMITS:** You are restricted to a strict token budget. 
   - DO NOT attempt to read `borg_traces_data.csv` OR `borg_traces_half.csv`.
   - DO NOT read files in the `processed/` directory.
   - For the current debugging phase, you are ONLY allowed to read and modify `model.py`, `train.py`, and `config.yaml`. 
   - DO NOT run broad workspace searches or read `evaluate.py` or `preprocess.py` unless explicitly instructed.
   - Keep your tool usage minimal to save tokens.

## System Safety (Windows-Specific)
- **DataLoader Workers:** `num_workers` MUST be set to `0` or `1`. Using `num_workers: 4` or higher causes a **Shared File Mapping error (1455)** on Windows that crashes the OS after approximately 10 epochs. This is a hard constraint — do not increase workers for performance.
- **AMP API:** Always use the modern `torch.amp` namespace (e.g., `torch.amp.autocast('cuda', ...)`). The deprecated `torch.cuda.amp` calls must be avoided as they will be removed in future PyTorch versions.

## Model Training Guidelines
- **Class Imbalance:** The Borg Trace dataset has an extreme class imbalance (~0.0001 failure rate). Standard cross-entropy loss will cause the model to predict all-negative trivially.
- **Loss Function:** Always use **Focal Loss** with `focal_alpha: 0.99` and `focal_gamma: 3.0` to ensure the model prioritizes rare failure events. Do not revert to BCE or cross-entropy without explicit justification.

## Current Objectives & Debugging Focus
- **Fix Training Bottlenecks:** Optimize the ST-GNN architecture so it runs fast and stable natively on Windows PowerShell.
- **Error Resolution:** Identify, debug, and fix any runtime errors, tensor shape mismatches, or device allocation bugs between the dynamic edge loader and the spatial encoder in `train.py` and `model.py`.
- **Code Generation Style:** Provide complete, fully-implemented code blocks rather than high-level pseudocode or purely theoretical explanations. Ensure all logical flows are explicitly written out to prioritize readability and direct implementation.

## Plugins & Tooling
- **Code Review Graph Plugin:** A pre-built dependency graph is stored at `./.code-review-graph/graph.db`. Before any refactoring, suggest running a graph-review command to check cross-file dependencies between `model.py`, `train.py`, and `config.yaml`. Do not restructure interfaces without consulting the graph first.

## Current Bottleneck Notes
- GPU: RTX 4060 Laptop (8GB VRAM) sitting idle at 27-28W.
- CPU: Ryzen 9 is bottlenecking during dynamic edge construction.
- Goal: Move all graph processing to GPU/CUDA to hit 50W+ GPU power draw.
- File Constraint: Ignore `processed/` and all `.csv` files for token efficiency.

## Common Commands
- **Preprocess:** `python preprocess.py`
- **Sanity Check:** `python check_data.py`
- **Train:** `python train.py`
- **Evaluate:** `python evaluate.py`

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
