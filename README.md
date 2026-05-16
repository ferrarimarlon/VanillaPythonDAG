# VanillaPythonDAG

A minimal Python orchestrator built on graphs, explicit dependencies, and pure functions.  
No framework. No magic.

---

## The idea

Most pipelines break in the same way: one node fails, and everything downstream either crashes silently or runs on bad data.

vanillaDAG treats that as a graph problem.

Each step is a function. Dependencies between steps form a directed acyclic graph. The engine runs a topological sort, executes each node in order, and propagates failures downstream — so broken data never reaches production.

---

## How the graph runs

```
extract_customers
├── validate_customers → dim_customers ──────────────────────┐
│                                                             ├──→ fact_sales → publish_kpis
├── extract_orders ───────────────────────────────────────────┘
│         └──────────→ quality_check ✗ → downstream_blocked 🔒
└── extract_order_items ────────────────────────────────────────┘
```

`quality_check` detects an **8.1% late delivery rate** against a 5% SLA threshold — fails, blocks its downstream, and leaves the rest of the pipeline untouched.

```
  [RUNNING]  extract_customers
  [SUCCESS]  extract_customers       ✓
  [RUNNING]  extract_orders
  [SUCCESS]  extract_orders          ✓
  [RUNNING]  quality_check
  [FAILED]   quality_check           ✗  — 8.1% late delivery rate exceeds 5% threshold
  [BLOCKED]  downstream_blocked      🔒
  [RUNNING]  fact_sales
  [SUCCESS]  fact_sales              ✓
  [RUNNING]  publish_kpis
  [SUCCESS]  publish_kpis            ✓
```

Failure is isolated. The rest runs.

---

## Engine internals

**Topological ordering** — Kahn's algorithm (BFS + in-degree tracking).  
Every node waits for all its dependencies before it's eligible to run.

```python
in_degree = {node: len(node.upstream) for node in self.nodes.values()}
queue     = deque(node for node, deg in in_degree.items() if deg == 0)

while queue:
    node = queue.popleft()
    ordered.append(node)
    for child in node.downstream:
        in_degree[child] -= 1
        if in_degree[child] == 0:
            queue.append(child)
```

**Failure propagation** — BFS from the failed node.  
All descendants are marked `BLOCKED` before execution reaches them.

```python
def _propagate_failure(self, failed_node: Node) -> None:
    queue = deque(failed_node.downstream)
    while queue:
        node = queue.popleft()
        if node.status not in (Status.SUCCESS, Status.FAILED):
            node.status = Status.BLOCKED
        queue.extend(node.downstream)
```

**Shared context** — a plain dict passed to every node.  
Nodes read what they need, write what they produce. No global state, no coupling.

```python
def fact_sales(ctx: dict):
    dim    = ctx["dim_customers"]
    orders = ctx["orders"]
    items  = ctx["order_items"]
    # ... merge, aggregate, write back
    ctx["fact_sales"] = result
```

---

## Data

Real [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) Brazilian e-commerce dataset.  
99,441 orders · 112,650 items · R$ 15,843,553.24 gross revenue across 27 states.

```
STATE      REVENUE (R$)   ORDERS     LATE
──────── ────────────── ──────── ────────
SP         5,921,678.12   41,746    5.9%
RJ         2,129,681.98   12,852   13.5%
MG         1,856,161.49   11,635    5.6%
RS           885,826.76    5,466    7.1%
PR           800,935.44    5,045    5.0%
BA           611,506.67    3,380   14.0%
SC           610,213.60    3,637    9.8%
```

---

## Run

```bash
pip install pandas pytest
```

**Run the pipeline directly:**

```bash
python src/pipeline.py
```

**Run both scenarios** (default failure + reprocess recovery):

```bash
python scenarios.py           # both in sequence
python scenarios.py default   # scenario 1 only — strict SLA, quality_check fails
python scenarios.py reprocess # scenario 2 only — relaxed SLA, reprocess succeeds
```

**Run the tests:**

```bash
python -m pytest tests/ -v
```

CSVs are downloaded on first run and cached in `data/`.

---

## Configuration

Every threshold and sanity check lives in `config.py` as a plain dataclass — no env vars, no YAML, no hidden defaults buried in node logic.

```python
@dataclass
class PipelineConfig:
    # quality thresholds
    max_late_delivery_rate: float = 0.05

    # sanity checks — fail fast if a dataset is suspiciously small
    min_customers:   int = 50_000
    min_orders:      int = 50_000
    min_order_items: int = 50_000

    # required columns — catch schema drift before transforms run
    required_customer_columns: list[str] = ["customer_id", "customer_state", ...]
    required_order_columns:    list[str] = ["order_id", "customer_id", "order_status", ...]
    required_item_columns:     list[str] = ["order_id", "price", "freight_value"]
```

The config is injected into the shared context before the pipeline runs:

```python
dag.context["config"] = PipelineConfig()
dag.run()
```

Every node that needs a threshold or column list reads it from `ctx["config"]`. Override any value at runtime without touching pipeline logic:

```python
dag.context["config"] = PipelineConfig(
    max_late_delivery_rate=0.10,  # more lenient SLA
    min_orders=10_000,            # smaller dataset in staging
)
```

**What each setting guards:**

| Field | Default | Enforced in |
|---|---|---|
| `max_late_delivery_rate` | `5%` | `quality_check` |
| `min_customers` | `50,000` | `extract_customers` |
| `min_orders` | `50,000` | `extract_orders` |
| `min_order_items` | `50,000` | `extract_order_items` |
| `required_*_columns` | per dataset | each extract node |

---

## File structure

```
vanillaDAG/
├── src/
│   ├── vanilla_dag.py   # engine: Node, DAG, topological sort, failure propagation
│   ├── pipeline.py      # Olist sales pipeline
│   └── config.py        # all thresholds and sanity checks in one place
├── tests/
│   ├── test_engine.py   # engine unit tests (topological sort, propagation, reprocess)
│   └── test_pipeline.py # pipeline node tests (each function in isolation)
├── scenarios.py         # runnable demo: default failure + reprocess recovery
├── conftest.py          # pytest path setup
└── data/                # cached CSVs (auto-created on first run)
```

---

## Tests

**40 tests** covering every function in the engine and pipeline, with no network calls — all pipeline tests run against in-memory synthetic DataFrames.

```bash
python -m pytest tests/ -v
```

```
tests/test_engine.py::test_topological_sort_linear                  PASSED
tests/test_engine.py::test_topological_sort_diamond                 PASSED
tests/test_engine.py::test_topological_sort_cycle_raises            PASSED
tests/test_engine.py::test_run_all_success                          PASSED
tests/test_engine.py::test_run_failure_blocks_direct_downstream     PASSED
tests/test_engine.py::test_run_failure_blocks_transitive_downstream PASSED
tests/test_engine.py::test_run_failure_does_not_block_sibling_branch PASSED
tests/test_engine.py::test_context_is_shared_between_nodes         PASSED
tests/test_engine.py::test_reprocess_resets_target_and_ancestors   PASSED
tests/test_engine.py::test_reprocess_leaves_unrelated_nodes_intact PASSED
...
40 passed in 0.15s
```

**`test_engine.py`** — pure engine logic, no I/O:

| Group | What's tested |
|---|---|
| Topological sort | linear chain, diamond graph, cycle detection, independent nodes |
| Run | all success, single failure, transitive blocking, sibling isolation, root node |
| Context | shared dict flows between nodes, pre-loaded values survive |
| Reprocess | ancestor collection, unrelated nodes untouched, failure still blocks, multiple targets |

**`test_pipeline.py`** — each node in isolation, `_load` mocked out:

| Node | What's tested |
|---|---|
| `extract_*` | loads DataFrame, raises on missing columns, raises below min rows |
| `validate_customers` | drops null customer_id / state, passes when all valid |
| `dim_customers` | builds index by customer_id, deduplicates |
| `quality_check` | fails above threshold, passes below, on-time set excludes late orders |
| `fact_sales` | one row per state, correct revenue per state, correct late flag, order totals |
| `publish_kpis` | runs without error, prints correct totals |
| `PipelineConfig` | default values, field-level overrides |

---

## Scenarios

`scenarios.py` demonstrates both the failure and recovery paths end-to-end against the real Olist dataset:

**Scenario 1 — default run (strict SLA: 5%)**  
The pipeline runs with the default config. `quality_check` detects 8.1% late deliveries, exceeds the 5% threshold, fails, and blocks `downstream_blocked`. The rest of the graph completes normally.

**Scenario 2 — reprocess after relaxing SLA to 10%**  
The SLA threshold is updated in config. `dag.reprocess("downstream_blocked")` walks upstream, resets only the 4 affected nodes, and re-executes them. `quality_check` now passes at 10%, `downstream_blocked` runs, and the full graph ends green.

```bash
python scenarios.py           # both scenarios in sequence
python scenarios.py default   # scenario 1 only
python scenarios.py reprocess # scenario 2 only (runs scenario 1 first to create the failed state)
```

---

## Reprocessing

When a node fails, you don't re-run the entire pipeline. You reprocess just the affected node — vanillaDAG walks upstream to collect every transitive dependency, resets them to `PENDING`, and re-executes only that subgraph. Everything else keeps its previous `SUCCESS` result.

```python
dag.context["config"] = PipelineConfig(max_late_delivery_rate=0.10)
dag.reprocess("downstream_blocked")
```

```
=== olist_sales_pipeline [reprocess: downstream_blocked] ===
    resetting 4 node(s): ['extract_customers', 'extract_orders', 'quality_check', 'downstream_blocked']

  [RUNNING]  extract_customers   ← re-executed
  [SUCCESS]  extract_customers
  [RUNNING]  extract_orders      ← re-executed
  [SUCCESS]  extract_orders
  [RUNNING]  quality_check       ← re-executed, now passes at 10% threshold
  [SUCCESS]  quality_check
  [RUNNING]  downstream_blocked  ← finally runs
  [SUCCESS]  downstream_blocked
```

`validate_customers`, `dim_customers`, `extract_order_items`, `fact_sales`, and `publish_kpis` were not touched — their results carried over from the previous run.

The final report reflects the full graph:

```
  extract_customers     ✓ SUCCESS
  validate_customers    ✓ SUCCESS
  extract_orders        ✓ SUCCESS
  extract_order_items   ✓ SUCCESS
  dim_customers         ✓ SUCCESS
  quality_check         ✓ SUCCESS
  fact_sales            ✓ SUCCESS
  downstream_blocked    ✓ SUCCESS
  publish_kpis          ✓ SUCCESS
```

---

## Node status

| Status    | Meaning                                      |
|-----------|----------------------------------------------|
| `PENDING` | waiting to be scheduled                      |
| `RUNNING` | currently executing                          |
| `SUCCESS` | completed without errors                     |
| `FAILED`  | raised an exception                          |
| `BLOCKED` | an upstream dependency failed or was blocked |

---

