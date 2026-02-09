# GraphAgent Examples - All Features

Comprehensive collection of small, focused examples demonstrating every GraphAgent feature.

---

## Quick Start

Run any example with:
```bash
cd /Users/drahnreb/Downloads/adk-python
source venv/bin/activate
python -m contributing.samples.graph_examples.<example_name>.agent
```

---

## Examples Overview

### 🟢 Core Features

#### **01_basic** - Basic GraphAgent Workflow
Simple directed graph with nodes and edges.
```bash
python -m contributing.samples.graph_examples.01_basic.agent
```
**Demonstrates:**
- Creating a graph
- Adding nodes (agents)
- Adding edges (transitions)
- Executing workflow

---

#### **02_conditional_routing** - Conditional Routing
State-based routing decisions.
```bash
python -m contributing.samples.graph_examples.02_conditional_routing.agent
```
**Demonstrates:**
- Conditional edges
- State-based decisions
- Multiple routing paths
- Dynamic workflow control

---

#### **04_checkpointing** - Checkpointing & Resume
Automatic state persistence.
```bash
python -m contributing.samples.graph_examples.04_checkpointing.agent
```
**Demonstrates:**
- Automatic checkpointing
- State persistence
- Checkpoint metadata
- Execution path tracking

---

#### **05_interrupts_basic** - Basic Interrupts
Human-in-the-loop interrupts.
```bash
python -m contributing.samples.graph_examples.05_interrupts_basic.agent
```
**Demonstrates:**
- InterruptService integration
- AFTER interrupt timing
- Interrupt actions (continue, rerun, pause)
- Manual intervention points

---

#### **08_rewind** - Rewind Integration
Time-travel debugging.
```bash
python -m contributing.samples.graph_examples.08_rewind.agent
```
**Demonstrates:**
- Invocation tracking
- Rewinding to specific node
- State restoration
- Re-execution after rewind

---

### ⚡ Parallel Execution

#### **09_parallel_wait_all** - Parallel Execution (WAIT_ALL)
Concurrent node execution, wait for all.
```bash
python -m contributing.samples.graph_examples.09_parallel_wait_all.agent
```
**Demonstrates:**
- Parallel node execution
- WAIT_ALL join strategy
- Speedup vs sequential (2.25x)
- Event streaming from parallel nodes

**Output:**
```
[150ms] ✅ Fetched data from products_db (100ms)
[150ms] ✅ Fetched data from users_db (150ms)
[200ms] ✅ Fetched data from orders_db (200ms)

Total: ~200ms (vs 450ms sequential)
Speedup: ~2.25x
```

---

#### **10_parallel_wait_any** - Parallel Execution (WAIT_ANY)
Race condition, first-to-complete wins.
```bash
python -m contributing.samples.graph_examples.10_parallel_wait_any.agent
```
**Demonstrates:**
- Racing multiple data sources
- WAIT_ANY join strategy
- Automatic cancellation of slower nodes
- Cache-DB-API fallback pattern

**Output:**
```
[50ms] ✅ Data from CACHE (50ms)

Winner: Cache
Cancelled: Database, API
```

---

### 🔗 Combined Features

#### **14_parallel_rewind** - Parallel Execution + Rewind
Rewind works with parallel workflows!
```bash
python -m contributing.samples.graph_examples.14_parallel_rewind.agent
```
**Demonstrates:**
- Parallel + Rewind integration
- Invocation tracking in parallel groups
- Re-execution of entire parallel group
- State consistency across rewind

**Key Insight:**
- Rewind to parallel node → entire parallel group re-executes
- All branches get new invocations
- Deterministic re-execution

---

## Feature Matrix

| Example | Parallel | Rewind | Checkpoints | Interrupts | Callbacks |
|---------|----------|--------|-------------|------------|-----------|
| 01_basic | - | - | - | - | - |
| 02_conditional_routing | - | - | - | - | - |
| 04_checkpointing | - | - | ✅ | - | - |
| 05_interrupts_basic | - | - | - | ✅ | - |
| 08_rewind | - | ✅ | - | - | - |
| 09_parallel_wait_all | ✅ | - | - | - | - |
| 10_parallel_wait_any | ✅ | - | - | - | - |
| 14_parallel_rewind | ✅ | ✅ | - | - | - |

---

## Architectural Insights

### Parallel Execution Architecture

```
┌─────────────┐
│   validate  │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
       ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  fetch_A     │ │  fetch_B     │ │  fetch_C     │
│  (isolated)  │ │  (isolated)  │ │  (isolated)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │   aggregate  │
              │(merged state)│
              └──────────────┘
```

**Key Points:**
- Each branch has **isolated state** during execution
- No race conditions possible
- State **merged** after all branches complete
- Events **streamed** as branches complete (FIRST_COMPLETED)

---

### Rewind with Parallel Execution

```
1. Initial Execution:
   validate → (fetch_A || fetch_B || fetch_C) → aggregate

   Invocations created:
   - validate: ["inv_1"]
   - fetch_A: ["inv_2"]
   - fetch_B: ["inv_3"]
   - fetch_C: ["inv_4"]
   - aggregate: ["inv_5"]

2. Rewind to fetch_A (inv_2):
   Session state restored to BEFORE inv_2

3. Re-execution:
   (fetch_A || fetch_B || fetch_C) → aggregate

   New invocations:
   - fetch_A: ["inv_2", "inv_6"]
   - fetch_B: ["inv_3", "inv_7"]
   - fetch_C: ["inv_4", "inv_8"]
   - aggregate: ["inv_5", "inv_9"]
```

**Key Points:**
- Rewind works seamlessly with parallel groups
- Entire parallel group re-executes
- New invocations created on re-execution
- Deterministic behavior guaranteed

---

### State Isolation

**Problem:** Multiple nodes modifying same state → race conditions

**Solution:** Isolated state copies per branch

```python
# During parallel execution
for node in parallel_group.nodes:
    # Each branch gets ISOLATED copy
    branch_state = state.copy()

    # Modify branch state
    execute_node(node, branch_state)

# After all complete
merged_state = merge(all_branch_states)
```

**Benefits:**
- No race conditions
- Deterministic results
- Safe concurrent execution

---

## Performance Comparison

### Sequential vs Parallel (WAIT_ALL)

**Scenario:** Fetch from 3 sources (100ms, 150ms, 200ms each)

**Sequential:**
```
Total time = 100 + 150 + 200 = 450ms
```

**Parallel (WAIT_ALL):**
```
Total time = max(100, 150, 200) = 200ms
Speedup: 450ms / 200ms = 2.25x
```

**Parallel (WAIT_ANY):**
```
Total time = min(100, 150, 200) = 100ms
Speedup: 450ms / 100ms = 4.5x
```

---

## Common Patterns

### 1. Data Pipeline (WAIT_ALL)
Fetch data from multiple sources concurrently.
```python
ParallelNodeGroup(
    nodes=["fetch_users", "fetch_products", "fetch_orders"],
    join_strategy=JoinStrategy.WAIT_ALL
)
```

### 2. Cache-DB-API Fallback (WAIT_ANY)
Race multiple data sources, use fastest.
```python
ParallelNodeGroup(
    nodes=["from_cache", "from_db", "from_api"],
    join_strategy=JoinStrategy.WAIT_ANY
)
```

### 3. ML Model Ensemble (WAIT_N)
Run multiple models, proceed when N complete.
```python
ParallelNodeGroup(
    nodes=["model1", "model2", "model3"],
    join_strategy=JoinStrategy.WAIT_N,
    wait_n=2  # 2 out of 3
)
```

### 4. Interrupt-Driven Review
Human review after key nodes.
```python
InterruptConfig(
    mode=InterruptMode.AFTER,
    nodes=["draft", "review"]
)
```

### 5. Checkpoint-Resume Workflow
Long-running workflows with state persistence.
```python
GraphAgent(
    name="workflow",
    checkpointing=True
)
```

---

## Error Handling

### Parallel Error Policies

#### FAIL_FAST (default)
```python
ParallelNodeGroup(
    nodes=["task1", "task2", "task3"],
    error_policy=ErrorPolicy.FAIL_FAST
)
# One error → cancel all → raise exception
```

#### CONTINUE
```python
ParallelNodeGroup(
    nodes=["task1", "task2", "task3"],
    error_policy=ErrorPolicy.CONTINUE
)
# One error → continue others → log error
```

#### COLLECT
```python
ParallelNodeGroup(
    nodes=["task1", "task2", "task3"],
    error_policy=ErrorPolicy.COLLECT
)
# All errors → collect all → raise at end
```

---

## Testing

All examples have corresponding tests in:
```
tests/unittests/agents/
├── test_graph_agent.py (42 tests)
├── test_graph_rewind.py (10 tests)
└── test_graph_parallel.py (9 tests)

Total: 61 tests, all passing ✅
```

Run tests:
```bash
pytest tests/unittests/agents/test_graph_*.py -v
```

---

## Related Documentation

- **GraphAgent Design**: `contributing/docs/graph_agent_design.md`
- **Interrupt Architecture**: `contributing/docs/interrupt_service_architecture.md`
- **Checkpoint Architecture**: `contributing/docs/checkpoint_service_architecture.md`

---

## Next Steps

1. **Try the examples** - Run each one to see features in action
2. **Modify examples** - Change parameters, add nodes, experiment
3. **Combine features** - Mix parallel + rewind + checkpoints
4. **Build your workflow** - Use patterns for your use case

---

## Support

Questions? Check:
- Examples: `contributing/samples/graph_examples/`
- Tests: `tests/unittests/agents/test_graph_*.py`
- Source: `src/google/adk/agents/graph/`
