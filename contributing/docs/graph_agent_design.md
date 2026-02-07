# GraphAgent Design Document

**Author**: ADK Team
**Date**: 2026-01-25
**Status**: Experimental

---

## Motivation

ADK provides three workflow agents (SequentialAgent, ParallelAgent, LoopAgent) that execute in **fixed patterns**. While these can be composed, they cannot make **runtime decisions** based on state.

GraphAgent fills this gap by enabling **conditional routing** - the ability to choose different execution paths based on runtime state.

---

## Use Case: Why Existing Agents Are Insufficient

### Problem: Data Validation Pipeline with Retry

**Requirement**:
1. Validate input data
2. If valid → process data
3. If invalid → retry validation (max 3 times)
4. After max retries → route to error handler

### Attempt 1: SequentialAgent ❌

```python
sequential = SequentialAgent(sub_agents=[
    validator,
    processor,  # ❌ Always runs, even if validation failed
])
```

**Problem**: Cannot skip processor if validation fails.

### Attempt 2: Composition ❌

```python
# Try to handle errors with conditional logic
sequential = SequentialAgent(sub_agents=[
    validator,
    # ❌ No way to conditionally route here
    processor,
])
```

**Problem**: Sequential/Parallel/Loop agents don't support conditional edges.

### Solution: GraphAgent ✅

```python
graph = GraphAgent(name="pipeline")

# Add nodes
graph.add_node(GraphNode(name="validate", agent=validator))
graph.add_node(GraphNode(name="process", agent=processor))
graph.add_node(GraphNode(name="retry", agent=retry_handler))
graph.add_node(GraphNode(name="error", agent=error_handler))

# Conditional edges based on runtime state
graph.add_edge(
    "validate",
    "process",
    condition=lambda state: state.data.get("valid", False) is True
)
graph.add_edge(
    "validate",
    "retry",
    condition=lambda state: (
        not state.data.get("valid", False)
        and state.metadata.get("retry_count", 0) < 3
    )
)
graph.add_edge(
    "validate",
    "error",
    condition=lambda state: (
        not state.data.get("valid", False)
        and state.metadata.get("retry_count", 0) >= 3
    )
)
graph.add_edge("retry", "validate")  # Loop back
```

**This cannot be achieved with SequentialAgent, ParallelAgent, or LoopAgent.**

---

## Key Capabilities

### 1. Conditional Routing

Execute different paths based on runtime state:

```python
graph.add_edge(
    "node_a",
    "node_b",
    condition=lambda state: state.data["score"] > 0.8
)
graph.add_edge(
    "node_a",
    "node_c",
    condition=lambda state: state.data["score"] <= 0.8
)
```

### 2. Cyclic Execution (Loops)

Loop back to previous nodes:

```python
graph.add_edge("validate", "process")
graph.add_edge("process", "validate")  # Loop back for next iteration
```

Protected by `max_iterations` to prevent infinite loops.

### 3. State Management

Track state across node executions:

```python
class GraphState:
    data: Dict[str, Any]  # Node outputs
    metadata: Dict[str, Any]  # Execution metadata (iteration, path, etc.)
```

State can be updated by nodes via output mappers.

### 4. Checkpointing (Optional)

Save state after each node for resume capability:

```python
graph = GraphAgent(name="workflow", checkpointing=True)
# State automatically saved to session after each node
```

Resume from checkpoint:

```python
graph, checkpoint = await GraphAgent.resume_from_checkpoint(
    session_service, app_name, user_id, session_id
)
await graph.continue_from_checkpoint(checkpoint, session_service)
```

---

## Architecture

### Core Components

**GraphNode**: Wrapper around BaseAgent + metadata
- `name`: Node identifier
- `agent`: BaseAgent to execute
- `edges`: Conditional edges to other nodes
- `output_mapper`: Transform agent output to state

**GraphEdge**: Conditional transition
- `to_node`: Target node name
- `condition`: Predicate on GraphState (optional)

**GraphState**: Execution state
- `data`: Accumulated results from nodes
- `metadata`: Iteration count, execution path, etc.

### Execution Flow

1. Start at `start_node`
2. Execute current node's agent
3. Update state with node output (via output_mapper)
4. Evaluate edge conditions to find next node
5. If multiple conditions match → take first matching edge
6. If no conditions match → stop (must be at end_node)
7. Repeat until end_node reached or max_iterations exceeded

### Integration with ADK

GraphAgent is a proper **BaseAgent**:
- ✅ Extends `BaseAgent`
- ✅ Implements `_run_async_impl(ctx: InvocationContext)`
- ✅ Yields `Event` objects
- ✅ Uses `EventActions.state_delta` for state persistence
- ✅ Works with any `SessionService` (InMemory, SQLite, VertexAI)

GraphAgent uses **event-driven state**:
- State updates via `EventActions.state_delta`
- No manual state tracking
- Automatically persisted by SessionService
- Works with all ADK services

---

## Comparison to Alternatives

### vs SequentialAgent

| Feature | SequentialAgent | GraphAgent |
|---------|----------------|------------|
| Execution | Fixed sequence | Conditional routing |
| Loops | No | Yes (with max_iterations) |
| Branching | No | Yes (conditional edges) |
| State management | Basic | Rich (GraphState) |
| Use case | Simple pipelines | Complex workflows |

### vs ParallelAgent

| Feature | ParallelAgent | GraphAgent |
|---------|--------------|------------|
| Execution | All sub-agents in parallel | Conditional sequential/parallel |
| Dependencies | None | Explicit (via edges) |
| Conditional | No | Yes |
| Use case | Independent tasks | Dependent workflow |

### vs LangGraph

GraphAgent is **simpler** and **ADK-native**:
- No new concepts (uses BaseAgent, Events, SessionService)
- No custom state management (uses ADK primitives)
- No custom checkpointing (uses EventActions.state_delta)
- Direct integration with ADK ecosystem

---

## Implementation Details

### Size and Complexity

- **Core implementation**: ~1,231 lines
- **Tests**: 82 tests, 2,665 lines
- **Marked**: `@experimental` (API may change)

### Why This Size?

GraphAgent includes:
- Graph structure management (nodes, edges)
- Conditional routing logic
- State management (GraphState, reducers)
- Checkpointing integration
- Resume/continue from checkpoint
- Checkpoint management (list, delete, export, import)
- Error handling and validation

**Note**: Future refactoring may extract checkpointing utilities for reuse across all agents.

---

## Limitations and Future Work

### Current Limitations

1. **No parallel node execution**: Nodes execute sequentially
   - **Future**: Add parallel node groups

2. **Simple condition evaluation**: First matching edge wins
   - **Future**: Add priority/weight to edges

3. **Checkpoint management coupled to GraphAgent**
   - **Future**: Extract CheckpointUtils for all agents

4. **HITL not fully implemented**: `escalate=False` (notification only)
   - **Future**: Integrate with Runner-level HITL handling

### Planned Enhancements

- Extract checkpointing to reusable utilities
- Add parallel node execution
- Enhance conditional routing (edge priorities)
- Full HITL integration
- D3 visualization improvements

---

## Usage Guidance

### When to Use GraphAgent

**Good fit**:
- ✅ Workflows with conditional routing
- ✅ Multi-step pipelines with error recovery
- ✅ Iterative refinement (loops)
- ✅ State-dependent execution paths

**Not recommended**:
- ❌ Simple sequential workflows (use SequentialAgent)
- ❌ Independent parallel tasks (use ParallelAgent)
- ❌ Simple loops (use LoopAgent)

### Getting Started

See `contributing/samples/graph_agent_basic/agent.py` for a complete example.

---

## API Stability

**Status**: `@experimental`

**What this means**:
- API may change in future releases
- Use in production at your own risk
- Provide feedback to help stabilize API
- Migration guide will be provided when API changes

**When will it graduate?**:
- After 2-3 release cycles
- After real-world usage validation
- After addressing feedback
- When API is proven stable

---

## References

- Source: `src/google/adk/agents/graph_agent.py`
- Tests: `tests/unittests/agents/test_graph_agent.py`
- Sample: `contributing/samples/graph_agent_basic/agent.py`
- Similar: LangGraph, Apache Airflow DAGs
