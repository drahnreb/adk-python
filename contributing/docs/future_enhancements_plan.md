# GraphAgent Future Enhancements - Implementation Plan

This document outlines the roadmap for future enhancements to GraphAgent, CheckpointService, and InterruptService.

---

## 1. Parallel Node Execution with Dependency Management

**Status**: Not started
**Priority**: High
**Complexity**: Medium
**Estimated effort**: 2-3 weeks

### Motivation

Currently, GraphAgent executes nodes sequentially. Many workflows have independent nodes that could run in parallel to improve throughput.

### Design

**Parallel Node Groups**:
```python
graph = GraphAgent(name="parallel_graph")

# Define parallel group (nodes can execute concurrently)
graph.add_parallel_group(
    nodes=["fetch_user", "fetch_products", "fetch_orders"],
    join_node="merge_data",  # Wait for all to complete
)

# Dependency-based parallelism
graph.add_node("fetch_user", agent=fetch_user_agent, dependencies=[])
graph.add_node("fetch_products", agent=fetch_products_agent, dependencies=[])
graph.add_node("merge_data", agent=merge_agent, dependencies=["fetch_user", "fetch_products"])
```

**Implementation Requirements**:
1. **DAG Analysis**: Detect independent nodes via topological sort
2. **Concurrent Execution**: Use `asyncio.gather()` for parallel node execution
3. **Join Semantics**: Wait for all parallel nodes before proceeding
4. **Error Handling**: If any parallel node fails, cancel others or continue based on policy
5. **State Merging**: Combine outputs from parallel nodes into single state

**API Changes**:
```python
class GraphAgent(BaseAgent):
    parallel_execution: bool = False  # Enable parallel execution
    parallel_policy: ParallelPolicy = ParallelPolicy.WAIT_ALL  # or WAIT_ANY, WAIT_N

class ParallelPolicy(Enum):
    WAIT_ALL = "all"      # Wait for all parallel nodes
    WAIT_ANY = "any"      # Continue when first node completes
    WAIT_N = "n"          # Wait for N nodes (specify via parallel_threshold)
```

**Testing**:
- Test parallel execution order (verify concurrency)
- Test join semantics (all nodes complete before join)
- Test error propagation (fail-fast vs continue)
- Test state merging (multiple outputs combined)

**Breaking Changes**: None (opt-in via `parallel_execution=True`)

---

## 2. Enhanced Conditional Routing (Edge Priorities & Weights)

**Status**: Not started
**Priority**: Medium
**Complexity**: Low
**Estimated effort**: 1 week

### Motivation

Currently, first matching edge wins. Complex workflows need weighted routing or priority-based decisions.

### Design

**Edge Priorities**:
```python
# Higher priority edges evaluated first
graph.add_edge("router", "high_priority_path", priority=10, condition=lambda s: s.urgent)
graph.add_edge("router", "normal_path", priority=5, condition=lambda s: not s.urgent)
graph.add_edge("router", "fallback", priority=1)  # Always true, lowest priority
```

**Edge Weights (Probabilistic Routing)**:
```python
# For A/B testing, load balancing
graph.add_edge("router", "variant_a", weight=0.7)  # 70% traffic
graph.add_edge("router", "variant_b", weight=0.3)  # 30% traffic
```

**Implementation Requirements**:
1. Add `priority: int` and `weight: float` to `GraphEdge`
2. Sort edges by priority before evaluation
3. Add `RoutingMode` enum: FIRST_MATCH (current), WEIGHTED, HIGHEST_PRIORITY
4. Implement weighted random selection for WEIGHTED mode

**API Changes**:
```python
class GraphEdge:
    priority: int = 0       # Higher = evaluated first
    weight: float = 1.0     # For weighted routing

class RoutingMode(Enum):
    FIRST_MATCH = "first"         # Current behavior
    WEIGHTED = "weighted"         # Random selection by weight
    HIGHEST_PRIORITY = "priority" # Priority-based
```

**Testing**:
- Test priority ordering (high priority evaluated first)
- Test weighted distribution (verify statistical distribution over 1000 runs)
- Test fallback edge (priority=0 always matches if no others do)

**Breaking Changes**: None (default priority=0 maintains current behavior)

---

## 3. Extract Checkpointing to Reusable Utilities

**Status**: Not started
**Priority**: High
**Complexity**: Medium
**Estimated effort**: 2 weeks

### Motivation

CheckpointService is currently designed for GraphAgent. Other agents (LoopAgent, SequentialAgent) could benefit from checkpointing.

### Design

**Checkpoint Mixin Pattern**:
```python
# Base mixin for any agent
class CheckpointableMixin:
    """Mixin to add checkpointing to any agent."""

    checkpoint_service: Optional[CheckpointService] = None
    auto_checkpoint: bool = False
    checkpoint_interval: int = 10  # Checkpoint every N iterations

    async def _create_checkpoint(self, ctx: InvocationContext, metadata: dict):
        """Create checkpoint with current state."""
        if self.checkpoint_service:
            return await self.checkpoint_service.create_checkpoint(
                session=ctx.session,
                description=f"{self.name} checkpoint",
                metadata=metadata,
            )

    async def _restore_checkpoint(self, ctx: InvocationContext, checkpoint_id: str):
        """Restore from checkpoint."""
        if self.checkpoint_service:
            return await self.checkpoint_service.restore_checkpoint(
                session=ctx.session,
                checkpoint_id=checkpoint_id,
            )

# Usage in any agent
class LoopAgent(BaseAgent, CheckpointableMixin):
    async def run_async(self, ctx: InvocationContext):
        for i in range(self.max_iterations):
            # Auto-checkpoint every N iterations
            if self.auto_checkpoint and i % self.checkpoint_interval == 0:
                await self._create_checkpoint(ctx, {"iteration": i})

            # Execute loop body
            ...
```

**Checkpoint Utilities**:
```python
# src/google/adk/checkpoints/utils.py

class CheckpointUtils:
    """Utility functions for checkpointing."""

    @staticmethod
    def serialize_state(state: Any) -> dict:
        """Serialize agent state to JSON-compatible dict."""
        ...

    @staticmethod
    def deserialize_state(data: dict) -> Any:
        """Deserialize agent state from dict."""
        ...

    @staticmethod
    async def diff_checkpoints(cp1: Checkpoint, cp2: Checkpoint) -> dict:
        """Compute diff between two checkpoints."""
        ...

    @staticmethod
    async def merge_checkpoints(base: Checkpoint, changes: Checkpoint) -> Checkpoint:
        """Merge checkpoint changes."""
        ...
```

**Implementation Requirements**:
1. Extract checkpoint logic from GraphAgent into CheckpointableMixin
2. Create CheckpointUtils for common operations
3. Update GraphAgent to use mixin
4. Add examples for LoopAgent and SequentialAgent with checkpointing

**API Changes**: None (GraphAgent maintains current API via mixin)

**Testing**:
- Test LoopAgent with checkpointing
- Test SequentialAgent with checkpointing
- Test checkpoint serialization/deserialization
- Test checkpoint diffing and merging

**Breaking Changes**: None

---

## 4. D3 Visualization Improvements

**Status**: Not started
**Priority**: Low
**Complexity**: Medium
**Estimated effort**: 1-2 weeks

### Motivation

Current visualization shows basic graph structure. Enhanced visualization could show:
- Interrupt points (before/after nodes)
- Callback hooks (observability)
- Execution history (which paths were taken)
- State evolution (how state changed at each node)

### Design

**Enhanced Visualization Data**:
```python
class GraphVisualizationData:
    """Enhanced data for D3 visualization."""

    nodes: List[NodeVisualization]
    edges: List[EdgeVisualization]
    execution_history: List[ExecutionEvent]
    interrupt_points: List[InterruptPoint]
    callback_hooks: List[CallbackHook]

class NodeVisualization:
    name: str
    type: str  # "agent", "function", "parallel_group"
    has_before_callback: bool
    has_after_callback: bool
    interrupt_mode: Optional[InterruptMode]  # BEFORE, AFTER, BOTH
    execution_count: int  # How many times executed
    average_duration: float  # Average execution time

class EdgeVisualization:
    from_node: str
    to_node: str
    condition: Optional[str]  # Condition function source
    priority: int
    weight: float
    execution_count: int  # How many times taken

class ExecutionEvent:
    timestamp: float
    node: str
    event_type: str  # "start", "complete", "interrupt", "callback"
    state_snapshot: dict  # State at this point
```

**D3 Features**:
1. **Color coding**:
   - Green = completed successfully
   - Yellow = interrupted
   - Red = failed
   - Blue = has callbacks
2. **Interactive timeline**: Replay execution step-by-step
3. **State diff viewer**: Show state changes at each node
4. **Interrupt markers**: Show where interrupts occurred
5. **Callback indicators**: Show callback invocations
6. **Parallel execution**: Show concurrent nodes side-by-side

**Implementation**:
1. Add `collect_visualization_data()` method to GraphAgent
2. Enhance `export_to_dict()` with execution history
3. Update D3 templates with new visualization features
4. Add interactive controls (play/pause, step, zoom)

**Testing**:
- Test visualization data generation
- Visual regression testing for D3 output
- Test interactive controls

**Breaking Changes**: None (enhancement to existing export)

---

## 5. Distributed/Production Enhancements

**Status**: Not planned (future consideration)
**Priority**: Low
**Complexity**: High
**Estimated effort**: 4-6 weeks

These enhancements are for large-scale production deployments:

### 5.1 Persistent State (Database-backed)

Replace in-memory session state with database storage:

```python
class DatabaseCheckpointService(CheckpointService):
    """Checkpoint service with database persistence."""

    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url)
        self.session_factory = async_sessionmaker(self.engine)
```

**Benefits**:
- Crash recovery (survive process restarts)
- Distributed execution (multiple workers)
- Long-running workflows (sessions persist indefinitely)

**Databases**: PostgreSQL, Cloud Spanner, Firestore

### 5.2 Distributed Interrupt Service (Redis-backed)

Replace in-memory queues with Redis:

```python
class RedisInterruptService(InterruptService):
    """Interrupt service with Redis pub/sub."""

    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)

    async def send_interrupt(self, session_id: str, message: InterruptMessage):
        # Publish to Redis channel
        await self.redis.publish(f"interrupts:{session_id}", message.json())
```

**Benefits**:
- Multi-process deployments (interrupt any worker)
- Horizontal scaling (distributed queue)
- High availability (Redis cluster)

### 5.3 Priority Queues & Message Expiration

Add priority and TTL to interrupt messages:

```python
class InterruptMessage:
    priority: int = 0       # Higher = processed first
    ttl_seconds: int = 300  # Message expires after 5 minutes
```

### 5.4 Batch Operations

Bulk session management:

```python
# Pause all sessions for a user
await interrupt_service.pause_all(user_id="user123")

# Cancel all sessions older than 1 hour
await interrupt_service.cancel_stale(max_age_seconds=3600)
```

### 5.5 Webhooks & Admin API

HTTP callbacks and REST API:

```python
# Webhooks on state changes
await interrupt_service.register_webhook(
    session_id="session123",
    event="paused",
    webhook_url="https://example.com/webhook",
)

# Admin API
GET /api/sessions              # List all sessions
GET /api/sessions/{id}         # Get session details
POST /api/sessions/{id}/pause  # Pause session
DELETE /api/sessions/{id}      # Cancel session
```

---

## Implementation Priority

**Phase 1 (High Priority)**: 0-3 months
1. Parallel node execution ⭐⭐⭐
2. Extract checkpointing utilities ⭐⭐⭐

**Phase 2 (Medium Priority)**: 3-6 months
3. Enhanced conditional routing ⭐⭐
4. D3 visualization improvements ⭐⭐

**Phase 3 (Future)**: 6+ months
5. Distributed/production enhancements ⭐

---

## Success Metrics

**Phase 1**:
- [ ] Parallel execution improves throughput by 2-3x on independent nodes
- [ ] All agent types support checkpointing via mixin
- [ ] No performance regression on sequential workloads

**Phase 2**:
- [ ] Edge priorities enable complex routing patterns
- [ ] Visualization shows execution history and interrupt points
- [ ] Developer satisfaction increases (measured via surveys)

**Phase 3**:
- [ ] System handles 1000+ concurrent sessions
- [ ] Database-backed checkpoints enable crash recovery
- [ ] Distributed interrupts work across multi-process deployments

---

## References

- Current implementation: `src/google/adk/agents/graph/`
- Design doc: `contributing/docs/graph_agent_design.md`
- Tests: `tests/unittests/agents/test_graph_agent.py`
