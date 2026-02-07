# InterruptService Architecture

**Author**: ADK Team
**Date**: 2026-02-04
**Status**: Production-Ready
**Coverage**: 97% (96 tests)

---

## Overview

InterruptService enables **human-in-the-loop workflows** for GraphAgent by providing dynamic runtime interrupts, pause/resume control, and message injection during graph execution.

Unlike static `interrupt_before` configuration, InterruptService provides:
- **Dynamic runtime interrupts** (not pre-configured)
- **Actual pause/resume** (escalate=True)
- **Message queue** for human input
- **Per-session isolation** for concurrent use
- **Queue bounds** to prevent OOM
- **Timeout and cancellation** support

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        InterruptService                              │
│                                                                      │
│  Config:                                                             │
│  - max_queue_size (default: 100)                                    │
│  - default_timeout (default: 300s)                                  │
│  - session_inactive_timeout (default: 3600s)                        │
│  - max_sessions (default: unlimited)                                │
│  - max_message_length (default: 10KB)                               │
│  - max_metadata_size (default: 100KB)                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ register_session(session_id)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Session State (Per Session)                      │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Message Queue   │  │  Pause Event    │  │ Cancel Event     │  │
│  │  (Bounded: 100)  │  │  (asyncio.Event)│  │ (asyncio.Event)  │  │
│  │                  │  │                 │  │                  │  │
│  │  [msg1, msg2,..]│  │  set/clear      │  │  set/clear       │  │
│  └──────────────────┘  └─────────────────┘  └──────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Session Metrics                            │  │
│  │                                                                │  │
│  │  - messages_queued: 0                                         │  │
│  │  - pause_count: 0                                             │  │
│  │  - resume_count: 0                                            │  │
│  │  - cancel_count: 0                                            │  │
│  │  - max_queue_depth: 0                                         │  │
│  │  - created_at: timestamp                                      │  │
│  │  - last_activity_at: timestamp (for eviction)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   GraphAgent │    │  Human/Client    │    │  Monitoring  │
│              │    │                  │    │              │
│ Operations:  │    │ Operations:      │    │ Operations:  │
│              │    │                  │    │              │
│ • check_     │    │ • pause()        │    │ • get_queue_ │
│   interrupt()│    │ • resume()       │    │   status()   │
│              │    │ • cancel()       │    │              │
│ • wait_if_   │    │ • send_message() │    │ • list_      │
│   paused()   │    │                  │    │   queued_    │
│              │    │                  │    │   messages() │
└──────────────┘    └──────────────────┘    └──────────────┘
```

---

## Core Components

### 1. InterruptServiceConfig

Configuration dataclass with validation:

```python
@dataclass
class InterruptServiceConfig:
    # Queue Management
    max_queue_size: int = 100              # Bounded queue
    default_timeout: float = 300.0         # 5 minutes

    # Security
    max_message_length: int = 10000        # 10KB text
    max_metadata_size: int = 100000        # 100KB metadata
    allowed_actions: Optional[list[str]]   # Action allowlist

    # Scale Management
    session_inactive_timeout: float = 3600.0  # 1 hour
    max_sessions: int = 0                  # 0 = unlimited

    # Observability
    enable_metrics: bool = True
```

### 2. Session State Management

Each session gets isolated resources:

**Message Queue** (asyncio.Queue):
- Bounded size (default: 100 messages)
- FIFO ordering
- Non-blocking `get_nowait()` for check_interrupt()
- Timeout-based `put()` for send_message()

**Pause Event** (asyncio.Event):
- `clear()` → paused (agent blocks)
- `set()` → resumed (agent continues)
- Used by `wait_if_paused()`

**Cancellation Event** (asyncio.Event):
- `clear()` → not cancelled
- `set()` → cancelled (agent exits)
- Checked by `wait_if_paused()`

**Metrics** (SessionMetrics):
- Counters: messages_queued, pause_count, resume_count, cancel_count
- Watermarks: max_queue_depth
- Timestamps: created_at, last_activity_at

### 3. OpenTelemetry Integration

**Gauges**:
- `interrupt.queue_depth`: Real-time queue depth per session

**Counters**:
- `interrupt.operations`: Operation counts (pause, resume, message, cancel)
- `checkpoint.operations`: Checkpoint operation counts

**Histograms**:
- `interrupt.latency`: Operation latency in milliseconds
- `checkpoint.latency`: Checkpoint operation latency

**Traces**:
- Spans with attributes: session_id, operation, status, queue_depth
- Structured logs for all operations

---

## Operation Flows

### Flow 1: Pause and Send Message

```
Human/Client                InterruptService              GraphAgent
     │                            │                           │
     │  pause(session_id) ───────▶│                           │
     │                            │  pause_event.clear()      │
     │                            │  metrics.pause_count++    │
     │                            │  update_activity()        │
     │◀─────────────────────────  │                           │
     │                            │                           │
     │  send_message(...) ───────▶│                           │
     │                            │  validate_message()       │
     │                            │  queue.put(message)       │
     │                            │  metrics.messages_queued++│
     │                            │  update_activity()        │
     │◀─────────────────────────  │                           │
     │                            │                           │
     │                            │  ◀──── check_interrupt()  │
     │                            │  queue.get_nowait()       │
     │                            │  update_activity() ─────▶ │
     │                            │                           │
     │                            │  ◀──── wait_if_paused()   │
     │                            │  pause_event.wait() ───┐  │
     │                            │         (blocks)       │  │
     │  resume(session_id) ──────▶│                       │  │
     │                            │  pause_event.set()    │  │
     │                            │  metrics.resume_count++│  │
     │                            │  update_activity()    │  │
     │                            │  (unblocks) ◀─────────┘  │
     │                            │  ──────────────────────▶ │
     │                            │      (continues)          │
```

### Flow 2: Cancellation

```
Human/Client                InterruptService              GraphAgent
     │                            │                           │
     │  cancel(session_id) ──────▶│                           │
     │                            │  cancel_event.set()       │
     │                            │  queue.clear()            │
     │                            │  metrics.cancel_count++   │
     │◀─────────────────────────  │                           │
     │                            │                           │
     │                            │  ◀──── wait_if_paused()   │
     │                            │  cancel_event.is_set()?   │
     │                            │  return False ──────────▶ │
     │                            │      (agent exits)         │
```

### Flow 3: Session Eviction (Scale Management)

```
Background Task             InterruptService              Sessions
     │                            │                           │
     │  evict_inactive_sessions()─▶│                           │
     │                            │  current_time - last_    │
     │                            │  activity_at > timeout?   │
     │                            │                           │
     │                            │  unregister_session() ───▶│
     │                            │  (cleanup resources)      │
     │◀─────── evicted_count ──── │                           │
     │                            │                           │
     │  register_session() ───────▶│                           │
     │  (exceeds max_sessions)    │                           │
     │                            │  evict_oldest_sessions()  │
     │                            │  (LRU eviction)           │
     │                            │  unregister_session() ───▶│
     │◀─────────────────────────  │                           │
```

---

## Security Features

### 1. Input Validation (Defense in Depth)

**Session ID Validation**:
```python
# Pattern: alphanumeric + hyphens/underscores only
# Max length: 256 characters
# Prevents: Directory traversal, injection attacks
if not re.match(r"^[a-zA-Z0-9\-_]+$", session_id):
    raise ValueError("Invalid session_id format")
```

**Message Validation**:
```python
# Text length: max 10KB (configurable)
# Metadata size: max 100KB (configurable)
# Prevents: Memory exhaustion, DoS attacks
if len(text) > config.max_message_length:
    raise ValueError("Message exceeds max length")
```

**Action Allowlist**:
```python
# Optional action type validation
# Prevents: Unauthorized operations
if action and config.allowed_actions:
    if action not in config.allowed_actions:
        raise ValueError("Action not allowed")
```

### 2. Resource Limits

**Bounded Queues**:
- Max 100 messages per session (default)
- Prevents memory exhaustion
- Timeout-based backpressure

**Session Limits**:
- Optional max_sessions limit
- LRU eviction when limit reached
- Prevents unbounded memory growth

**Inactivity Timeout**:
- Auto-eviction after 1 hour (default)
- Configurable per deployment
- Cleans up abandoned sessions

---

## Scale Management

### Eviction Policies

**1. Time-Based Eviction** (Automatic):
```python
# Called periodically (e.g., every 10 minutes)
evicted = interrupt_service.evict_inactive_sessions()
# Removes sessions inactive > session_inactive_timeout
```

**2. LRU Eviction** (Triggered):
```python
# Triggered automatically on register_session() when max_sessions exceeded
# Evicts oldest sessions based on last_activity_at
evicted = interrupt_service.evict_oldest_sessions(keep_count=max_sessions)
```

**3. Manual Cleanup**:
```python
# Manual session cleanup
interrupt_service.unregister_session(session_id)
```

### Activity Tracking

Activity timestamp updated on:
- `pause()` - Human pauses session
- `resume()` - Human resumes session
- `send_message()` - Human sends message
- `check_interrupt()` - Agent checks for messages

Used for:
- Time-based eviction (inactive sessions)
- LRU eviction (least recently used)
- Observability (session activity metrics)

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `register_session()` | O(1) | Dict insert + Event creation |
| `unregister_session()` | O(1) | Dict delete + cleanup |
| `pause()` | O(1) | Event.clear() |
| `resume()` | O(1) | Event.set() |
| `send_message()` | O(1) amortized | Queue.put() with timeout |
| `check_interrupt()` | O(1) | Queue.get_nowait() |
| `wait_if_paused()` | O(1) | Event.wait() (blocking) |
| `cancel()` | O(n) | Queue clear, n = queue_depth |
| `get_queue_status()` | O(1) | Queue.qsize() |
| `list_queued_messages()` | O(n) | n = page_size |
| `evict_inactive_sessions()` | O(s) | s = active_sessions |
| `evict_oldest_sessions()` | O(s log s) | Sort by activity |
| `list_active_sessions()` | O(s log s) | Sort by activity |

### Space Complexity

Per session:
- Message queue: O(max_queue_size) = O(100) default
- Events: O(1) = 3 events per session
- Metrics: O(1) = 7 fields
- **Total**: O(max_queue_size) ≈ **100 messages**

For 1000 concurrent sessions:
- **Memory**: ~10MB (100 messages × 10KB × 0.001)
- **Bounded**: Predictable memory usage

### Memory Safety

**Bounded Resources**:
- Queue size: max 100 messages (default)
- Message size: max 10KB text + 100KB metadata
- Session count: optional max_sessions limit

**Automatic Cleanup**:
- Inactive sessions evicted after 1 hour
- LRU eviction when max_sessions exceeded
- Manual cleanup via unregister_session()

---

## UX Features

### Context Manager

Automatic session lifecycle management:

```python
async with interrupt_service.session(session_id):
    # Session automatically registered
    await interrupt_service.pause(session_id)
    await interrupt_service.send_message(session_id, "Test")
    # Session automatically unregistered on exit (even on exception)
```

### Convenience Methods

Simplified common operations:

```python
# Combine pause + send
await interrupt_service.pause_and_send(session_id, "Test")

# Quick checks
if interrupt_service.is_active(session_id):
    if interrupt_service.has_queued_messages(session_id):
        depth = interrupt_service.get_queue_depth(session_id)

# Clear messages without cancelling
cleared = interrupt_service.clear_queue(session_id)
```

### Improved Error Messages

Actionable error messages with diagnostic context:

```python
# Session ID validation
"Invalid session_id format: 'session@123'. "
"Found invalid characters: ['@']. "
"session_id must contain only alphanumeric characters, hyphens (-), and underscores (_)"

# Message validation
"Message text exceeds maximum length: 10001 chars (max: 10000). "
"Consider splitting into multiple messages or adjusting config. "
"Preview: 'This is a very long message...'"

# Timeout error
"Timed out waiting for resume on session 'session-1' after 300s. "
"Session is paused and waiting for human input. "
"Call resume(session_id) or cancel(session_id) to continue. "
"Current status: paused=True, cancelled=False, queue_depth=3"
```

---

## Testing Strategy

### Test Coverage: 97% (96 tests)

**Test Categories**:
1. **Configuration** (4 tests): Validation, defaults, custom config
2. **Session Management** (6 tests): Register, unregister, idempotent
3. **Context Manager** (6 tests): Lifecycle, exceptions, multiple sessions
4. **Pause/Resume** (6 tests): State changes, metrics, edge cases
5. **Message Queuing** (7 tests): FIFO, bounds, timeout, validation
6. **Wait If Paused** (6 tests): Blocking, timeout, cancellation
7. **Cancellation** (4 tests): Event, queue clear, metrics
8. **Queue Status** (5 tests): Status, state reflection, metrics
9. **Message Pagination** (6 tests): Pagination, bounds, edge cases
10. **Telemetry** (5 tests): Traces, metrics, gauge updates
11. **Error Handling** (7 tests): Exceptions, timeouts, cleanup
12. **Input Validation** (10 tests): Session ID, message, metadata, action
13. **Convenience Methods** (13 tests): All convenience methods
14. **Session Eviction** (9 tests): Time-based, LRU, auto-eviction

**Test Patterns**:
- AsyncMock fixtures for isolation
- Property-based validation tests
- Integration tests with GraphAgent
- Performance benchmarks
- Error injection tests

---

## Production Deployment

### Configuration Best Practices

**Multi-Tenant Environments**:
```python
config = InterruptServiceConfig(
    max_queue_size=50,              # Lower per-session limit
    max_sessions=1000,              # Enforce global limit
    session_inactive_timeout=1800,  # 30 minutes
    max_message_length=5000,        # Stricter limits
    max_metadata_size=50000,
    enable_metrics=True,
)
```

**High-Throughput Single-Tenant**:
```python
config = InterruptServiceConfig(
    max_queue_size=500,             # Higher queue capacity
    max_sessions=0,                 # Unlimited sessions
    session_inactive_timeout=7200,  # 2 hours
    max_message_length=50000,       # More lenient
    max_metadata_size=500000,
    enable_metrics=True,
)
```

**Resource-Constrained**:
```python
config = InterruptServiceConfig(
    max_queue_size=20,              # Minimal queues
    max_sessions=100,               # Strict limit
    session_inactive_timeout=600,   # 10 minutes
    max_message_length=1000,        # Very strict
    max_metadata_size=10000,
    enable_metrics=False,           # Disable for performance
)
```

### Monitoring

**Key Metrics**:
- `interrupt.queue_depth`: Alert when > 80% of max_queue_size
- `interrupt.operations{status=timeout}`: Alert on timeout spikes
- `interrupt.operations{status=error}`: Alert on error rate > 1%
- Active session count: Alert when approaching max_sessions

**Health Checks**:
```python
# Periodic health check
session_count = interrupt_service.get_active_session_count()
if session_count > 0.9 * config.max_sessions:
    logger.warning(f"Approaching session limit: {session_count}")

# Eviction maintenance
evicted = interrupt_service.evict_inactive_sessions()
if evicted > 0:
    logger.info(f"Evicted {evicted} inactive sessions")
```

---

## Future Enhancements

**Potential improvements** (not currently planned):

1. **Persistent State**: Store session state in database for crash recovery
2. **Distributed Mode**: Redis-backed queues for multi-process deployments
3. **Priority Queues**: High/low priority messages
4. **Message Expiration**: TTL for queued messages
5. **Batch Operations**: Bulk pause/resume/cancel
6. **Webhooks**: HTTP callbacks on state changes
7. **Admin API**: REST API for session management

---

## References

- **Implementation**: `src/google/adk/agents/graph/interrupt_service.py`
- **Tests**: `tests/unittests/agents/test_interrupt_service.py`
- **Telemetry**: `src/google/adk/telemetry/checkpoint_tracing.py`
- **GraphAgent**: `src/google/adk/agents/graph/graph_agent.py`
- **Related**: Five-Hat Security Review (FIVE_HAT_REVIEW.md)
