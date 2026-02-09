"""Example 10: Parallel Execution - WAIT_ANY (Race)

Demonstrates:
- Racing multiple data sources
- WAIT_ANY join strategy
- First-to-complete wins
- Automatic cancellation of slower nodes

Run: python -m contributing.samples.graph_examples.10_parallel_wait_any.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import (
    GraphAgent,
    GraphNode,
    ParallelNodeGroup,
    JoinStrategy,
)
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class DataSourceAgent(BaseAgent):
    """Simulates fetching from different data sources."""

    def __init__(self, name: str, source_type: str, latency_ms: int, **kwargs):
        super().__init__(name=name, **kwargs)
        self._source_type = source_type
        self._latency_ms = latency_ms

    async def _run_async_impl(self, ctx):
        await asyncio.sleep(self._latency_ms / 1000.0)

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part(
                        text=f"✅ Data from {self._source_type} ({self._latency_ms}ms)"
                    )
                ]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 10: Parallel Execution - WAIT_ANY (Race)")
    print("=" * 60 + "\n")

    # Create agents representing different data sources
    from_cache = DataSourceAgent(name="from_cache", source_type="CACHE", latency_ms=50)
    from_database = DataSourceAgent(
        name="from_database", source_type="DATABASE", latency_ms=150
    )
    from_api = DataSourceAgent(name="from_api", source_type="API", latency_ms=300)

    # Build graph
    graph = GraphAgent(name="race_workflow")
    graph.add_node(GraphNode(name="from_cache", agent=from_cache))
    graph.add_node(GraphNode(name="from_database", agent=from_database))
    graph.add_node(GraphNode(name="from_api", agent=from_api))

    # Add parallel group with WAIT_ANY strategy (race!)
    graph.add_parallel_group(
        "data_race",
        ParallelNodeGroup(
            nodes=["from_cache", "from_database", "from_api"],
            join_strategy=JoinStrategy.WAIT_ANY,  # First to finish wins!
        ),
    )

    graph.set_start("from_cache")
    graph.set_end("from_cache")

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="race_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🏁 Starting data source race...")
    print("   Competitors:")
    print("   - Cache:    50ms")
    print("   - Database: 150ms")
    print("   - API:      300ms")
    print("   Strategy: WAIT_ANY (first to complete)\n")

    import time

    start_time = time.time()

    new_message = types.Content(parts=[types.Part(text="Start")])
    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    elapsed = int((time.time() - start_time) * 1000)
                    print(f"   [{elapsed:3d}ms] {part.text}")

    total_time = int((time.time() - start_time) * 1000)

    print(f"\n✅ Race complete in ~{total_time}ms!")
    print("   Winner: Cache (fastest source)")
    print("   Slower sources: Cancelled automatically")
    print("   Use case: Cache-DB-API fallback strategy\n")


if __name__ == "__main__":
    asyncio.run(main())
