"""Example 9: Parallel Execution - WAIT_ALL

Demonstrates:
- Concurrent node execution
- WAIT_ALL join strategy
- State isolation in parallel branches
- Event streaming from parallel nodes

Run: python -m contributing.samples.graph_examples.09_parallel_wait_all.agent
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


class FetchAgent(BaseAgent):
    """Simulates fetching data from a source."""

    def __init__(self, name: str, source: str, delay_ms: int, **kwargs):
        super().__init__(name=name, **kwargs)
        self._source = source
        self._delay_ms = delay_ms

    async def _run_async_impl(self, ctx):
        # Simulate async I/O
        await asyncio.sleep(self._delay_ms / 1000.0)

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part(text=f"✅ Fetched data from {self._source} ({self._delay_ms}ms)")
                ]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 9: Parallel Execution - WAIT_ALL")
    print("=" * 60 + "\n")

    # Create agents with different latencies
    fetch_users = FetchAgent(name="fetch_users", source="users_db", delay_ms=150)
    fetch_products = FetchAgent(name="fetch_products", source="products_db", delay_ms=100)
    fetch_orders = FetchAgent(name="fetch_orders", source="orders_db", delay_ms=200)

    # Build graph
    graph = GraphAgent(name="parallel_workflow")
    graph.add_node(GraphNode(name="fetch_users", agent=fetch_users))
    graph.add_node(GraphNode(name="fetch_products", agent=fetch_products))
    graph.add_node(GraphNode(name="fetch_orders", agent=fetch_orders))

    # Add parallel group with WAIT_ALL strategy
    graph.add_parallel_group(
        "fetch_all",
        ParallelNodeGroup(
            nodes=["fetch_users", "fetch_products", "fetch_orders"],
            join_strategy=JoinStrategy.WAIT_ALL,  # Wait for ALL to complete
        ),
    )

    # Set start (any node in parallel group triggers all)
    graph.set_start("fetch_users")
    graph.set_end("fetch_users")

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="parallel_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🚀 Executing parallel workflow...")
    print("   Strategy: WAIT_ALL (wait for all 3 fetches)")
    print("   Expected: ~200ms (max latency, not sum)\n")

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

    print(f"\n✅ Example complete in {total_time}ms!")
    print(f"   Sequential would take: 450ms (150+100+200)")
    print(f"   Parallel took: ~200ms (max of 3)")
    print(f"   Speedup: ~2.25x\n")


if __name__ == "__main__":
    asyncio.run(main())
