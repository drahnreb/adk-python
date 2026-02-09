"""Example 14: Parallel Execution + Rewind

Demonstrates:
- Parallel node execution
- Invocation tracking in parallel workflows
- Rewinding to parallel node
- Re-execution of parallel group

Run: python -m contributing.samples.graph_examples.14_parallel_rewind.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import (
    GraphAgent,
    GraphNode,
    ParallelNodeGroup,
)
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class TaskAgent(BaseAgent):
    """Agent that executes a task."""

    def __init__(self, name: str, task_name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._task_name = task_name
        self._count = 0

    async def _run_async_impl(self, ctx):
        self._count += 1
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part(
                        text=f"✅ {self._task_name} completed (execution #{self._count})"
                    )
                ]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 14: Parallel Execution + Rewind")
    print("=" * 60 + "\n")

    # Create agents
    task1 = TaskAgent(name="task1", task_name="Data fetch")
    task2 = TaskAgent(name="task2", task_name="Data transform")
    merge = TaskAgent(name="merge", task_name="Merge results")

    # Build graph
    graph = GraphAgent(name="parallel_rewind_workflow")
    graph.add_node(GraphNode(name="task1", agent=task1))
    graph.add_node(GraphNode(name="task2", agent=task2))
    graph.add_node(GraphNode(name="merge", agent=merge))

    # Add parallel group
    graph.add_parallel_group(
        "parallel_tasks", ParallelNodeGroup(nodes=["task1", "task2"])
    )

    graph.add_edge("task1", "merge")
    graph.add_edge("task2", "merge")

    graph.set_start("task1")
    graph.set_end("merge")

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="parallel_rewind_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🚀 First execution (parallel tasks)...\n")

    new_message = types.Content(parts=[types.Part(text="Start")])
    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"   {part.text}")

    # Check invocations
    session = await session_service.get_session(
        app_name="parallel_rewind_demo", user_id="user1", session_id="session1"
    )
    node_invocations = session.state.get("node_invocations", {})

    print(f"\n📊 Invocation Tracking:")
    for node_name, invocations in node_invocations.items():
        print(f"   {node_name}: {len(invocations)} invocation(s)")

    # Rewind to task1 (part of parallel group)
    print(f"\n⏪ Rewinding to 'task1' (parallel group node)...")
    await graph.rewind_to_node(
        session_service,
        app_name="parallel_rewind_demo",
        user_id="user1",
        session_id="session1",
        node_name="task1",
        invocation_index=-1,
    )

    print("   ✅ Rewind successful!")

    # Re-execute from rewind point
    print("\n🚀 Re-execution after rewind (parallel group re-runs)...\n")

    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"   {part.text}")

    print("\n✅ Example complete!")
    print("\n   Key Points:")
    print("   - Rewind works with parallel nodes")
    print("   - Entire parallel group re-executes")
    print("   - Invocations tracked per node")
    print("   - Execution counts show: task1=#1, task2=#2, merge=#2\n")


if __name__ == "__main__":
    asyncio.run(main())
