"""Example 8: Rewind Integration

Demonstrates:
- Invocation tracking per node
- Rewinding to specific node execution
- Re-execution after rewind
- State restoration

Run: python -m contributing.samples.graph_examples.08_rewind.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class CounterAgent(BaseAgent):
    """Agent that tracks execution count."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._count = 0

    async def _run_async_impl(self, ctx):
        self._count += 1
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"✅ {self.name} executed (count: {self._count})")]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 8: Rewind Integration")
    print("=" * 60 + "\n")

    # Create agents
    step1 = CounterAgent(name="step1")
    step2 = CounterAgent(name="step2")
    step3 = CounterAgent(name="step3")

    # Build graph
    graph = GraphAgent(name="rewind_workflow")
    graph.add_node(GraphNode(name="step1", agent=step1))
    graph.add_node(GraphNode(name="step2", agent=step2))
    graph.add_node(GraphNode(name="step3", agent=step3))

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")

    graph.set_start("step1")
    graph.set_end("step3")

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="rewind_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🚀 First execution...\n")

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
        app_name="rewind_demo", user_id="user1", session_id="session1"
    )
    node_invocations = session.state.get("node_invocations", {})

    print(f"\n📊 Invocation Tracking:")
    for node_name, invocations in node_invocations.items():
        print(f"   {node_name}: {len(invocations)} invocation(s)")

    # Rewind to step2
    print(f"\n⏪ Rewinding to 'step2'...")
    await graph.rewind_to_node(
        session_service,
        app_name="rewind_demo",
        user_id="user1",
        session_id="session1",
        node_name="step2",
        invocation_index=-1,  # Last invocation
    )

    print("   ✅ Rewind successful! State restored to before step2")

    # Re-execute from rewind point
    print("\n🚀 Re-execution after rewind...\n")

    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"   {part.text}")

    print("\n✅ Example complete!")
    print("   Note: step1 count stays at 1, step2 & step3 executed again\n")


if __name__ == "__main__":
    asyncio.run(main())
