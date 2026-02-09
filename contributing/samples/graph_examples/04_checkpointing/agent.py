"""Example 4: Checkpointing & Resume

Demonstrates:
- Automatic checkpointing at each node
- Listing checkpoints
- State persistence
- Checkpoint metadata

Run: python -m contributing.samples.graph_examples.04_checkpointing.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode
from google.adk.checkpoints import CheckpointService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class StepAgent(BaseAgent):
    """Agent that represents a workflow step."""

    def __init__(self, name: str, step_num: int, **kwargs):
        super().__init__(name=name, **kwargs)
        self._step_num = step_num

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"✅ Completed step {self._step_num}")]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 4: Checkpointing & Resume")
    print("=" * 60 + "\n")

    # Create agents
    step1 = StepAgent(name="step1", step_num=1)
    step2 = StepAgent(name="step2", step_num=2)
    step3 = StepAgent(name="step3", step_num=3)

    # Setup checkpoint service
    session_service = InMemorySessionService()
    checkpoint_service = CheckpointService(session_service)

    # Build graph with checkpointing enabled
    graph = GraphAgent(name="checkpoint_workflow", checkpointing=True)
    graph.add_node(GraphNode(name="step1", agent=step1))
    graph.add_node(GraphNode(name="step2", agent=step2))
    graph.add_node(GraphNode(name="step3", agent=step3))

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")

    graph.set_start("step1")
    graph.set_end("step3")

    # Execute with checkpointing
    runner = Runner(
        app_name="checkpoint_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🚀 Executing workflow with checkpointing enabled...\n")

    new_message = types.Content(parts=[types.Part(text="Start")])
    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"   {part.text}")

    # Get session and check checkpoint data
    session = await session_service.get_session(
        app_name="checkpoint_demo", user_id="user1", session_id="session1"
    )

    checkpoint_data = session.state.get("graph_checkpoint", {})
    print(f"\n📊 Checkpoint Information:")
    print(f"   Last checkpoint at: {checkpoint_data.get('node', 'N/A')}")
    print(f"   Iteration: {checkpoint_data.get('iteration', 'N/A')}")

    # Show execution path
    path = session.state.get("graph_path", [])
    print(f"   Execution path: {' → '.join(path)}")

    print("\n✅ Example complete!")
    print("   Note: Checkpoints created at each node for state persistence\n")


if __name__ == "__main__":
    asyncio.run(main())
