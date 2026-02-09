"""Example 5: Basic Interrupts

Demonstrates:
- InterruptService integration
- AFTER interrupt timing
- Manual interrupt actions (continue, rerun, pause)
- Interrupt message handling

Run: python -m contributing.samples.graph_examples.05_interrupts_basic.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import (
    GraphAgent,
    GraphNode,
    InterruptConfig,
    InterruptMode,
)
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class ReviewAgent(BaseAgent):
    """Agent that generates content for review."""

    def __init__(self, name: str, content: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._content = content

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"📝 Generated: {self._content}")]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 5: Basic Interrupts")
    print("=" * 60 + "\n")

    # Create agents
    draft = ReviewAgent(name="draft", content="Initial draft content")
    review = ReviewAgent(name="review", content="Review feedback")
    finalize = ReviewAgent(name="finalize", content="Final version")

    # Setup interrupt service
    interrupt_service = InterruptService()

    # Build graph with interrupt support
    graph = GraphAgent(
        name="interrupt_workflow",
        interrupt_service=interrupt_service,
        interrupt_config=InterruptConfig(
            mode=InterruptMode.AFTER,  # Interrupt AFTER node execution
            nodes=["draft"],  # Only interrupt at draft node
        ),
    )

    graph.add_node(GraphNode(name="draft", agent=draft))
    graph.add_node(GraphNode(name="review", agent=review))
    graph.add_node(GraphNode(name="finalize", agent=finalize))

    graph.add_edge("draft", "review")
    graph.add_edge("review", "finalize")

    graph.set_start("draft")
    graph.set_end("finalize")

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="interrupt_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🚀 Executing workflow with interrupt support...\n")
    print("   Note: Interrupt configured AFTER 'draft' node")
    print("   (This demo doesn't actually send interrupts)\n")

    new_message = types.Content(parts=[types.Part(text="Start")])
    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"   {part.text}")

    print("\n✅ Example complete!")
    print("\n   To send interrupt in real scenario:")
    print("   await interrupt_service.send_interrupt(")
    print('       session_id="session1",')
    print('       text="Revise the draft",')
    print('       action="rerun"')
    print("   )\n")


if __name__ == "__main__":
    asyncio.run(main())
