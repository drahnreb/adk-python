"""Example 1: Basic GraphAgent Workflow

Demonstrates:
- Creating a simple directed graph
- Adding nodes (agents)
- Adding edges (transitions)
- Setting start and end nodes
- Executing the workflow

Run: python -m contributing.samples.graph_examples.01_basic.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class SimpleAgent(BaseAgent):
    """A simple agent that outputs a message."""

    def __init__(self, name: str, message: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._message = message

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=self._message)]),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 1: Basic GraphAgent Workflow")
    print("=" * 60 + "\n")

    # Create agents
    validate = SimpleAgent(name="validate", message="✅ Validation passed")
    process = SimpleAgent(name="process", message="⚙️  Processing data")
    complete = SimpleAgent(name="complete", message="✅ Workflow complete")

    # Build graph
    graph = GraphAgent(name="basic_workflow")
    graph.add_node(GraphNode(name="validate", agent=validate))
    graph.add_node(GraphNode(name="process", agent=process))
    graph.add_node(GraphNode(name="complete", agent=complete))

    # Add edges (transitions)
    graph.add_edge("validate", "process")
    graph.add_edge("process", "complete")

    # Set start and end
    graph.set_start("validate")
    graph.set_end("complete")

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="basic_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    print("🚀 Executing workflow: validate → process → complete\n")

    new_message = types.Content(parts=[types.Part(text="Start workflow")])
    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"   {part.text}")

    print("\n✅ Example complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
