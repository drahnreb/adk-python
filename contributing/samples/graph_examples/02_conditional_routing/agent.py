"""Example 2: Conditional Routing

Demonstrates:
- Conditional edges based on state
- Multiple routing paths
- State-based decision making

Run: python -m contributing.samples.graph_examples.02_conditional_routing.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode, GraphState
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class ValidatorAgent(BaseAgent):
    """Validates input and sets quality score."""

    def __init__(self, name: str, score: int, **kwargs):
        super().__init__(name=name, **kwargs)
        self._score = score

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"✅ Validation complete (score: {self._score})")]
            ),
        )


class ProcessAgent(BaseAgent):
    """Process based on quality."""

    def __init__(self, name: str, quality: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._quality = quality

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"⚙️  {self._quality} quality processing")]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 2: Conditional Routing")
    print("=" * 60 + "\n")

    # Test with different scores
    for test_score in [95, 75, 45]:
        print(f"🎯 Testing with score: {test_score}")

        # Create agents
        validate = ValidatorAgent(name="validate", score=test_score)
        high_quality = ProcessAgent(name="high_quality", quality="HIGH")
        medium_quality = ProcessAgent(name="medium_quality", quality="MEDIUM")
        low_quality = ProcessAgent(name="low_quality", quality="LOW")

        # Build graph with conditional routing
        graph = GraphAgent(name="conditional_workflow")
        graph.add_node(
            GraphNode(
                name="validate",
                agent=validate,
                output_mapper=lambda output, state: GraphState(
                    data={**state.data, "score": test_score},
                    metadata=state.metadata,
                ),
            )
        )
        graph.add_node(GraphNode(name="high_quality", agent=high_quality))
        graph.add_node(GraphNode(name="medium_quality", agent=medium_quality))
        graph.add_node(GraphNode(name="low_quality", agent=low_quality))

        # Conditional edges based on score
        graph.add_edge(
            "validate", "high_quality", condition=lambda s: s.data.get("score", 0) >= 80
        )
        graph.add_edge(
            "validate",
            "medium_quality",
            condition=lambda s: 50 <= s.data.get("score", 0) < 80,
        )
        graph.add_edge(
            "validate", "low_quality", condition=lambda s: s.data.get("score", 0) < 50
        )

        graph.set_start("validate")
        graph.set_end("high_quality")
        graph.set_end("medium_quality")
        graph.set_end("low_quality")

        # Execute
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="routing_demo",
            agent=graph,
            session_service=session_service,
            auto_create_session=True,
        )

        new_message = types.Content(parts=[types.Part(text="Start")])
        async for event in runner.run_async(
            user_id="user1",
            session_id=f"session_{test_score}",
            new_message=new_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"   {part.text}")

        print()

    print("✅ Example complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
