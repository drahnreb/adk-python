"""Example 3: Enhanced Routing (Priority, Weight, Fallback)

Demonstrates:
- Priority-based routing (higher priority evaluated first)
- Weighted random selection (probabilistic routing)
- Fallback edges (priority=0 always matches)

Run: python -m contributing.samples.graph_examples.03_enhanced_routing.agent
"""

import asyncio
from google.genai import types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import (
    GraphAgent,
    GraphNode,
    EdgeCondition,
    GraphState,
)
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


class SimpleAgent(BaseAgent):
    """Agent that outputs a message."""

    def __init__(self, name: str, message: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._message = message

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=self._message)]),
        )


class ScoreAgent(BaseAgent):
    """Agent that sets a risk score."""

    def __init__(self, name: str, score: float, **kwargs):
        super().__init__(name=name, **kwargs)
        self._score = score

    async def _run_async_impl(self, ctx):
        ctx.session.state["risk_score"] = self._score
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"Risk score: {self._score}")]
            ),
        )


async def main():
    print("\n" + "=" * 60)
    print("Example 3: Enhanced Routing")
    print("=" * 60 + "\n")

    # ===== Example 1: Priority-based Routing =====
    print("📊 Example 1: Priority-based Routing\n")

    graph1 = GraphAgent(name="priority_routing")

    analyze = ScoreAgent(name="analyze", score=0.85)
    critical = SimpleAgent(name="critical", message="🚨 CRITICAL: Immediate action required")
    warning = SimpleAgent(name="warning", message="⚠️  WARNING: Review recommended")
    normal = SimpleAgent(name="normal", message="✅ NORMAL: No action needed")

    graph1.add_node(GraphNode(name="analyze", agent=analyze))
    graph1.add_node(GraphNode(name="critical", agent=critical))
    graph1.add_node(GraphNode(name="warning", agent=warning))
    graph1.add_node(GraphNode(name="normal", agent=normal))

    # Set output mapper to persist risk_score in state
    def store_score(output, state):
        new_state = GraphState(data=state.data.copy(), metadata=state.metadata.copy())
        new_state.data["risk_score"] = 0.85  # Score from analyze agent
        return new_state

    graph1.nodes["analyze"].output_mapper = store_score

    # Priority-based routing: higher priority evaluated first
    graph1.nodes["analyze"].edges = [
        EdgeCondition(
            target_node="critical",
            condition=lambda s: s.data.get("risk_score", 0) > 0.9,
            priority=10,  # Highest priority
        ),
        EdgeCondition(
            target_node="warning",
            condition=lambda s: s.data.get("risk_score", 0) > 0.7,
            priority=5,  # Medium priority - THIS WILL MATCH
        ),
        EdgeCondition(
            target_node="normal",
            priority=0,  # Fallback (priority=0 always matches if no other matched)
        ),
    ]

    graph1.set_start("analyze")
    graph1.set_end("critical")
    graph1.set_end("warning")
    graph1.set_end("normal")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="routing_demo",
        agent=graph1,
        session_service=session_service,
        auto_create_session=True,
    )

    async for event in runner.run_async(
        user_id="user1", session_id="session1", new_message=types.Content(parts=[types.Part(text="Analyze")])
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"   {event.content.parts[0].text}")

    print("\n   💡 Score was 0.85 → matched 'warning' (priority=5)")
    print("   💡 'critical' didn't match (0.85 < 0.9)")
    print("   💡 'normal' fallback not needed (higher priority matched)\n")

    # ===== Example 2: Weighted Random Routing =====
    print("🎲 Example 2: Weighted Random Routing\n")

    graph2 = GraphAgent(name="weighted_routing")

    start = SimpleAgent(name="start", message="Starting load balancer...")
    server_a = SimpleAgent(name="server_a", message="   → Routed to Server A")
    server_b = SimpleAgent(name="server_b", message="   → Routed to Server B")
    server_c = SimpleAgent(name="server_c", message="   → Routed to Server C")

    graph2.add_node(GraphNode(name="start", agent=start))
    graph2.add_node(GraphNode(name="server_a", agent=server_a))
    graph2.add_node(GraphNode(name="server_b", agent=server_b))
    graph2.add_node(GraphNode(name="server_c", agent=server_c))

    # Weighted routing: all at same priority, different weights
    graph2.nodes["start"].edges = [
        EdgeCondition(
            target_node="server_a",
            condition=lambda s: True,  # All match
            priority=1,  # Same priority
            weight=0.5,  # 50% probability
        ),
        EdgeCondition(
            target_node="server_b",
            condition=lambda s: True,
            priority=1,  # Same priority
            weight=0.3,  # 30% probability
        ),
        EdgeCondition(
            target_node="server_c",
            condition=lambda s: True,
            priority=1,  # Same priority
            weight=0.2,  # 20% probability
        ),
    ]

    graph2.set_start("start")
    graph2.set_end("server_a")
    graph2.set_end("server_b")
    graph2.set_end("server_c")

    runner2 = Runner(
        app_name="weighted_demo",
        agent=graph2,
        session_service=session_service,
        auto_create_session=True,
    )

    # Run multiple times to show distribution
    counts = {"server_a": 0, "server_b": 0, "server_c": 0}
    trials = 20

    print(f"   Running {trials} trials with weights (A:50%, B:30%, C:20%):\n")

    for i in range(trials):
        async for event in runner2.run_async(
            user_id="user1",
            session_id=f"session_weighted_{i}",
            new_message=types.Content(parts=[types.Part(text="Route")]),
        ):
            if event.content and event.content.parts and event.author in counts:
                text = event.content.parts[0].text
                counts[event.author] += 1
                print(f"   Trial {i+1:2d}: {text}")

    print(f"\n   📊 Distribution after {trials} trials:")
    print(f"   Server A: {counts['server_a']:2d}/{trials} ({counts['server_a']/trials*100:.0f}%)")
    print(f"   Server B: {counts['server_b']:2d}/{trials} ({counts['server_b']/trials*100:.0f}%)")
    print(f"   Server C: {counts['server_c']:2d}/{trials} ({counts['server_c']/trials*100:.0f}%)\n")

    # ===== Example 3: Fallback Edge =====
    print("🛡️  Example 3: Fallback Edge (priority=0)\n")

    graph3 = GraphAgent(name="fallback_routing")

    validate = ScoreAgent(name="validate", score=0.5)
    premium = SimpleAgent(name="premium", message="🌟 Premium path (VIP users)")
    standard = SimpleAgent(name="standard", message="📦 Standard path (regular users)")
    fallback = SimpleAgent(name="fallback", message="🔒 Fallback path (default handler)")

    graph3.add_node(GraphNode(name="validate", agent=validate))
    graph3.add_node(GraphNode(name="premium", agent=premium))
    graph3.add_node(GraphNode(name="standard", agent=standard))
    graph3.add_node(GraphNode(name="fallback", agent=fallback))

    def store_score_fallback(output, state):
        new_state = GraphState(data=state.data.copy(), metadata=state.metadata.copy())
        new_state.data["risk_score"] = 0.5
        # Don't set is_vip or is_standard - will fall through to fallback
        return new_state

    graph3.nodes["validate"].output_mapper = store_score_fallback

    graph3.nodes["validate"].edges = [
        EdgeCondition(
            target_node="premium",
            condition=lambda s: s.data.get("is_vip", False),
            priority=10,  # High priority - won't match
        ),
        EdgeCondition(
            target_node="standard",
            condition=lambda s: s.data.get("is_standard", False),
            priority=5,  # Medium priority - won't match
        ),
        EdgeCondition(
            target_node="fallback",
            priority=0,  # FALLBACK - always matches if reached
        ),
    ]

    graph3.set_start("validate")
    graph3.set_end("premium")
    graph3.set_end("standard")
    graph3.set_end("fallback")

    runner3 = Runner(
        app_name="fallback_demo",
        agent=graph3,
        session_service=session_service,
        auto_create_session=True,
    )

    async for event in runner3.run_async(
        user_id="user1", session_id="session_fallback", new_message=types.Content(parts=[types.Part(text="Validate")])
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"   {event.content.parts[0].text}")

    print("\n   💡 No is_vip or is_standard flag set")
    print("   💡 All higher priority edges failed to match")
    print("   💡 Fallback (priority=0) caught it!\n")

    print("=" * 60)
    print("✅ Enhanced Routing Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
