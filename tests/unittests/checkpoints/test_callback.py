"""Tests for CheckpointCallback integration with BaseAgent."""

import pytest

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.checkpoints import CheckpointCallback
from google.adk.checkpoints import CheckpointService
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService


@pytest.fixture
async def session_service():
    """Create InMemorySessionService."""
    return InMemorySessionService()


@pytest.fixture
async def artifact_service():
    """Create InMemoryArtifactService."""
    return InMemoryArtifactService()


@pytest.fixture
async def checkpoint_service(session_service, artifact_service):
    """Create CheckpointService."""
    return CheckpointService(
        session_service=session_service,
        artifact_service=artifact_service,
    )


@pytest.fixture
async def session(session_service):
    """Create test session."""
    return await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
    )


@pytest.fixture
def simple_agent():
    """Create simple test agent."""

    class TestAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(author=self.name)

    return TestAgent(name="test_agent", description="Test agent")


class TestCheckpointCallback:
    """Test CheckpointCallback functionality."""

    @pytest.mark.asyncio
    async def test_before_agent_creates_checkpoint(
        self, checkpoint_service, session, simple_agent
    ):
        """Test that before_agent callback creates a checkpoint."""
        callback = CheckpointCallback(checkpoint_service)

        # Create invocation context
        ctx = InvocationContext(
            session=session,
            agent=simple_agent,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation-1",
        )

        # Create callback context
        callback_context = CallbackContext(invocation_context=ctx)

        # Call before_agent
        result = await callback.before_agent(callback_context)

        # Should return None (doesn't override execution)
        assert result is None

        # Checkpoint should be created
        checkpoint_id = f"{session.id}-{simple_agent.name}-before"
        checkpoint = await checkpoint_service.get_checkpoint(
            session=session,
            checkpoint_id=checkpoint_id,
        )

        assert checkpoint is not None
        assert checkpoint.checkpoint_id == checkpoint_id
        assert checkpoint.agent_name == simple_agent.name
        assert "Before test_agent execution" in checkpoint.description

    @pytest.mark.asyncio
    async def test_after_agent_creates_checkpoint(
        self, checkpoint_service, session, simple_agent
    ):
        """Test that after_agent callback creates a checkpoint."""
        callback = CheckpointCallback(checkpoint_service)

        ctx = InvocationContext(
            session=session,
            agent=simple_agent,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation-2",
        )

        callback_context = CallbackContext(invocation_context=ctx)

        # Call after_agent
        result = await callback.after_agent(callback_context)

        # Should return None
        assert result is None

        # Checkpoint should be created
        checkpoint_id = f"{session.id}-{simple_agent.name}-after"
        checkpoint = await checkpoint_service.get_checkpoint(
            session=session,
            checkpoint_id=checkpoint_id,
        )

        assert checkpoint is not None
        assert checkpoint.checkpoint_id == checkpoint_id
        assert "After test_agent execution" in checkpoint.description

    @pytest.mark.asyncio
    async def test_checkpoint_before_only(
        self, checkpoint_service, session, simple_agent
    ):
        """Test callback with checkpoint_before=True, checkpoint_after=False."""
        callback = CheckpointCallback(
            checkpoint_service,
            checkpoint_before=True,
            checkpoint_after=False,
        )

        ctx = InvocationContext(
            session=session,
            agent=simple_agent,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation",
        )
        callback_context = CallbackContext(invocation_context=ctx)

        # Before should create checkpoint
        await callback.before_agent(callback_context)
        before_id = f"{session.id}-{simple_agent.name}-before"
        before_cp = await checkpoint_service.get_checkpoint(session, before_id)
        assert before_cp is not None

        # After should NOT create checkpoint
        await callback.after_agent(callback_context)
        after_id = f"{session.id}-{simple_agent.name}-after"
        after_cp = await checkpoint_service.get_checkpoint(session, after_id)
        assert after_cp is None

    @pytest.mark.asyncio
    async def test_checkpoint_after_only(
        self, checkpoint_service, session, simple_agent
    ):
        """Test callback with checkpoint_before=False, checkpoint_after=True."""
        callback = CheckpointCallback(
            checkpoint_service,
            checkpoint_before=False,
            checkpoint_after=True,
        )

        ctx = InvocationContext(
            session=session,
            agent=simple_agent,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation",
        )
        callback_context = CallbackContext(invocation_context=ctx)

        # Before should NOT create checkpoint
        await callback.before_agent(callback_context)
        before_id = f"{session.id}-{simple_agent.name}-before"
        before_cp = await checkpoint_service.get_checkpoint(session, before_id)
        assert before_cp is None

        # After should create checkpoint
        await callback.after_agent(callback_context)
        after_id = f"{session.id}-{simple_agent.name}-after"
        after_cp = await checkpoint_service.get_checkpoint(session, after_id)
        assert after_cp is not None

    @pytest.mark.asyncio
    async def test_multiple_agents_same_session(self, checkpoint_service, session):
        """Test checkpointing multiple different agents in same session."""

        class Agent1(BaseAgent):
            async def _run_async_impl(self, ctx):
                yield Event(author=self.name)

        class Agent2(BaseAgent):
            async def _run_async_impl(self, ctx):
                yield Event(author=self.name)

        agent1 = Agent1(name="agent1", description="First agent")
        agent2 = Agent2(name="agent2", description="Second agent")

        callback = CheckpointCallback(checkpoint_service)

        # Checkpoint agent1
        ctx1 = InvocationContext(
            session=session,
            agent=agent1,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation-agent1",
        )
        callback_ctx1 = CallbackContext(invocation_context=ctx1)
        await callback.before_agent(callback_ctx1)

        # Checkpoint agent2
        ctx2 = InvocationContext(
            session=session,
            agent=agent2,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation-agent2",
        )
        callback_ctx2 = CallbackContext(invocation_context=ctx2)
        await callback.before_agent(callback_ctx2)

        # Both checkpoints should exist
        cp1 = await checkpoint_service.get_checkpoint(
            session, f"{session.id}-agent1-before"
        )
        cp2 = await checkpoint_service.get_checkpoint(
            session, f"{session.id}-agent2-before"
        )

        assert cp1 is not None
        assert cp2 is not None
        assert cp1.agent_name == "agent1"
        assert cp2.agent_name == "agent2"

    @pytest.mark.asyncio
    async def test_callback_with_agent_execution(
        self, checkpoint_service, session_service, artifact_service
    ):
        """Test callback integration with actual agent execution."""

        class CounterAgent(BaseAgent):
            async def _run_async_impl(self, ctx):
                # Simulate agent work
                ctx.session.state["counter"] = ctx.session.state.get("counter", 0) + 1
                yield Event(author=self.name)

        agent = CounterAgent(name="counter", description="Counter agent")

        # Create callback
        checkpoint_callback = CheckpointCallback(
            CheckpointService(session_service, artifact_service)
        )

        # Set callbacks on agent
        agent.before_agent_callback = checkpoint_callback.before_agent
        agent.after_agent_callback = checkpoint_callback.after_agent

        # Create session
        session = await session_service.create_session(
            app_name="test",
            user_id="user",
            session_id="callback_test",
        )

        # Run agent
        ctx = InvocationContext(
            session=session,
            agent=agent,
            session_service=checkpoint_service.session_service,
            invocation_id="test-invocation-callback",
        )
        events = []
        async for event in agent.run_async(ctx):
            events.append(event)

        # Checkpoints should exist
        service = CheckpointService(session_service, artifact_service)
        response = await service.list_checkpoints(session)

        # Should have before and after checkpoints
        assert len(response.checkpoints) >= 2

        checkpoint_ids = [cp.checkpoint_id for cp in response.checkpoints]
        assert any("before" in cp_id for cp_id in checkpoint_ids)
        assert any("after" in cp_id for cp_id in checkpoint_ids)
