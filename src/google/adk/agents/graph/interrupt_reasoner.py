"""LLM-based interrupt reasoning for GraphAgent.

This module provides an LLM agent that intelligently reasons about
interrupt messages and decides what action to take based on context.

The InterruptReasoner is a2a compatible and can be used as a standard
LlmAgent in the ADK framework.

Example:
    ```python
    from google.adk.agents.graph import (
        GraphAgent,
        InterruptConfig,
        InterruptMode,
    )
    from google.adk.agents.graph.interrupt_reasoner import (
        InterruptReasoner,
        InterruptReasonerConfig,
    )

    # Create reasoner with custom config
    reasoner = InterruptReasoner(InterruptReasonerConfig(
        model="gemini-2.0-flash-exp",
        available_actions=["continue", "rerun", "go_back", "pause", "defer"],
    ))

    # Use in GraphAgent
    graph = GraphAgent(
        name="my_graph",
        interrupt_config=InterruptConfig(
            mode=InterruptMode.AFTER,
            reasoner=reasoner,
        ),
    )
    ```
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm_agent import LlmAgent as LlmAgentType
else:
    LlmAgentType = Any

from google import genai

from ..llm_agent import LlmAgent
from .graph_state import GraphState
from .interrupt import InterruptAction
from .interrupt_service import InterruptMessage

logger = logging.getLogger("google_adk." + __name__)


@dataclass
class InterruptReasonerConfig:
    """Configuration for InterruptReasoner.

    Attributes:
        model: LLM model to use for reasoning (default: gemini-2.0-flash-exp)
        instruction: System instruction for the reasoner
        available_actions: List of available actions the reasoner can choose
        custom_actions: Dict of custom action handlers (extensible)
        include_state_in_prompt: Whether to include full state in prompt (default: True)
        max_state_size: Maximum state size to include in prompt (default: 10000)
    """

    model: str = "gemini-2.0-flash-exp"
    instruction: str = (
        "You are an interrupt reasoning agent for a graph-based workflow system. "
        "Analyze interrupt messages from humans and decide what action to take based on "
        "the current execution context, node output, and state."
    )
    available_actions: List[str] = None  # type: ignore[assignment]
    custom_actions: Dict[str, Callable[..., Any]] = None  # type: ignore[assignment]
    include_state_in_prompt: bool = True
    max_state_size: int = 10000

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.available_actions is None:
            self.available_actions = [
                "continue",
                "rerun",
                "go_back",
                "pause",
                "defer",
                "skip",
            ]
        if self.custom_actions is None:
            self.custom_actions = {}


class InterruptReasoner(LlmAgent):  # type: ignore[misc]
    """LLM agent that reasons about interrupt messages and decides actions.

    This agent receives interrupt messages, analyzes the execution context,
    and uses an LLM to intelligently decide what action to take.

    The reasoner is a2a compatible and can be used as a standard ADK agent.

    Attributes:
        config: InterruptReasonerConfig for this reasoner
        available_actions: List of actions the reasoner can choose from
        custom_actions: Dictionary of custom action handlers
    """

    def __init__(
        self,
        config: InterruptReasonerConfig,
        name: str = "interrupt_reasoner",
        **kwargs: Any,
    ):
        """Initialize InterruptReasoner.

        Args:
            config: Configuration for the reasoner
            name: Agent name (default: "interrupt_reasoner")
            **kwargs: Additional arguments passed to LlmAgent
        """
        super().__init__(
            name=name,
            model=config.model,
            instruction=config.instruction,
            **kwargs,
        )
        # Store in private attributes (Pydantic allows these)
        self._config = config
        self._available_actions = config.available_actions
        self._custom_actions = config.custom_actions

    async def reason_about_interrupt(
        self,
        message: InterruptMessage,
        state: GraphState,
        current_node: str,
        ctx: Any,  # InvocationContext
    ) -> InterruptAction:
        """Use LLM to reason about interrupt message and decide action.

        Args:
            message: Interrupt message from human
            state: Current graph state
            current_node: Node that just executed (or is about to execute)
            ctx: Invocation context

        Returns:
            InterruptAction with decision (action, reasoning, parameters)
        """
        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(message, state, current_node)

        logger.debug(
            f"InterruptReasoner: reasoning about interrupt at node '{current_node}'"
        )

        # Call LLM via self.run_async()
        try:
            content = genai.types.Content(
                role="user", parts=[genai.types.Part(text=prompt)]
            )
            node_ctx = ctx.model_copy(update={"user_content": content})

            response_text = ""
            async for event in self.run_async(node_ctx):
                if event.content and event.content.parts:
                    response_text = event.content.parts[0].text or ""

            # Parse JSON response
            decision = self._parse_llm_response(response_text)
            return InterruptAction(
                action=decision.get("action", "continue"),
                reasoning=decision.get("reasoning", ""),
                parameters=decision.get("parameters", {}),
            )

        except Exception as e:
            logger.error(f"InterruptReasoner: Error during reasoning: {e}")
            # Fallback to default action
            return InterruptAction(
                action="continue",
                reasoning=f"Failed to parse LLM response: {str(e)}",
                parameters={},
            )

    def _build_reasoning_prompt(
        self, message: InterruptMessage, state: GraphState, current_node: str
    ) -> str:
        """Build reasoning prompt for LLM.

        Args:
            message: Interrupt message
            state: Current graph state
            current_node: Current node name

        Returns:
            Formatted prompt string
        """
        # Truncate state if too large
        state_str = json.dumps(state.data, indent=2)
        if len(state_str) > self._config.max_state_size:
            state_str = state_str[: self._config.max_state_size] + "\n... (truncated)"

        # Build prompt
        prompt = f"""
Current Situation:
- Node: {current_node}
- State: {state_str if self._config.include_state_in_prompt else "<state hidden>"}
- Execution path: {state.metadata.get('path', [])}
- Iteration: {state.metadata.get('iteration', 'unknown')}

Human Interrupt Message:
{message.text}

Metadata: {json.dumps(message.metadata, indent=2) if message.metadata else "None"}

Available Actions: {', '.join(self._available_actions)}

Analyze the interrupt and decide what to do. Consider:
- What did the node just produce?
- What is the human asking for?
- Should we continue, rerun with guidance, go back, pause, or defer?

Respond with JSON only (no markdown):
{{
    "action": "<one of available actions>",
    "reasoning": "why you chose this action",
    "parameters": {{
        "target_node": "node name if go_back",
        "guidance": "guidance text if rerun",
        "message": "message if defer"
    }}
}}
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM JSON response.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed decision dictionary

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        decision: Dict[str, Any] = json.loads(text)

        # Validate action
        action = decision.get("action", "continue")
        if action not in self._available_actions:
            logger.warning(
                f"InterruptReasoner: Invalid action '{action}', defaulting to 'continue'"
            )
            decision["action"] = "continue"

        return decision
