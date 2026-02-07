"""Basic GraphAgent example demonstrating conditional routing.

This example shows how GraphAgent enables conditional workflow routing
based on runtime state, which cannot be achieved with SequentialAgent
or ParallelAgent composition.

Use case: Data validation pipeline with retry logic.
- If validation passes -> process data
- If validation fails -> retry validation
- After max retries -> route to error handler
"""

from google.adk.agents import GraphAgent, GraphNode, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# --- Validator Agent ---
validator = LlmAgent(
    name="validator",
    model="gemini-2.0-flash",
    instruction="""
    You validate input data quality.
    Check if the input contains valid JSON.
    Return {"valid": true} if valid, {"valid": false, "error": "reason"} if invalid.
    """,
)

# --- Processor Agent ---
processor = LlmAgent(
    name="processor",
    model="gemini-2.0-flash",
    instruction="""
    You process validated data.
    Transform the input JSON and return processed results.
    """,
)

# --- Error Handler Agent ---
error_handler = LlmAgent(
    name="error_handler",
    model="gemini-2.0-flash",
    instruction="""
    You handle validation errors.
    Provide helpful error messages and suggestions for fixing invalid data.
    """,
)

# --- Create GraphAgent with Conditional Routing ---
graph = GraphAgent(name="validation_pipeline")

# Add nodes
graph.add_node(GraphNode(name="validate", agent=validator))
graph.add_node(GraphNode(name="process", agent=processor))
graph.add_node(GraphNode(name="error", agent=error_handler))

# Add conditional edges
# If validation passes (state.data["valid"] == True) -> process
graph.add_edge(
    "validate",
    "process",
    condition=lambda state: state.data.get("valid", False) is True,
)

# If validation fails (state.data["valid"] == False) -> error handler
graph.add_edge(
    "validate",
    "error",
    condition=lambda state: state.data.get("valid", False) is False,
)

# Define workflow
graph.set_start("validate")
graph.set_end("process")  # Success path ends at process
graph.set_end("error")  # Error path ends at error handler

# --- Run the workflow ---
if __name__ == "__main__":
    import asyncio

    async def main():
        runner = Runner(
            app_name="validation_pipeline",
            agent=graph,
            session_service=InMemorySessionService(),
        )

        # Example: Valid input
        print("=== Testing with valid JSON ===")
        async for event in runner.run_async(
            user_id="user_1",
            session_id="session_1",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text='{"name": "John", "age": 30}')],
            ),
        ):
            if event.content and event.content.parts:
                print(f"{event.author}: {event.content.parts[0].text}")

        # Example: Invalid input
        print("\n=== Testing with invalid JSON ===")
        async for event in runner.run_async(
            user_id="user_1",
            session_id="session_2",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text='{"name": "Invalid data')],
            ),
        ):
            if event.content and event.content.parts:
                print(f"{event.author}: {event.content.parts[0].text}")

    asyncio.run(main())
