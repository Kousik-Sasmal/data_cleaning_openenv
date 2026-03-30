# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Cleaning Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import DataCleaningAction, DataCleaningObservation


class DataCleaningEnv(
    EnvClient[DataCleaningAction, DataCleaningObservation, State]
):
    """
    Client for the Data Cleaning Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DataCleaningEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     result = client.step(DataCleaningAction(command="drop_duplicates", params={}))
        ...     print(result.observation.message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DataCleaningEnv.from_docker_image("data-cleaning-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DataCleaningAction(command="drop_duplicates", params={}))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DataCleaningAction) -> Dict:
        """
        Convert DataCleaningAction to JSON payload for step message.

        Args:
            action: DataCleaningAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "command": action.command,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DataCleaningObservation]:
        """
        Parse server response into StepResult[DataCleaningObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DataCleaningObservation
        """
        obs_data = payload.get("observation", {})
        observation = DataCleaningObservation(
            dataset_preview=obs_data.get("dataset_preview", ""),
            schema_info=obs_data.get("schema_info", ""),
            message=obs_data.get("message", ""),
            current_score=obs_data.get("current_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )





async def main():
    # 1. Connect to your running local server (async context manager)
    print("--- Connecting to Data Cleaning Environment (Async) ---")
    
    async with DataCleaningEnv(base_url="http://localhost:8000") as client:
        
        # 2. Reset the environment
        print("\n[Action] Resetting Environment...")
        result = await client.reset()
        print(f"Server Response: {result.observation.message}")
        print(f"Schema: {result.observation.schema_info}")
        print(f"Preview: {result.observation.dataset_preview}")
        
        # 3. Take a step: Send a message
        print("\n[Action] Sending Data Cleaning Command...")
        # Note: We use await here because client.step is an async call
        result = await client.step(DataCleaningAction(command="drop_duplicates", params={}))
        
        # 4. Print the results from the observation
        print(f"System Message: {result.observation.message}")
        print(f"New Preview: {result.observation.dataset_preview}")
        print(f"Step Reward: {result.reward}")
        print(f"Is Done?: {result.done}")
        print(f"Grader Score: {result.observation.current_score}")

        # 5. Optional: Check the state
        # State is usually a property, but in some async clients it might require await
        state = await client.state()
        print(f"\n--- Session State ---")
        print(f"Episode ID: {state.episode_id}")
        print(f"Total Steps: {state.step_count}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())