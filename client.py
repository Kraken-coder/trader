# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trader Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TraderAction, TraderObservation


class TraderEnv(
    EnvClient[TraderAction, TraderObservation, State]
):
    """
    Client for the Trader Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TraderEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(TraderAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TraderEnv.from_docker_image("trader-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TraderAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TraderAction) -> Dict:
        """
        Convert TraderAction to JSON payload for step message.

        Args:
            action: TraderAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "position": action.position,
            "take_profit_price": action.take_profit_price,
            "stop_loss_price": action.stop_loss_price,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TraderObservation]:
        """
        Parse server response into StepResult[TraderObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TraderObservation
        """
        obs_data = payload.get("observation", {})
        observation = TraderObservation(
            market_features=obs_data.get("market_features", {}),
            symbol=obs_data.get("symbol", ""),
            open_price=obs_data.get("open_price", 0.0),
            high_price=obs_data.get("high_price", 0.0),
            low_price=obs_data.get("low_price", 0.0),
            current_price=obs_data.get("current_price", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
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
