"""
Message Routing Gym — HTTP Client.

Connects to the FastAPI OpenEnv server (locally or on HF Spaces) and exposes
the same reset() / step() interface used during GRPO training.

Mirrors client.py structure from kube-sre-gym:
    https://github.com/sid-rp/kube-sre-gym/blob/main/client.py

Usage
-----
    from client import MessageRoutingEnvClient
    from message_routing_gym import MessageRoutingAction

    with MessageRoutingEnvClient(base_url="http://localhost:8000") as client:
        obs_result = client.reset()
        step_result = client.step(
            MessageRoutingAction(action_type="route_directory",
                                 message_id="1",
                                 target_directory="promotions")
        )
        print(step_result.reward)

Or via the module-level convenience alias:
    from message_routing_gym import MessageRoutingEnv   # resolves to this class
"""

from __future__ import annotations

import os
import json
import logging
from contextlib import contextmanager

import httpx

from message_routing_gym.models import (
    MessageRoutingAction,
    MessageRoutingObservation,
    ResetRequest,
    StepResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_URL = os.getenv("OPENENV_URL", "http://localhost:8000")


class MessageRoutingEnvClient:
    """
    Thread-safe HTTP client for the Message Routing Gym OpenEnv server.

    Parameters
    ----------
    base_url : str
        Root URL of the OpenEnv server (e.g. "https://elizabeth07-m-email-gym.hf.space").
    timeout  : float
        Request timeout in seconds.
    """

    def __init__(self, base_url: str = _DEFAULT_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._client.close()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Ping the server health endpoint."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> StepResponse:
        """
        Reset the environment and start a new episode.

        Parameters
        ----------
        task_id : optional — force a specific task scenario
        seed    : optional — random seed for reproducibility
        """
        payload = ResetRequest(task_id=task_id, seed=seed).model_dump(exclude_none=True)
        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return StepResponse.model_validate(resp.json())

    def step(self, action: MessageRoutingAction) -> StepResponse:
        """
        Send one action and receive the next observation + reward.

        Parameters
        ----------
        action : MessageRoutingAction
        """
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResponse.model_validate(resp.json())

    def state(self) -> dict:
        """Return the full unrestricted environment state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def schema(self) -> dict:
        """Return action / observation JSON schemas."""
        resp = self._client.get("/schema")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Convenience alias (matches import pattern expected by train.py)
# ---------------------------------------------------------------------------

MessageRoutingEnv = MessageRoutingEnvClient


# ---------------------------------------------------------------------------
# Quick sanity-check if run as main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_URL
    print(f"Connecting to {url}...")

    with MessageRoutingEnvClient(base_url=url) as client:
        print("Health:", client.health())

        result = client.reset()
        print(f"\n[RESET] Task: {result.observation.task_id}")
        print(f"  Queue: {len(result.observation.queue)} messages")
        print(f"  Directive: {result.observation.active_directive[:80]}...")

        if result.observation.queue:
            first_msg = result.observation.queue[0]
            action = MessageRoutingAction(
                action_type="route_directory",
                message_id=first_msg.id,
                target_directory="vault",
            )
            step_result = client.step(action)
            print(f"\n[STEP 1] Routed #{first_msg.id} → vault")
            print(f"  Reward: {step_result.reward:+.3f}")
            print(f"  Done:   {step_result.done}")
            print(f"  Grade:  {step_result.info.get('grade', 0)*100:.1f}%")

    print("\n✅ Client test passed.")
