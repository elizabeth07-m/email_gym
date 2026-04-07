# message_routing_gym/__init__.py (root package)
"""
Root package init — re-exports MessageRoutingEnv HTTP client for convenience.

Usage
-----
    from message_routing_gym import MessageRoutingAction, MessageRoutingEnv

    with MessageRoutingEnv(base_url="http://localhost:8000") as env:
        obs = env.reset()
        result = env.step(MessageRoutingAction(
            action_type="route_directory",
            message_id="1",
            target_directory="promotions",
        ))
"""

from message_routing_gym.models import (
    MessageRoutingAction,
    MessageRoutingObservation,
    MessageItem,
    StepResponse,
)
from message_routing_gym.constants import AlertLevel, ActionType, DirectoryName, DifficultyTier

__all__ = [
    "MessageRoutingAction",
    "MessageRoutingObservation",
    "MessageItem",
    "StepResponse",
    "AlertLevel",
    "ActionType",
    "DirectoryName",
    "DifficultyTier",
]

__version__ = "0.1.0"
