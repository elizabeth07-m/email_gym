"""
Message Routing Gym — OpenEnv-compliant RL environment.

A self-improving environment where an agent learns to triage, route,
and respond to operational messages through adversarial curricula and GRPO.

Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon.
"""

from message_routing_gym.models import (
    MessageRoutingAction,
    MessageRoutingObservation,
    MessageItem,
)
from message_routing_gym.constants import AlertLevel, ActionType, DirectoryName, DifficultyTier

__all__ = [
    "MessageRoutingAction",
    "MessageRoutingObservation",
    "MessageItem",
    "AlertLevel",
    "ActionType",
    "DirectoryName",
    "DifficultyTier",
]

__version__ = "0.1.0"
