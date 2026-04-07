"""
Pydantic models for the Message Routing Gym.

These define the exact JSON schemas for the OpenEnv /reset and /step endpoints.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from message_routing_gym.constants import AlertLevel, ActionType


# ---------------------------------------------------------------------------
# Environment state primitives
# ---------------------------------------------------------------------------

class MessageItem(BaseModel):
    """A single message sitting in the agent's routing queue."""
    id: str = Field(description="Unique string identifier for this message.")
    source: str = Field(description="Originating address / system name.")
    topic: str = Field(description="Subject line or event type.")
    content: str = Field(description="Full body text of the message.")
    alert_level: AlertLevel = Field(description="Priority signal: low | normal | high | critical.")
    acknowledged: bool = Field(default=False, description="Whether the agent has acted on this message.")


# ---------------------------------------------------------------------------
# Observation  (what the agent receives from /reset and /step)
# ---------------------------------------------------------------------------

class MessageRoutingObservation(BaseModel):
    """
    Complete observation returned to the agent after each environment step.
    Mirrors the HallucinationObservation pattern from the winning project.
    """
    task_id: str = Field(description="Unique identifier for the current routing scenario.")
    difficulty: str = Field(description="Current curriculum tier: warmup | intermediate | advanced.")
    active_directive: str = Field(description="Natural language instructions for this episode.")

    # Live queue
    queue: List[MessageItem] = Field(
        description="Messages currently awaiting routing decisions."
    )
    # Sorted directory counts
    directories: Dict[str, int] = Field(
        default_factory=lambda: {"promotions": 0, "vault": 0, "operations": 0, "management": 0},
        description="Count of messages sorted into each directory so far."
    )

    # Per-step feedback
    step_feedback: str = Field(
        default="",
        description="Human-readable feedback about the last action taken."
    )
    last_execution_error: str = Field(
        default="",
        description="Error message if the last action was invalid."
    )
    steps_remaining: int = Field(description="Steps left before episode timeout.")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated this episode.")
    action_history: List[str] = Field(
        default_factory=list,
        description="String summary of each action taken so far."
    )

    # Terminal signals
    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: float = Field(default=0.0, description="Reward earned on this step.")
    grader_score: Optional[float] = Field(
        default=None,
        description="Final composite grader score (0.0–1.0) on episode completion."
    )


# ---------------------------------------------------------------------------
# Action  (what the agent sends to /step)
# ---------------------------------------------------------------------------

class MessageRoutingAction(BaseModel):
    """
    Agent action sent to the POST /step endpoint.

    Examples
    --------
    Route message "3" to promotions:
        MessageRoutingAction(action_type="route_directory", message_id="3",
                             target_directory="promotions")

    Respond to message "2" with acknowledgment:
        MessageRoutingAction(action_type="respond", message_id="2",
                             response_payload="Acknowledged. Will review immediately.")

    Dismiss message "1" to vault:
        MessageRoutingAction(action_type="dismiss", message_id="1")
    """
    action_type: ActionType = Field(
        description="The operation to perform: dismiss | route_directory | respond."
    )
    message_id: str = Field(
        description="The 'id' field of the target MessageItem in the queue."
    )
    target_directory: str = Field(
        default="",
        description="Required when action_type=route_directory. "
                    "Valid values: promotions | vault | operations | management."
    )
    response_payload: str = Field(
        default="",
        description="Required when action_type=respond. The reply text to dispatch."
    )
    reasoning: str = Field(
        default="",
        description="Optional chain-of-thought explanation from the agent."
    )


# ---------------------------------------------------------------------------
# Convenience wrappers for the HTTP API
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Optional parameters for POST /reset."""
    task_id: Optional[str] = Field(
        default=None,
        description="Force a specific task scenario. If None, selected by curriculum."
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")


class StepResponse(BaseModel):
    """Envelope returned by POST /step."""
    observation: MessageRoutingObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
