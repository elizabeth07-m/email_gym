"""
Task definitions for Message Routing Gym.

Each RoutingTask encapsulates:
  - A narrative scenario with an active_directive
  - An initial environment state
  - A grading_spec for CompositeGrader

Difficulty tiers mirror the hallucination-detector-gym pattern:
  Tier 1 — Warmup:       simple routing, one clear signal
  Tier 2 — Intermediate: requires generating a contextual response
  Tier 3 — Advanced:     multi-message triage with conflicting signals and
                          time-critical scheduling constraints
"""

from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass, field

from message_routing_gym.constants import AlertLevel, DIRECTORY_ORDER
from message_routing_gym.graders import CompositeGrader


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class RoutingTask:
    """Abstract base for all routing tasks."""
    task_id: str
    description: str          # active_directive shown to the agent
    difficulty: str           # warmup | intermediate | advanced
    level_tier: int           # 1 | 2 | 3 — controls curriculum escalation
    grading_spec: Dict[str, Any] = field(default_factory=dict)
    response_checks: List[Dict[str, Any]] = field(default_factory=list)
    programmatic_weight: float = 0.60
    semantic_weight: float = 0.40

    def setup_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def grade(self, state: Dict[str, Any]) -> float:
        grader = CompositeGrader(
            programmatic_weight=self.programmatic_weight,
            semantic_weight=self.semantic_weight,
        )
        return grader.grade(
            state,
            grading_spec=self.grading_spec,
            response_checks=self.response_checks,
        )

    def _empty_directories(self) -> Dict[str, List]:
        return {d: [] for d in DIRECTORY_ORDER}

    def _base_state(self) -> Dict[str, Any]:
        return {
            "directories": self._empty_directories(),
            "dispatched_responses": [],
            "purged_messages": [],
        }


# ---------------------------------------------------------------------------
# Task 1 — Warmup: Noise Filtering
# ---------------------------------------------------------------------------

class Task1_NoiseFiler(RoutingTask):
    """
    Tier 1 — Warmup

    The agent receives a mixed queue of promotional noise and legitimate
    operational mail. Objective: sort them correctly without contaminating
    the operations pipeline with marketing content.

    Real-world analogue: an on-call engineer's alert inbox flooded with
    vendor newsletters and a genuine P1 incident alert.
    """

    def __init__(self):
        super().__init__(
            task_id="task_warmup_noise_filter",
            description=(
                "[Warmup] Noise Filter: Route all promotional/vendor broadcasts "
                "to 'promotions'. Route legitimate operational mail to 'operations' "
                "or 'vault'. Do NOT mix them."
            ),
            difficulty="warmup",
            level_tier=1,
            grading_spec={
                "required_route": {"2": "promotions", "4": "promotions"},
                "forbidden_routes": {"1": "promotions", "3": "promotions"},
            },
            programmatic_weight=1.0,
            semantic_weight=0.0,
        )

    def setup_state(self) -> Dict[str, Any]:
        from message_routing_gym.models import MessageItem
        state = self._base_state()
        state["queue"] = [
            MessageItem(id="1", source="internal-sys@ops.net",  topic="Build Pipeline: artifact deployed",
                        content="Artifact v2.4.1 deployed successfully to staging.", alert_level=AlertLevel.NORMAL),
            MessageItem(id="2", source="marketing@vendor.com",  topic="Exclusive Offer — 50% off!",
                        content="Claim your limited-time discount before midnight.", alert_level=AlertLevel.LOW),
            MessageItem(id="3", source="cto@hq.net",           topic="Architecture Review — action required",
                        content="Please review the updated microservices design doc by EOD.", alert_level=AlertLevel.HIGH),
            MessageItem(id="4", source="sales@tooling.io",     topic="Your trial is expiring",
                        content="Upgrade now to keep access to premium features.", alert_level=AlertLevel.LOW),
        ]
        return state


# ---------------------------------------------------------------------------
# Task 2 — Intermediate: Stakeholder Acknowledgment
# ---------------------------------------------------------------------------

class Task2_StakeholderAck(RoutingTask):
    """
    Tier 2 — Intermediate

    The agent must identify a high-priority management request requiring a
    polite, professional acknowledgment response. Low-signal automated telemetry
    should be filed to vault.

    Real-world analogue: a VP sends a critical-path update request buried
    among a wall of cron-job metric digests.
    """

    def __init__(self):
        super().__init__(
            task_id="task_intermediate_stakeholder_ack",
            description=(
                "[Intermediate] Stakeholder Sync: Identify the VP Engineering message "
                "and dispatch a polite acknowledgment confirming receipt. "
                "File automated metric reports to 'vault'."
            ),
            difficulty="intermediate",
            level_tier=2,
            grading_spec={
                "required_route": {"1": "vault"},
                "required_response_ids": ["2"],
            },
            response_checks=[
                {
                    "message_id": "2",
                    "expected_concepts": ["received", "acknowledged", "got it", "noted", "review"],
                    "negative_concepts": ["no", "busy", "cannot", "unavailable"],
                }
            ],
            programmatic_weight=0.50,
            semantic_weight=0.50,
        )

    def setup_state(self) -> Dict[str, Any]:
        from message_routing_gym.models import MessageItem
        state = self._base_state()
        state["queue"] = [
            MessageItem(id="1", source="metrics@monitoring.local", topic="Daily Node Health Report",
                        content="Cluster CPU at 43%, memory at 61%. No anomalies detected.", alert_level=AlertLevel.LOW),
            MessageItem(id="2", source="vp_engineering@hq.net",    topic="Critical Path Tracker — Review Required",
                        content="Please confirm you have seen the updated Q2 delivery timeline before the 3 PM standup.",
                        alert_level=AlertLevel.HIGH),
        ]
        return state


# ---------------------------------------------------------------------------
# Task 3 — Advanced: Conflicting Scheduling Triage
# ---------------------------------------------------------------------------

class Task3_ConflictScheduling(RoutingTask):
    """
    Tier 3 — Advanced (Adversarial)

    Three messages arrive simultaneously: a DevOps engineer proposes a deployment
    window that conflicts with an active database maintenance lock, and a vendor
    sends a red-herring integration webinar invite.

    The agent must:
      1. Respond to DevOps proposing 15:00 (not 14:00, which conflicts with DB lock)
      2. Route the DB maintenance alert to 'operations'
      3. Route the vendor invite to 'promotions'

    Red herring: the webinar invite is marked HIGH priority but is irrelevant noise.
    Real-world analogue: an SRE triaging conflicting on-call notifications with
    a production deployment window request.
    """

    def __init__(self):
        super().__init__(
            task_id="task_advanced_conflict_scheduling",
            description=(
                "[Advanced] Conflict Triage: DevOps wants to deploy at 14:00 but DB "
                "maintenance locks the system 13:00–14:30. Respond to DevOps proposing "
                "'15:00' as the safe window. Route the DB alert to 'operations'. "
                "Route the vendor invite to 'promotions'. Do not be misled by alert levels."
            ),
            difficulty="advanced",
            level_tier=3,
            grading_spec={
                "required_route": {"2": "operations", "3": "promotions"},
                "required_response_ids": ["1"],
                "forbidden_routes": {"2": "vault", "2": "promotions"},  # type: ignore[dict-item]
            },
            response_checks=[
                {
                    "message_id": "1",
                    "expected_concepts": ["15:00", "3 pm", "1500", "after 14:30", "safe window"],
                    "negative_concepts": ["14:00", "2 pm", "1400"],
                }
            ],
            programmatic_weight=0.55,
            semantic_weight=0.45,
        )

    def setup_state(self) -> Dict[str, Any]:
        from message_routing_gym.models import MessageItem
        state = self._base_state()
        state["queue"] = [
            MessageItem(id="1", source="devops@ops.net",       topic="Deployment Window Request",
                        content="Can we push the v3.1 release at 14:00 today? Need sign-off ASAP.",
                        alert_level=AlertLevel.NORMAL),
            MessageItem(id="2", source="system@cron.local",    topic="[SCHEDULED] DB Maintenance Lock",
                        content="Database cluster locked for maintenance from 13:00 to 14:30. "
                                "All writes will be blocked during this window.",
                        alert_level=AlertLevel.CRITICAL),
            MessageItem(id="3", source="partner@vendor.com",   topic="Join our Integration Webinar",
                        content="Live demo of our latest DevOps integration features. "
                                "Register now — limited seats.",
                        alert_level=AlertLevel.HIGH),
        ]
        return state


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: List[RoutingTask] = [
    Task1_NoiseFiler(),
    Task2_StakeholderAck(),
    Task3_ConflictScheduling(),
]

TASK_MAP: Dict[str, RoutingTask] = {t.task_id: t for t in TASKS}
