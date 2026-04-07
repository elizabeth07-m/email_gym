"""
MessageRoutingEnvironment — core OpenEnv environment class.

Responsibilities:
  - Manage episode state (queue, directories, dispatched responses)
  - Integrate DifficultyManager for curriculum-based task selection
  - Delegate reward computation to RewardEngine
  - Delegate grading to CompositeGrader (via RoutingTask)
  - Produce clean MessageRoutingObservation after every step
"""

from __future__ import annotations

import random
import sys, os

# Make project root importable when running inside server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, Any, List, Optional, Tuple

from message_routing_gym.constants import (
    ActionType, DirectoryName, DIRECTORY_ORDER,
    TIER_LABELS, MAX_STEPS,
)
from message_routing_gym.models import (
    MessageRoutingAction,
    MessageRoutingObservation,
    StepResponse,
)
from message_routing_gym.rewards import RewardEngine
from message_routing_gym.tasks import TASKS, TASK_MAP, RoutingTask


# ---------------------------------------------------------------------------
# Difficulty manager
# ---------------------------------------------------------------------------

class DifficultyManager:
    """
    Tracks mastery to escalate curriculum tier.

    Escalation: 2 consecutive episodes with avg_grade ≥ 0.80 → tier up
    Regression:  2 consecutive episodes with avg_grade ≤ 0.20 → tier down
    """

    def __init__(self, max_tier: int = 3):
        self.history: List[float] = []
        self.current_max_tier: int = 1
        self.total_max_tier: int = max_tier
        self.mastery_log: List[Dict] = []

    def update_mastery(self, final_grade: float) -> None:
        self.history.append(final_grade)
        self.mastery_log.append({"grade": final_grade, "tier": self.current_max_tier})

        if len(self.history) >= 2:
            recent_avg = sum(self.history[-2:]) / 2.0
            if recent_avg >= 0.80 and self.current_max_tier < self.total_max_tier:
                self.current_max_tier += 1
                self.history.clear()
            elif recent_avg <= 0.20 and self.current_max_tier > 1:
                self.current_max_tier -= 1
                self.history.clear()

    @property
    def level_name(self) -> str:
        return TIER_LABELS.get(self.current_max_tier, "advanced")


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------

class MessageRoutingEnvironment:
    """
    OpenEnv-compliant environment core (without FastAPI routing).

    This class is instantiated once by the FastAPI application and mutated
    via reset() / step() calls from HTTP handlers.
    """

    def __init__(self, max_steps: int = MAX_STEPS, seed: Optional[int] = None):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.difficulty_mgr = DifficultyManager()
        self.reward_engine = RewardEngine(max_steps=max_steps)

        # Per-episode state ─────────────────────────────────────────
        self._current_task: Optional[RoutingTask] = None
        self._internal_state: Dict[str, Any] = {}
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._action_history: List[str] = []
        self._last_grade: float = 0.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> StepResponse:
        """Start a new episode. Returns the initial observation."""
        if seed is not None:
            self._rng.seed(seed)

        # Select task ───────────────────────────────────────────────
        if task_id and task_id in TASK_MAP:
            self._current_task = TASK_MAP[task_id]
        else:
            eligible = [t for t in TASKS if t.level_tier <= self.difficulty_mgr.current_max_tier]
            self._current_task = self._rng.choice(eligible)

        # Initialise episode state ──────────────────────────────────
        self._internal_state = self._current_task.setup_state()
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_history = []
        self._last_grade = 0.0
        self._done = False
        self.reward_engine.reset()

        obs = self._build_observation(reward=0.0, feedback="Episode started. Inbox loaded.")
        return StepResponse(
            observation=obs,
            reward=0.0,
            done=False,
            info={"curriculum_tier": self.difficulty_mgr.level_name, "task_id": self._current_task.task_id},
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: MessageRoutingAction) -> StepResponse:
        """Process one agent action and return the next observation + reward."""
        if self._done:
            obs = self._build_observation(reward=0.0, feedback="Episode is done. Please reset.")
            return StepResponse(observation=obs, reward=0.0, done=True, info={})

        self._step_count += 1
        error_msg = ""
        feedback_parts: List[str] = []

        # ── Locate message ─────────────────────────────────────────
        queue = self._internal_state["queue"]
        message = next((m for m in queue if m.id == action.message_id), None)

        is_valid_message = message is not None
        is_valid_directory = action.target_directory in DIRECTORY_ORDER or action.action_type != ActionType.ROUTE_DIRECTORY

        if not is_valid_message:
            error_msg = f"Message ID '{action.message_id}' not found in queue. Did the agent hallucinate?"
            feedback_parts.append(f"❌ Invalid message ID: {action.message_id}")

        elif action.action_type == ActionType.DISMISS:
            queue.remove(message)
            self._internal_state["directories"]["vault"].append(message)
            feedback_parts.append(f"📥 Dismissed '{message.topic}' → vault")

        elif action.action_type == ActionType.ROUTE_DIRECTORY:
            if action.target_directory not in DIRECTORY_ORDER:
                error_msg = f"Directory '{action.target_directory}' does not exist."
                feedback_parts.append(f"❌ Unknown directory: {action.target_directory}")
                is_valid_directory = False
            else:
                queue.remove(message)
                self._internal_state["directories"][action.target_directory].append(message)
                feedback_parts.append(
                    f"📂 Routed '{message.topic}' → {action.target_directory}"
                )

        elif action.action_type == ActionType.RESPOND:
            self._internal_state["dispatched_responses"].append(
                {"message_id": message.id, "payload": action.response_payload}
            )
            feedback_parts.append(
                f"📤 Response dispatched to '{message.source}' "
                f"({len(action.response_payload)} chars)"
            )

        # ── Grade current state ─────────────────────────────────────
        current_grade = self._current_task.grade(self._internal_state)
        grade_delta = max(0.0, current_grade - self._last_grade)
        self._last_grade = current_grade

        # ── Compute reward ──────────────────────────────────────────
        step_reward = self.reward_engine.compute_step_reward(
            action_type=action.action_type,
            message_id=action.message_id,
            target_directory=action.target_directory,
            is_valid_message=is_valid_message,
            is_valid_directory=is_valid_directory,
            grade_delta=grade_delta,
        )

        # ── Check done conditions ───────────────────────────────────
        done = False
        resolution_bonus = 0.0

        if current_grade >= 0.99:
            resolution_bonus = self.reward_engine.compute_resolution_bonus(current_grade)
            step_reward += resolution_bonus
            feedback_parts.append(
                f"🏆 Directive fully resolved! Bonus: +{resolution_bonus:.2f}"
            )
            done = True

        if not queue:
            feedback_parts.append("📭 Queue empty.")
            done = True

        if self._step_count >= self.max_steps:
            if not done:
                # Missed the mark — apply timeout floor
                timeout_delta = self.reward_engine.apply_timeout_floor()
                step_reward += timeout_delta
                feedback_parts.append(
                    f"⏰ Episode timed out. Timeout floor applied ({timeout_delta:+.2f})."
                )
            done = True

        self._cumulative_reward += step_reward
        self._done = done

        # ── Record action history ───────────────────────────────────
        history_entry = (
            f"Step {self._step_count}: {action.action_type} on #{action.message_id}"
        )
        if action.target_directory:
            history_entry += f" → {action.target_directory}"
        self._action_history.append(history_entry)

        if done:
            self.difficulty_mgr.update_mastery(current_grade)
            feedback_parts.append(f"📊 Final Grade: {current_grade * 100:.1f}%")

        obs = self._build_observation(
            reward=step_reward,
            feedback=" | ".join(feedback_parts),
            error=error_msg,
            grade=current_grade if done else None,
        )

        return StepResponse(
            observation=obs,
            reward=step_reward,
            done=done,
            info={
                "grade": current_grade,
                "curriculum_tier": self.difficulty_mgr.level_name,
                "resolution_bonus": resolution_bonus,
                "step_count": self._step_count,
            },
        )

    # ------------------------------------------------------------------
    # state (unrestricted, for LLM judge / grader)
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return the full environment state (not filtered for the agent)."""
        return {
            "task_id": self._current_task.task_id if self._current_task else None,
            "step_count": self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "grade": self._last_grade,
            "done": self._done,
            "queue_size": len(self._internal_state.get("queue", [])),
            "directories": {
                k: [m.id for m in v]
                for k, v in self._internal_state.get("directories", {}).items()
            },
            "dispatched_responses": self._internal_state.get("dispatched_responses", []),
            "curriculum_tier": self.difficulty_mgr.level_name,
            "mastery_log": self.difficulty_mgr.mastery_log[-10:],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        *,
        reward: float,
        feedback: str,
        error: str = "",
        grade: Optional[float] = None,
    ) -> MessageRoutingObservation:
        dirs = self._internal_state.get("directories", {})
        dir_counts = {k: len(v) for k, v in dirs.items()}

        return MessageRoutingObservation(
            task_id=self._current_task.task_id if self._current_task else "none",
            difficulty=self.difficulty_mgr.level_name,
            active_directive=self._current_task.description if self._current_task else "",
            queue=list(self._internal_state.get("queue", [])),
            directories=dir_counts,
            step_feedback=feedback,
            last_execution_error=error,
            steps_remaining=max(0, self.max_steps - self._step_count),
            cumulative_reward=self._cumulative_reward,
            action_history=list(self._action_history),
            done=self._done,
            reward=reward,
            grader_score=grade,
        )
