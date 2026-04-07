"""
RewardEngine — dense, multi-signal reward computation for Message Routing Gym.

Design goals (matching kube-sre-gym reward philosophy):
  - Per-step signals: small immediate rewards for valid actions
  - Repeat-action penalty: discourage mindless repetition
  - Efficiency-scaled resolution bonus: faster fixes → higher bonus
  - Timeout floor: failed episodes wiped to -2.0 so GRPO gets clear variance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Optional

from message_routing_gym.constants import (
    REWARD_ROUTE_STEP,
    REWARD_RESPOND_STEP,
    REWARD_GRADE_WEIGHT,
    REWARD_RESOLUTION_BONUS,
    REWARD_INVALID_ACTION,
    REWARD_BAD_DIRECTORY,
    REWARD_REPEAT_PENALTY,
    REWARD_TIMEOUT_FLOOR,
    MAX_STEPS,
)


@dataclass
class RewardEngine:
    """
    Stateful per-episode reward computer.

    Tracks repeated actions and applies efficiency scaling to resolution bonuses.
    Must be reset at the start of each episode.
    """
    max_steps: int = MAX_STEPS
    _step_count: int = field(default=0, init=False, repr=False)
    _cumulative: float = field(default=0.0, init=False, repr=False)
    _action_fingerprints: Set[str] = field(default_factory=set, init=False, repr=False)

    def reset(self) -> None:
        """Reset all episode state. Call at the start of each episode."""
        self._step_count = 0
        self._cumulative = 0.0
        self._action_fingerprints = set()

    # ------------------------------------------------------------------
    # Per-step reward
    # ------------------------------------------------------------------

    def compute_step_reward(
        self,
        *,
        action_type: str,
        message_id: str,
        target_directory: str,
        is_valid_message: bool,
        is_valid_directory: bool,
        grade_delta: float,
    ) -> float:
        """
        Compute the reward signal for a single environment step.

        Parameters
        ----------
        action_type:       "dismiss" | "route_directory" | "respond"
        message_id:        The message ID the agent targeted
        target_directory:  The directory chosen (empty string if not routing)
        is_valid_message:  Whether message_id existed in the queue
        is_valid_directory: Whether target_directory is a valid directory
        grade_delta:       Change in composite grade score since last step (0–1)

        Returns
        -------
        float — step reward (may be negative)
        """
        self._step_count += 1
        reward = 0.0

        # ── Invalid message ID (agent hallucinated) ──────────────────
        if not is_valid_message:
            reward += REWARD_INVALID_ACTION
            self._cumulative += reward
            return reward

        # ── Invalid directory ─────────────────────────────────────────
        if action_type == "route_directory" and not is_valid_directory:
            reward += REWARD_BAD_DIRECTORY
            self._cumulative += reward
            return reward

        # ── Base per-action reward ────────────────────────────────────
        if action_type == "respond":
            reward += REWARD_RESPOND_STEP
        else:
            reward += REWARD_ROUTE_STEP

        # ── Grade-weighted signal ─────────────────────────────────────
        reward += grade_delta * REWARD_GRADE_WEIGHT

        # ── Repeat action penalty ─────────────────────────────────────
        fingerprint = f"{action_type}:{message_id}:{target_directory}"
        if fingerprint in self._action_fingerprints:
            reward += REWARD_REPEAT_PENALTY
        else:
            self._action_fingerprints.add(fingerprint)

        self._cumulative += reward
        return reward

    # ------------------------------------------------------------------
    # Episode completion rewards
    # ------------------------------------------------------------------

    def compute_resolution_bonus(self, grade: float) -> float:
        """
        Efficiency-scaled resolution bonus. Faster resolution → higher bonus.

        A grade of 1.0 at step 1 yields the maximum bonus.
        A grade of 0.5 at the last step yields a proportionally smaller bonus.
        """
        if grade < 0.99:
            return 0.0
        steps_used = self._step_count
        steps_ratio = 1.0 - (steps_used / self.max_steps)  # 1.0 = perfect speed
        bonus = REWARD_RESOLUTION_BONUS * (1.0 + steps_ratio)  # range: 1.0× to 2.0×
        return round(bonus, 4)

    def apply_timeout_floor(self) -> float:
        """
        Wipe the episode reward to the timeout floor if the episode timed out
        without resolution. Returns the correction delta applied.
        """
        if self._cumulative > REWARD_TIMEOUT_FLOOR:
            delta = REWARD_TIMEOUT_FLOOR - self._cumulative
            self._cumulative = REWARD_TIMEOUT_FLOOR
            return delta
        return 0.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def cumulative(self) -> float:
        return self._cumulative

    @property
    def step_count(self) -> int:
        return self._step_count
