"""
Integration tests for Message Routing Gym.

Tests the core environment logic directly (without HTTP layer) so they
can run in CI without a running server.

Run with:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from server.message_routing_environment import MessageRoutingEnvironment
from message_routing_gym.models import MessageRoutingAction
from message_routing_gym.constants import ActionType, DirectoryName
from message_routing_gym.rewards import RewardEngine
from message_routing_gym.graders import ProgrammaticGrader, SemanticGrader, CompositeGrader
from message_routing_gym.tasks import TASKS, TASK_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh environment instance for each test."""
    return MessageRoutingEnvironment(seed=42)


# ---------------------------------------------------------------------------
# Environment lifecycle
# ---------------------------------------------------------------------------

class TestEnvironmentLifecycle:

    def test_reset_returns_valid_observation(self, env):
        result = env.reset()
        obs = result.observation
        assert obs.task_id != "none"
        assert len(obs.queue) > 0
        assert obs.steps_remaining > 0
        assert obs.difficulty in ("warmup", "intermediate", "advanced")
        assert result.reward == 0.0
        assert result.done is False

    def test_step_valid_action_returns_observation(self, env):
        result = env.reset()
        first_msg = result.observation.queue[0]
        action = MessageRoutingAction(
            action_type=ActionType.DISMISS,
            message_id=first_msg.id,
        )
        step_result = env.step(action)
        assert step_result.observation is not None
        assert isinstance(step_result.reward, float)

    def test_step_invalid_message_id_penalised(self, env):
        env.reset()
        action = MessageRoutingAction(
            action_type=ActionType.DISMISS,
            message_id="nonexistent-id-999",
        )
        result = env.step(action)
        assert result.reward < 0
        assert "not found" in result.observation.last_execution_error.lower()

    def test_step_invalid_directory_penalised(self, env):
        reset_result = env.reset()
        msg = reset_result.observation.queue[0]
        action = MessageRoutingAction(
            action_type=ActionType.ROUTE_DIRECTORY,
            message_id=msg.id,
            target_directory="nonexistent_dir",
        )
        result = env.step(action)
        assert result.reward < 0

    def test_episode_terminates_after_max_steps(self, env):
        env.reset()
        done = False
        for _ in range(env.max_steps + 5):
            if done:
                break
            # Invalid action to burn steps quickly
            result = env.step(MessageRoutingAction(
                action_type=ActionType.DISMISS,
                message_id="bad_id",
            ))
            done = result.done
        assert done is True

    def test_multiple_resets_work(self, env):
        for _ in range(3):
            result = env.reset()
            assert result.observation.queue is not None
            assert result.done is False

    def test_done_env_returns_done(self, env):
        env.reset()
        env._done = True
        result = env.step(MessageRoutingAction(
            action_type=ActionType.DISMISS, message_id="1"
        ))
        assert result.done is True


# ---------------------------------------------------------------------------
# Task 1 — Noise Filter (complete episode)
# ---------------------------------------------------------------------------

class TestTask1NoiseFiler:

    def test_correct_routing_achieves_high_grade(self, env):
        """Route promos to promotions, ops to ops → grade ≥ 0.99."""
        result = env.reset(task_id="task_warmup_noise_filter")
        obs = result.observation

        correct_routes = {
            "2": DirectoryName.PROMOTIONS,
            "4": DirectoryName.PROMOTIONS,
            "1": DirectoryName.OPERATIONS,
            "3": DirectoryName.VAULT,
        }
        for msg_id, directory in correct_routes.items():
            result = env.step(MessageRoutingAction(
                action_type=ActionType.ROUTE_DIRECTORY,
                message_id=msg_id,
                target_directory=directory,
            ))

        assert result.done is True
        # grade is in info only on done; confirm high grade via env state
        assert env._last_grade >= 0.99

    def test_promo_in_operations_penalised(self, env):
        """Routing a promo to operations should reduce grade."""
        env.reset(task_id="task_warmup_noise_filter")
        result = env.step(MessageRoutingAction(
            action_type=ActionType.ROUTE_DIRECTORY,
            message_id="2",  # marketing promo
            target_directory=DirectoryName.OPERATIONS,
        ))
        # Grade should be 0 (wrong directory)
        assert result.info["grade"] < 0.5


# ---------------------------------------------------------------------------
# Task 2 — Stakeholder Acknowledgment
# ---------------------------------------------------------------------------

class TestTask2StakeholderAck:

    def test_respond_with_acknowledgment_scores_high(self, env):
        env.reset(task_id="task_intermediate_stakeholder_ack")

        # Vault the telemetry
        env.step(MessageRoutingAction(
            action_type=ActionType.ROUTE_DIRECTORY,
            message_id="1",
            target_directory=DirectoryName.VAULT,
        ))

        # Respond to VP
        result = env.step(MessageRoutingAction(
            action_type=ActionType.RESPOND,
            message_id="2",
            response_payload="Acknowledged. I have received and reviewed the updated timeline. Thanks.",
        ))

        assert result.info["grade"] >= 0.50

    def test_empty_response_scores_low(self, env):
        env.reset(task_id="task_intermediate_stakeholder_ack")
        env.step(MessageRoutingAction(
            action_type=ActionType.ROUTE_DIRECTORY,
            message_id="1",
            target_directory=DirectoryName.VAULT,
        ))
        result = env.step(MessageRoutingAction(
            action_type=ActionType.RESPOND,
            message_id="2",
            response_payload="",
        ))
        assert result.info["grade"] < 0.60


# ---------------------------------------------------------------------------
# Task 3 — Conflict Scheduling
# ---------------------------------------------------------------------------

class TestTask3ConflictScheduling:

    def test_correct_resolution_achieves_high_grade(self, env):
        env.reset(task_id="task_advanced_conflict_scheduling")

        # 1. Respond to DevOps with 15:00 (the safe window after DB lock ends at 14:30)
        env.step(MessageRoutingAction(
            action_type=ActionType.RESPOND,
            message_id="1",
            response_payload="The 14:00 slot conflicts with DB maintenance until 14:30. Let's target 15:00 as the safe deployment window.",
        ))
        # 2. Route DB alert to operations
        env.step(MessageRoutingAction(
            action_type=ActionType.ROUTE_DIRECTORY,
            message_id="2",
            target_directory=DirectoryName.OPERATIONS,
        ))
        # 3. Route vendor invite to promotions
        env.step(MessageRoutingAction(
            action_type=ActionType.ROUTE_DIRECTORY,
            message_id="3",
            target_directory=DirectoryName.PROMOTIONS,
        ))
        # 4. Dismiss DevOps message (respond doesn't remove from queue)
        result = env.step(MessageRoutingAction(
            action_type=ActionType.DISMISS,
            message_id="1",
        ))

        # Queue empty → done=True
        assert result.done is True
        # Grade reflects routing correctness + semantic quality of response
        assert env._last_grade >= 0.50


# ---------------------------------------------------------------------------
# Reward engine unit tests
# ---------------------------------------------------------------------------

class TestRewardEngine:

    def test_valid_route_gives_positive_reward(self):
        engine = RewardEngine()
        r = engine.compute_step_reward(
            action_type="route_directory",
            message_id="1",
            target_directory="promotions",
            is_valid_message=True,
            is_valid_directory=True,
            grade_delta=0.25,
        )
        assert r > 0

    def test_invalid_message_gives_negative_reward(self):
        engine = RewardEngine()
        r = engine.compute_step_reward(
            action_type="dismiss",
            message_id="ghost",
            target_directory="",
            is_valid_message=False,
            is_valid_directory=True,
            grade_delta=0.0,
        )
        assert r < 0

    def test_repeat_action_penalised(self):
        engine = RewardEngine()
        kwargs = dict(
            action_type="route_directory",
            message_id="1",
            target_directory="vault",
            is_valid_message=True,
            is_valid_directory=True,
            grade_delta=0.0,
        )
        r1 = engine.compute_step_reward(**kwargs)
        r2 = engine.compute_step_reward(**kwargs)
        assert r2 < r1  # second call penalised for repeat

    def test_resolution_bonus_scales_with_speed(self):
        engine = RewardEngine(max_steps=10)
        # Simulate 3 steps used
        for _ in range(3):
            engine.compute_step_reward(
                action_type="dismiss", message_id=str(_),
                target_directory="", is_valid_message=True,
                is_valid_directory=True, grade_delta=0.0,
            )
        bonus = engine.compute_resolution_bonus(1.0)
        assert bonus > 1.0  # efficiency-scaled


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestGraders:

    def test_programmatic_grader_correct(self):
        def _msg(id, dir_name):
            from message_routing_gym.models import MessageItem
            from message_routing_gym.constants import AlertLevel
            return MessageItem(id=id, source="x", topic="t", content="c", alert_level=AlertLevel.NORMAL)

        state = {
            "directories": {
                "promotions": [_msg("2", "promotions"), _msg("4", "promotions")],
                "operations": [_msg("1", "operations")],
                "vault": [],
                "management": [],
            },
            "dispatched_responses": [],
        }
        spec = {"required_route": {"2": "promotions", "4": "promotions"}}
        score = ProgrammaticGrader.grade(state, spec)
        assert score == 1.0

    def test_semantic_grader_concept_hit(self):
        score = SemanticGrader.grade(
            payload="Acknowledged. I have received and reviewed the timeline.",
            expected_concepts=["acknowledged", "received", "reviewed"],
        )
        assert score > 0.7

    def test_semantic_grader_negative_penalty(self):
        score = SemanticGrader.grade(
            payload="No, I cannot attend that meeting.",
            expected_concepts=["confirmed", "received"],
            negative_concepts=["no", "cannot"],
        )
        assert score < 0.3


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

class TestTaskRegistry:

    def test_all_tasks_in_registry(self):
        assert len(TASKS) == 3
        assert "task_warmup_noise_filter" in TASK_MAP
        assert "task_intermediate_stakeholder_ack" in TASK_MAP
        assert "task_advanced_conflict_scheduling" in TASK_MAP

    def test_each_task_has_queue(self):
        for task in TASKS:
            state = task.setup_state()
            assert "queue" in state
            assert len(state["queue"]) > 0
