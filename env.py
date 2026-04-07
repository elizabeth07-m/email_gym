from pydantic import BaseModel
from typing import Dict, Any, List
import random

from models import InboxState, AgentAction, AgentActionType
from tasks import TASKS

class DifficultyManager:
    """
    Tracks agent mastery to dynamically escalate difficulty.
    This manages the learning curve and provides a structured curriculum.
    """
    def __init__(self, max_tier: int = 3):
        self.history = []
        self.current_max_tier = 1
        self.total_max_tier = max_tier
        
    def update_mastery(self, final_grade: float):
        self.history.append(final_grade)
        if len(self.history) >= 2:
            recent_avg = sum(self.history[-2:]) / 2.0
            if recent_avg >= 0.8 and self.current_max_tier < self.total_max_tier:
                self.current_max_tier += 1
                self.history.clear() 
            elif recent_avg <= 0.2 and self.current_max_tier > 1:
                self.current_max_tier -= 1 
                self.history.clear()

    @property
    def level_name(self):
        return {1: "Warmup", 2: "Intermediate", 3: "Advanced"}.get(self.current_max_tier, "Expert")


class StepResult(BaseModel):
    observation: InboxState
    reward: float
    done: bool
    info: Dict[str, Any]


class MessageRoutingEnv:
    """
    OpenEnv compliant environment for playing a Message Routing scenario.
    Now equipped with an Adversarial Curriculum and LLM Judge capabilities.
    """
    def __init__(self):
        self.difficulty_mgr = DifficultyManager()
        self.current_task_idx = 0
        self.internal_state = {}
        self.step_count = 0
        self.max_steps = 10
        self.total_episode_reward = 0.0
        
    def reset(self) -> StepResult:
        valid_tasks = [i for i, t in enumerate(TASKS) if t.level_tier <= self.difficulty_mgr.current_max_tier]
        self.current_task_idx = random.choice(valid_tasks)
        task = TASKS[self.current_task_idx]
        
        self.internal_state = task.setup_state()
        self.step_count = 0
        self.total_episode_reward = 0.0
        
        obs = self._get_observation()
        return StepResult(observation=obs, reward=0.0, done=False, info={"curriculum_tier": self.difficulty_mgr.level_name})
        
    def state(self) -> Dict[str, Any]:
        """Returns the full internal unrestricted state."""
        return self.internal_state
        
    def step(self, action: AgentAction) -> StepResult:
        self.step_count += 1
        reward = 0.0
        done = False
        error_msg = ""
        
        task = TASKS[self.current_task_idx]
        
        message_found = None
        for m in self.internal_state["queue"]:
            if m.id == action.message_id:
                message_found = m
                break
                
        if not message_found:
            error_msg = f"Message ID {action.message_id} not found in queue. Agent hallucinated ID?"
            reward -= 0.2
        else:
            if action.action_type == AgentActionType.DISMISS:
                self.internal_state["queue"].remove(message_found)
                self.internal_state["directories"]["vault"].append(message_found)
                reward += 0.05
                
            elif action.action_type == AgentActionType.ROUTE_DIRECTORY:
                if action.target_directory in self.internal_state["directories"]:
                    self.internal_state["queue"].remove(message_found)
                    self.internal_state["directories"][action.target_directory].append(message_found)
                    reward += 0.05
                else:
                    error_msg = f"Directory '{action.target_directory}' does not exist."
                    reward -= 0.1
                    
            elif action.action_type == AgentActionType.RESPOND:
                self.internal_state["dispatched_responses"].append({
                    "message_id": message_found.id,
                    "payload": action.response_payload
                })
                # Responding doesn't auto-dismiss. Agent must explicitly route/dismiss the message.
                reward += 0.1
                
        current_grade = task.grade(self.internal_state)
        
        reward += (current_grade * 0.5)
        
        if current_grade >= 0.99:
            reward += 1.0 
            done = True
            
        if len(self.internal_state["queue"]) == 0:
            done = True
            
        if self.step_count >= self.max_steps:
            done = True
            
        self.total_episode_reward += reward
        
        if done:
            self.difficulty_mgr.update_mastery(current_grade)
            
        obs = self._get_observation()
        obs.last_execution_error = error_msg
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "grade": current_grade,
                "curriculum_tier": self.difficulty_mgr.level_name,
                "curriculum_progress": len(self.difficulty_mgr.history)
            }
        )
        
    def _get_observation(self) -> InboxState:
        dir_counts = {k: len(v) for k, v in self.internal_state["directories"].items()}
        return InboxState(
            queue=self.internal_state["queue"],
            directories=dir_counts,
            active_directive=f"[{self.difficulty_mgr.level_name}] " + TASKS[self.current_task_idx].description,
            system_clock="09:00 AM",
            last_execution_error=""
        )
