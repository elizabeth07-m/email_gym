from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict

class AlertLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class MessageItem(BaseModel):
    id: str
    source: str
    topic: str
    content: str
    alert_level: AlertLevel
    acknowledged: bool = False

class InboxState(BaseModel):
    queue: List[MessageItem] = Field(description="List of raw messages currently in the processing queue.")
    directories: Dict[str, int] = Field(
        default_factory=lambda: {"promotions": 0, "vault": 0, "operations": 0, "management": 0},
        description="Counts of messages sorted into each destination directory."
    )
    active_directive: str = Field(description="Instructions for the routing agent in the current scenario.")
    system_clock: str = Field(description="Current system time string.")
    last_execution_error: str = Field(default="", description="Any error encountered from the previous step.")

class AgentActionType(str, Enum):
    DISMISS = "dismiss"
    ROUTE_DIRECTORY = "route_directory"
    RESPOND = "respond"

class AgentAction(BaseModel):
    action_type: AgentActionType = Field(description="The action to perform: dismiss, route_directory, or respond.")
    message_id: str = Field(description="The 'id' of the target message.")
    target_directory: str = Field(default="", description="Required if action_type is route_directory. Valid options: promotions, operations, management, vault.")
    response_payload: str = Field(default="", description="Required if action_type is respond. The text response string to dispatch.")
