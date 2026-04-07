"""
Constants, enumerations, and static lookup tables for Message Routing Gym.
"""

from enum import Enum


class AlertLevel(str, Enum):
    """Priority level of an incoming message."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """All valid agent actions against the routing environment."""
    DISMISS = "dismiss"           # Route to vault and remove from queue
    ROUTE_DIRECTORY = "route_directory"  # Move to a named directory
    RESPOND = "respond"           # Dispatch a text reply to the sender


class DirectoryName(str, Enum):
    """Valid destination directories for routed messages."""
    PROMOTIONS = "promotions"
    VAULT = "vault"
    OPERATIONS = "operations"
    MANAGEMENT = "management"


class DifficultyTier(str, Enum):
    """Curriculum difficulty tiers, escalated by the DifficultyManager."""
    WARMUP = "warmup"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# Alert level to numeric priority (for vectorized observation)
ALERT_PRIORITY: dict[str, int] = {
    AlertLevel.LOW: 0,
    AlertLevel.NORMAL: 1,
    AlertLevel.HIGH: 2,
    AlertLevel.CRITICAL: 3,
}

# All valid directories in order (for numeric indexing in Gymnasium wrapper)
DIRECTORY_ORDER: list[str] = [
    DirectoryName.PROMOTIONS,
    DirectoryName.VAULT,
    DirectoryName.OPERATIONS,
    DirectoryName.MANAGEMENT,
]

# Accent colours used by the Gradio UI for each directory
DIRECTORY_COLORS: dict[str, str] = {
    DirectoryName.PROMOTIONS: "#f59e0b",
    DirectoryName.VAULT:      "#38bdf8",
    DirectoryName.OPERATIONS: "#10b981",
    DirectoryName.MANAGEMENT: "#a855f7",
}

# Alert level badge colours
ALERT_COLORS: dict[str, str] = {
    AlertLevel.CRITICAL: "#ef4444",
    AlertLevel.HIGH:     "#f59e0b",
    AlertLevel.NORMAL:   "#3b82f6",
    AlertLevel.LOW:      "#64748b",
}

# Tier display names
TIER_LABELS: dict[int, str] = {
    1: DifficultyTier.WARMUP,
    2: DifficultyTier.INTERMEDIATE,
    3: DifficultyTier.ADVANCED,
}

# Reward shaping constants
REWARD_ROUTE_STEP: float = 0.05       # Base reward for any valid routing step
REWARD_RESPOND_STEP: float = 0.10     # Base reward for a dispatched response
REWARD_GRADE_WEIGHT: float = 0.50     # Weight applied to per-step grade score
REWARD_RESOLUTION_BONUS: float = 1.5  # Bonus for fully resolving the episode
REWARD_INVALID_ACTION: float = -0.20  # Penalty for hallucinated message IDs
REWARD_BAD_DIRECTORY: float = -0.10   # Penalty for routing to nonexistent dir
REWARD_REPEAT_PENALTY: float = -0.15  # Penalty per repeated action
REWARD_TIMEOUT_FLOOR: float = -2.0    # Net reward floor for timed-out episodes
MAX_STEPS: int = 12                    # Max steps before timeout
