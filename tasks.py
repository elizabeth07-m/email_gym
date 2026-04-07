from typing import Dict, Any, List

class SemanticScorer:
    """
    Simulates a sophisticated LLM Judge for evaluating response quality.
    In a true production environment, this makes an API call. Here we simulate
    semantic and tone checking to save costs and run offline while maintaining
    the exact same architectural pattern.
    """
    @classmethod
    def evaluate_response_integrity(cls, payload: str, expected_concepts: List[str], negative_concepts: List[str] = None) -> float:
        score = 0.0
        text_lower = payload.lower()
        
        concept_hits = 0
        for concept in expected_concepts:
            if concept.lower() in text_lower:
                concept_hits += 1
                
        if len(expected_concepts) > 0:
            score += (concept_hits / len(expected_concepts)) * 0.8
            
        polite_words = ["please", "thank", "thanks", "appreciate", "regards", "best", "hello", "hi"]
        if any(w in text_lower for w in polite_words):
            score += 0.2
            
        if negative_concepts:
            for neg in negative_concepts:
                if neg.lower() in text_lower:
                    score -= 0.5
                    
        return max(0.0, min(1.0, score))


class RoutingTask:
    def __init__(self, description: str, difficulty: str, level_tier: int = 1):
        self.description = description
        self.difficulty = difficulty
        self.level_tier = level_tier

    def setup_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def grade(self, state: Dict[str, Any]) -> float:
        raise NotImplementedError


class Task1_Easy(RoutingTask):
    def __init__(self):
        super().__init__(
            description="Warmup: Clear your queue. Route all promotional broadcasts to the 'promotions' directory and legitimate operations mail to 'operations' or 'vault'.",
            difficulty="easy",
            level_tier=1
        )

    def setup_state(self) -> Dict[str, Any]:
        from models import MessageItem, AlertLevel
        return {
            "queue": [
                MessageItem(id="1", source="internal-sys@ops.net", topic="Build complete", content="Artifact deployed successfully.", alert_level=AlertLevel.NORMAL),
                MessageItem(id="2", source="marketing@vendor.com", topic="Exclusive Offer", content="Claim your discount.", alert_level=AlertLevel.LOW),
                MessageItem(id="3", source="cto@hq.net", topic="Architecture Review", content="Please review the new design docs.", alert_level=AlertLevel.HIGH),
                MessageItem(id="4", source="sales@tooling.io", topic="Upgrade now", content="Your trial is expiring.", alert_level=AlertLevel.LOW),
            ],
            "directories": {"promotions": [], "vault": [], "operations": [], "management": []},
            "dispatched_responses": [],
            "purged_messages": []
        }

    def grade(self, state: Dict[str, Any]) -> float:
        score = 0.0
        promo_ids = [m.id for m in state["directories"]["promotions"]]
        ops_or_vault = [m.id for m in state["directories"]["operations"]] + [m.id for m in state["directories"]["vault"]] + [m.id for m in state["directories"]["management"]]
        
        if "2" in promo_ids: score += 0.25
        if "4" in promo_ids: score += 0.25
        
        if "1" in ops_or_vault: score += 0.25
        if "3" in ops_or_vault: score += 0.25
        
        if "1" in promo_ids or "3" in promo_ids:
            score -= 0.5
            
        return max(0.0, min(1.0, score))


class Task2_Medium(RoutingTask):
    def __init__(self):
        super().__init__(
            description="Intermediate: Respond politely to the Management stakeholder acknowledging the system update. Route automated broadcasts to the vault.",
            difficulty="medium",
            level_tier=2
        )

    def setup_state(self) -> Dict[str, Any]:
        from models import MessageItem, AlertLevel
        return {
            "queue": [
                MessageItem(id="1", source="metrics@monitoring.local", topic="Daily Node Report", content="Cluster usage at 45%.", alert_level=AlertLevel.LOW),
                MessageItem(id="2", source="vp_engineering@hq.net", topic="Critical Path Tracker", content="Acknowledge you have seen the updated timeline.", alert_level=AlertLevel.HIGH),
            ],
            "directories": {"promotions": [], "vault": [], "operations": [], "management": []},
            "dispatched_responses": [],
            "purged_messages": []
        }

    def grade(self, state: Dict[str, Any]) -> float:
        score = 0.0
        
        vp_response = next((r for r in state["dispatched_responses"] if r["message_id"] == "2"), None)
        if vp_response:
            judge_score = SemanticScorer.evaluate_response_integrity(
                payload=vp_response["payload"],
                expected_concepts=["received", "got it", "acknowledged", "seen"],
                negative_concepts=["no", "busy", "cannot"]
            )
            score += judge_score * 0.7
            
        vaulted_ids = [m.id for m in state["directories"]["vault"]]
        if "1" in vaulted_ids:
            score += 0.3
        
        return max(0.0, min(1.0, score))


class Task3_Hard(RoutingTask):
    def __init__(self):
        super().__init__(
            description="Advanced (Adversarial): Respond to DevOps asserting '15:00' due to a deployment conflict. Route alerts to 'operations' and dismiss marketing to 'promotions'.",
            difficulty="hard",
            level_tier=3
        )

    def setup_state(self) -> Dict[str, Any]:
        from models import MessageItem, AlertLevel
        return {
            "queue": [
                MessageItem(id="1", source="devops@ops.net", topic="Deployment Window", content="Can we push the release at 14:00 today?", alert_level=AlertLevel.NORMAL),
                MessageItem(id="2", source="system@cron.local", topic="Database Maintenance", content="Database locked for maintenance from 13:00 to 14:30.", alert_level=AlertLevel.CRITICAL),
                MessageItem(id="3", source="partner@vendor.com", topic="New Integration", content="Webinar on latest features at 12:00.", alert_level=AlertLevel.HIGH),
            ],
            "directories": {"promotions": [], "vault": [], "operations": [], "management": []},
            "dispatched_responses": [],
            "purged_messages": []
        }

    def grade(self, state: Dict[str, Any]) -> float:
        score = 0.0
        
        devops_response = next((r for r in state["dispatched_responses"] if r["message_id"] == "1"), None)
        if devops_response:
            judge_score = SemanticScorer.evaluate_response_integrity(
                payload=devops_response["payload"],
                expected_concepts=["15:00", "3 pm", "1500"],
                negative_concepts=["14:00", "2 pm", "1400"]
            )
            score += (judge_score * 0.5)
            
        ops_ids = [m.id for m in state["directories"]["operations"]]
        if "2" in ops_ids:
            score += 0.25
            
        promo_ids = [m.id for m in state["directories"]["promotions"]]
        if "3" in promo_ids:
            score += 0.25
            
        return max(0.0, min(1.0, score))

TASKS = [Task1_Easy(), Task2_Medium(), Task3_Hard()]
