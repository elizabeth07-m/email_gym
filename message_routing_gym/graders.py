"""
Graders for Message Routing Gym.

Two complementary grading layers (mirrors hallucination-detector-gym):

1. ProgrammaticGrader  — exact-match checks: did the right messages land in
                          the right directories?

2. SemanticGrader      — LLM-judge-style keyword/tone analysis: does the
                          response text meet quality standards?

3. CompositeGrader     — combines both, returns a single 0.0–1.0 score.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional


# ---------------------------------------------------------------------------
# Programmatic grader
# ---------------------------------------------------------------------------

class ProgrammaticGrader:
    """
    Exact-match routing verifier.
    Each task supplies a `grading_spec` dict of the form:
        {
          "required_route": {message_id: directory_name, ...},
          "required_response_ids": [message_id, ...],
          "forbidden_routes": {message_id: directory_name, ...},
        }
    Returns 0.0–1.0.
    """

    @staticmethod
    def grade(state: Dict[str, Any], spec: Dict[str, Any]) -> float:
        score = 0.0
        total_checks = 0

        required_routes: Dict[str, str] = spec.get("required_route", {})
        required_response_ids: List[str] = spec.get("required_response_ids", [])
        forbidden_routes: Dict[str, str] = spec.get("forbidden_routes", {})

        # ── Required routing checks ───────────────────────────────────
        for msg_id, expected_dir in required_routes.items():
            total_checks += 1
            messages_in_dir = [m.id for m in state["directories"].get(expected_dir, [])]
            if msg_id in messages_in_dir:
                score += 1.0

        # ── Required response checks ──────────────────────────────────
        dispatched_ids = {r["message_id"] for r in state.get("dispatched_responses", [])}
        for msg_id in required_response_ids:
            total_checks += 1
            if msg_id in dispatched_ids:
                score += 1.0

        # ── Forbidden route penalties ─────────────────────────────────
        for msg_id, bad_dir in forbidden_routes.items():
            messages_in_bad_dir = [m.id for m in state["directories"].get(bad_dir, [])]
            if msg_id in messages_in_bad_dir:
                score -= 0.5  # Applies to the raw score before normalisation

        if total_checks == 0:
            return 0.0

        return max(0.0, min(1.0, score / total_checks))


# ---------------------------------------------------------------------------
# Semantic grader
# ---------------------------------------------------------------------------

class SemanticGrader:
    """
    Simulates an LLM judge evaluating response text quality.

    In production this would call an LLM API with a structured prompt.
    Here we use keyword matching plus polite-tone scoring to provide a clean
    reward signal without API costs during local training.

    Scoring:
      - Concept coverage: up to 0.80 (fraction of required concepts present)
      - Polite tone:      up to 0.20
      - Negative concept: -0.50 per hit (clamped)
    """

    POLITE_WORDS = frozenset([
        "please", "thank", "thanks", "appreciate", "regards",
        "best", "hello", "hi", "understood", "certainly", "of course",
    ])

    @classmethod
    def grade(
        cls,
        *,
        payload: str,
        expected_concepts: List[str],
        negative_concepts: Optional[List[str]] = None,
    ) -> float:
        if not payload:
            return 0.0

        text = payload.lower()
        score = 0.0

        # Concept coverage
        if expected_concepts:
            hits = sum(1 for c in expected_concepts if c.lower() in text)
            score += (hits / len(expected_concepts)) * 0.80

        # Polite tone
        if any(w in text for w in cls.POLITE_WORDS):
            score += 0.20

        # Negative concept penalty
        if negative_concepts:
            for neg in negative_concepts:
                if neg.lower() in text:
                    score -= 0.50

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

class CompositeGrader:
    """
    Combines ProgrammaticGrader and SemanticGrader into a single 0.0–1.0 score.

    Each task specifies `grading_spec` and `response_checks`, plus optional
    weights that sum to 1.0.
    """

    def __init__(
        self,
        *,
        programmatic_weight: float = 0.60,
        semantic_weight: float = 0.40,
    ):
        assert abs(programmatic_weight + semantic_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.prog_w = programmatic_weight
        self.sem_w = semantic_weight

    def grade(
        self,
        state: Dict[str, Any],
        *,
        grading_spec: Dict[str, Any],
        response_checks: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Compute the final composite grade.

        Parameters
        ----------
        state:          Full environment internal state dict
        grading_spec:   Spec for ProgrammaticGrader (required_route etc.)
        response_checks: List of {message_id, expected_concepts, negative_concepts}
                         for SemanticGrader.  If empty, semantic weight is folded
                         into programmatic.
        """
        prog_score = ProgrammaticGrader.grade(state, grading_spec)

        if not response_checks:
            # No response to evaluate — full weight on programmatic
            return prog_score

        sem_scores = []
        dispatched: List[Dict] = state.get("dispatched_responses", [])

        for check in response_checks:
            msg_id = check["message_id"]
            response = next((r for r in dispatched if r["message_id"] == msg_id), None)
            if response is None:
                sem_scores.append(0.0)
            else:
                sem_scores.append(
                    SemanticGrader.grade(
                        payload=response["payload"],
                        expected_concepts=check.get("expected_concepts", []),
                        negative_concepts=check.get("negative_concepts"),
                    )
                )

        avg_sem = sum(sem_scores) / len(sem_scores)
        return max(0.0, min(1.0, prog_score * self.prog_w + avg_sem * self.sem_w))
