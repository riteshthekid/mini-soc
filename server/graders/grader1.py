"""
Grader for Task 1: Alert Triage (Easy)
Scores 0.0–1.0 based on classification accuracy and priority assignment.
Fully deterministic — same inputs always produce same score.
"""
from __future__ import annotations
from typing import Dict, Any, List


# Classification aliases — LLMs often use synonyms
_CLASSIFICATION_ALIASES = {
    "critical":   {"critical", "crit", "high", "severe", "malicious", "true_positive"},
    "suspicious": {"suspicious", "susp", "medium", "possible", "uncertain", "maybe"},
    "benign":     {"benign", "safe", "clean", "false_positive", "low", "normal", "fp"},
}


def _normalize_classification(raw: str) -> str:
    """Map an LLM classification output to the canonical form."""
    c = raw.strip().lower().replace(" ", "_")
    for canonical, aliases in _CLASSIFICATION_ALIASES.items():
        if c in aliases:
            return canonical
    return c  # return as-is if not matched


# Ground truth for each alert in Task 1
GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    "ALT-001": {"classification": "critical",   "priority": "P1"},
    "ALT-002": {"classification": "critical",   "priority": "P1"},
    "ALT-010": {"classification": "suspicious", "priority": "P2"},
    "ALT-011": {"classification": "suspicious", "priority": "P2"},
    "ALT-012": {"classification": "critical",   "priority": "P1"},
    "ALT-020": {"classification": "benign",     "priority": "P4"},
    "ALT-030": {"classification": "benign",     "priority": "P4"},
    "ALT-031": {"classification": "benign",     "priority": "P3"},
    "ALT-032": {"classification": "benign",     "priority": "P4"},
    "ALT-033": {"classification": "suspicious", "priority": "P2"},
}

TOTAL_ALERTS = len(GROUND_TRUTH)
CLASSIFICATION_WEIGHT = 0.7  # 70% of score
PRIORITY_WEIGHT = 0.3        # 30% of score


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 1 episode.

    state must contain:
      agent_classifications: Dict[alert_id, {"classification": str, "priority": str}]

    Returns float in [0.0, 1.0].
    """
    agent_classifications: Dict[str, Dict[str, str]] = state.get("agent_classifications", {})

    if not agent_classifications:
        return 0.001

    classification_correct = 0
    priority_correct = 0
    total_attempted = 0

    for alert_id, truth in GROUND_TRUTH.items():
        agent = agent_classifications.get(alert_id)
        if agent is None:
            continue

        total_attempted += 1

        # Classification score (fuzzy match via aliases)
        agent_class = _normalize_classification(agent.get("classification", ""))
        if agent_class == truth["classification"]:
            classification_correct += 1

        # Priority score (within 1 level = partial credit)
        agent_prio = agent.get("priority", "")
        truth_prio = truth["priority"]
        if agent_prio == truth_prio:
            priority_correct += 1
        elif _priority_distance(agent_prio, truth_prio) == 1:
            priority_correct += 0.5  # partial credit for off-by-one

    if total_attempted == 0:
        return 0.001

    # Coverage penalty: penalize for not attempting all alerts
    coverage = total_attempted / TOTAL_ALERTS

    classification_score = (classification_correct / TOTAL_ALERTS) * CLASSIFICATION_WEIGHT
    priority_score = (priority_correct / TOTAL_ALERTS) * PRIORITY_WEIGHT
    raw_score = (classification_score + priority_score) * coverage

    return round(min(max(raw_score, 0.001), 0.999), 4)


def _priority_distance(p1: str, p2: str) -> int:
    """Returns distance between two priority levels (P1=0, P4=3)."""
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    v1 = order.get(p1, -1)
    v2 = order.get(p2, -1)
    if v1 == -1 or v2 == -1:
        return 99
    return abs(v1 - v2)


def compute_step_reward(
    alert_id: str,
    classification: str,
    priority: str,
) -> float:
    """
    Per-step reward for a single classification action.
    Called by env.step() to provide dense reward signal.
    """
    truth = GROUND_TRUTH.get(alert_id)
    if not truth:
        return -0.05  # classifying unknown alert

    reward = 0.0
    norm_class = _normalize_classification(classification)
    if norm_class == truth["classification"]:
        reward += 0.2
    else:
        # Penalize severity mismatch proportional to danger
        if truth["classification"] == "critical" and norm_class == "benign":
            reward -= 0.3  # worst: missing a critical as benign
        elif truth["classification"] == "benign" and norm_class == "critical":
            reward -= 0.1  # false alarm: wasted resources
        else:
            reward -= 0.05  # minor miss

    dist = _priority_distance(priority, truth["priority"])
    if dist == 0:
        reward += 0.1
    elif dist == 1:
        reward += 0.05

    return round(reward, 4)
