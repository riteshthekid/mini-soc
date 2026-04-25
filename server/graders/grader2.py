"""
Grader for Task 2: Incident Investigation (Medium)
Scores based on evidence gathering quality, timeline correlation, and verdict accuracy.
"""
from __future__ import annotations
from typing import Dict, Any, List, Set


GROUND_TRUTH = {
    "scenario_id": "brute_force_ssh_001",
    "verdict": "true_positive",
    "attack_type": "brute_force",
    "attacker_ip": "185.220.101.47",
    "target_hostname": "WEB-SERVER-01",
    "key_evidence_log_ids": {
        "AUTH-001",   # failed logins
        "AUTH-002",   # successful login after failures
        "FW-001",     # external SSH allowed
    },
    "key_log_sources": {"auth", "firewall"},
}

# Scoring weights
VERDICT_WEIGHT = 0.35
ATTACK_TYPE_WEIGHT = 0.20
EVIDENCE_WEIGHT = 0.30
ATTACKER_ID_WEIGHT = 0.15


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 2 episode.

    state must contain:
      agent_verdict: str                    — "true_positive" | "false_positive"
      agent_attack_type: str                — e.g. "brute_force"
      agent_attacker_ip: str                — identified attacker IP
      agent_queried_log_ids: List[str]      — log IDs the agent retrieved
      agent_queried_sources: List[str]      — log sources queried
    """
    verdict_score = _score_verdict(state)
    attack_type_score = _score_attack_type(state)
    evidence_score = _score_evidence(state)
    attacker_score = _score_attacker_id(state)

    total = (
        verdict_score * VERDICT_WEIGHT
        + attack_type_score * ATTACK_TYPE_WEIGHT
        + evidence_score * EVIDENCE_WEIGHT
        + attacker_score * ATTACKER_ID_WEIGHT
    )

    return round(min(max(total, 0.001), 0.999), 4)


def _score_verdict(state: Dict[str, Any]) -> float:
    agent = state.get("agent_verdict", "").lower()
    truth = GROUND_TRUTH["verdict"]
    return 1.0 if agent == truth else 0.0


def _score_attack_type(state: Dict[str, Any]) -> float:
    agent = state.get("agent_attack_type", "").lower().replace(" ", "_")
    truth = GROUND_TRUTH["attack_type"]
    return 1.0 if agent == truth else 0.0


def _score_evidence(state: Dict[str, Any]) -> float:
    """
    Partial credit for each key piece of evidence retrieved.
    Also rewards querying the right log sources.
    """
    queried_ids: Set[str] = set(state.get("agent_queried_log_ids", []))
    queried_sources: Set[str] = set(state.get("agent_queried_sources", []))

    key_ids = GROUND_TRUTH["key_evidence_log_ids"]
    key_sources = GROUND_TRUTH["key_log_sources"]

    id_score = len(queried_ids & key_ids) / len(key_ids) if key_ids else 0.0
    source_score = len(queried_sources & key_sources) / len(key_sources) if key_sources else 0.0

    # Penalize for querying too many irrelevant sources (thrashing)
    irrelevant = len(queried_sources - key_sources)
    noise_penalty = min(irrelevant * 0.05, 0.2)

    return max((id_score * 0.6 + source_score * 0.4) - noise_penalty, 0.0)


def _score_attacker_id(state: Dict[str, Any]) -> float:
    agent_ip = state.get("agent_attacker_ip", "").strip()
    truth_ip = GROUND_TRUTH["attacker_ip"]
    return 1.0 if agent_ip == truth_ip else 0.0


def compute_step_reward(action_type: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> float:
    """
    Dense per-step reward for Task 2.
    """
    reward = 0.0

    if action_type == "query_logs":
        source = parameters.get("log_source", "")
        if source in GROUND_TRUTH["key_log_sources"]:
            reward += 0.15  # relevant log source
        else:
            reward -= 0.03  # irrelevant query costs time

        # Bonus if key evidence log ID found
        queried_ids = set(state.get("agent_queried_log_ids", []))
        new_key_evidence = queried_ids & GROUND_TRUTH["key_evidence_log_ids"]
        reward += len(new_key_evidence) * 0.1

    elif action_type == "classify_alert":
        classification = parameters.get("classification", "")
        if classification == "critical":
            reward += 0.1  # correct initial classification

    elif action_type == "close_incident":
        verdict = parameters.get("verdict", "")
        attack_type = parameters.get("attack_type", "").replace(" ", "_")
        if verdict == GROUND_TRUTH["verdict"]:
            reward += 0.3
        if attack_type == GROUND_TRUTH["attack_type"]:
            reward += 0.2

    elif action_type == "escalate":
        reward += 0.05  # reasonable action for high-severity

    return round(reward, 4)
