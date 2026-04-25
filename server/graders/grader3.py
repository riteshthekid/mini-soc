"""
Grader for Task 3: Active Threat Response (Hard)
Scores containment effectiveness, collateral damage, speed, and report quality.
This is designed to genuinely challenge frontier models.
"""
from __future__ import annotations
from typing import Dict, Any, List, Set


GROUND_TRUTH = {
    "scenario_id": "phishing_lateral_001",
    "attack_type": "lateral_movement",
    "attacker_ip": "94.102.49.190",
    "assets_to_isolate": {"WS-HR-03"},          # must isolate (compromised)
    "assets_to_not_isolate": {"DC-01", "DB-FINANCE-01", "WEB-SERVER-01"},  # collateral damage if isolated
    "ips_to_block": {"94.102.49.190"},
    "key_evidence_sources": {"process", "network", "auth", "dns"},
    "kill_chain": ["phishing", "credential_theft", "lateral_movement"],
    "report_required_fields": {
        "summary", "attack_type", "affected_assets", "attacker_ip", "timeline"
    },
}

CONTAINMENT_WEIGHT = 0.30
COLLATERAL_WEIGHT = 0.20   # negative: penalize unnecessary isolations
EVIDENCE_WEIGHT = 0.20
SPEED_WEIGHT = 0.10
REPORT_WEIGHT = 0.20


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 3 episode.

    state must contain:
      agent_isolated_assets: List[str]
      agent_blocked_ips: List[str]
      agent_queried_sources: List[str]
      agent_report: Dict[str, Any]        — written report fields
      steps_taken: int
      max_steps: int
    """
    containment_score = _score_containment(state)
    collateral_penalty = _score_collateral(state)
    evidence_score = _score_evidence(state)
    speed_score = _score_speed(state)
    report_score = _score_report(state)

    total = (
        containment_score * CONTAINMENT_WEIGHT
        - collateral_penalty * COLLATERAL_WEIGHT
        + evidence_score * EVIDENCE_WEIGHT
        + speed_score * SPEED_WEIGHT
        + report_score * REPORT_WEIGHT
    )

    return round(min(max(total, 0.001), 0.999), 4)


def _score_containment(state: Dict[str, Any]) -> float:
    """Did the agent isolate the right assets and block the right IPs?"""
    isolated: Set[str] = set(state.get("agent_isolated_assets", []))
    blocked: Set[str] = set(state.get("agent_blocked_ips", []))

    must_isolate = GROUND_TRUTH["assets_to_isolate"]
    must_block = GROUND_TRUTH["ips_to_block"]

    isolation_score = len(isolated & must_isolate) / len(must_isolate) if must_isolate else 0.0
    block_score = len(blocked & must_block) / len(must_block) if must_block else 0.0

    return (isolation_score * 0.6 + block_score * 0.4)


def _score_collateral(state: Dict[str, Any]) -> float:
    """Penalty for isolating critical assets that weren't compromised."""
    isolated: Set[str] = set(state.get("agent_isolated_assets", []))
    should_not_isolate = GROUND_TRUTH["assets_to_not_isolate"]

    collateral = isolated & should_not_isolate
    if not collateral:
        return 0.0

    # Weight collateral by criticality
    criticality_map = {
        "DC-01": 1.0,          # domain controller — catastrophic
        "DB-FINANCE-01": 0.8,  # finance DB — severe
        "WEB-SERVER-01": 0.5,  # web server — bad but recoverable
    }
    penalty = sum(criticality_map.get(h, 0.3) for h in collateral)
    return min(penalty, 1.0)


def _score_evidence(state: Dict[str, Any]) -> float:
    """Did the agent query the right log sources to find the kill chain?"""
    queried: Set[str] = set(state.get("agent_queried_sources", []))
    key_sources = GROUND_TRUTH["key_evidence_sources"]
    return len(queried & key_sources) / len(key_sources)


def _score_speed(state: Dict[str, Any]) -> float:
    """Faster containment = higher speed score. Linear decay."""
    steps = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 30)
    if max_steps == 0:
        return 0.0
    # Full speed score if contained by 50% of budget, zero at 100%
    ratio = steps / max_steps
    speed = max(1.0 - (ratio * 2), 0.0)
    return round(speed, 4)


def _score_report(state: Dict[str, Any]) -> float:
    """Score the incident report quality based on required fields."""
    report: Dict[str, Any] = state.get("agent_report", {})
    if not report:
        return 0.0

    required = GROUND_TRUTH["report_required_fields"]
    found = sum(1 for field in required if report.get(field))
    field_score = found / len(required)

    # Bonus: correct attack_type in report
    if report.get("attack_type", "").lower().replace(" ", "_") == GROUND_TRUTH["attack_type"]:
        field_score = min(field_score + 0.15, 1.0)

    # Bonus: correct attacker IP in report
    if report.get("attacker_ip", "") == GROUND_TRUTH["attacker_ip"]:
        field_score = min(field_score + 0.1, 1.0)

    return round(field_score, 4)


def compute_step_reward(action_type: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> float:
    """Dense per-step reward for Task 3."""
    reward = 0.0

    if action_type == "query_logs":
        source = parameters.get("log_source", "")
        if source in GROUND_TRUTH["key_evidence_sources"]:
            reward += 0.10
        else:
            reward -= 0.02

    elif action_type == "isolate_asset":
        hostname = parameters.get("hostname", "")
        if hostname in GROUND_TRUTH["assets_to_isolate"]:
            reward += 0.25  # correct isolation
        elif hostname in GROUND_TRUTH["assets_to_not_isolate"]:
            crit_penalty = {"DC-01": -0.4, "DB-FINANCE-01": -0.3, "WEB-SERVER-01": -0.15}
            reward += crit_penalty.get(hostname, -0.2)  # collateral damage

    elif action_type == "block_ip":
        ip = parameters.get("ip_address", "")
        if ip in GROUND_TRUTH["ips_to_block"]:
            reward += 0.20
        else:
            reward -= 0.05  # blocking wrong IP

    elif action_type == "write_report":
        report = parameters.get("report", {})
        required = GROUND_TRUTH["report_required_fields"]
        filled = sum(1 for f in required if report.get(f))
        reward += (filled / len(required)) * 0.3

    elif action_type == "close_incident":
        # Reward only if proper containment already done
        isolated = set(state.get("agent_isolated_assets", []))
        if isolated & GROUND_TRUTH["assets_to_isolate"]:
            reward += 0.15

    return round(reward, 4)
