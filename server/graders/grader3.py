"""
Grader for Task 3: Active Threat Response (Hard)
Scores containment effectiveness, collateral damage, speed, and report quality.
This is designed to genuinely challenge frontier models.

Supports dynamic scenario selection — ground truth is derived from the active
scenario so Tier 2/3 scenarios (ransomware, APT) are graded correctly.
"""
from __future__ import annotations
from typing import Dict, Any, Set

# Dual-import for OpenEnv Docker compatibility
try:
    from server.simulator.attack_seeds import ATTACK_SCENARIOS, ASSET_INVENTORY
except ImportError:
    from ..simulator.attack_seeds import ATTACK_SCENARIOS, ASSET_INVENTORY


# Default ground truth (Tier 1 fallback)
_DEFAULT_SCENARIO = "phishing_lateral_001"

CONTAINMENT_WEIGHT = 0.30
COLLATERAL_WEIGHT = 0.20   # negative: penalize unnecessary isolations
EVIDENCE_WEIGHT = 0.20
SPEED_WEIGHT = 0.10
REPORT_WEIGHT = 0.20

# Report must always contain these fields
REPORT_REQUIRED_FIELDS = {"summary", "attack_type", "affected_assets", "attacker_ip", "timeline"}


def _get_ground_truth(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build grader ground truth dynamically from the active scenario.
    Falls back to phishing_lateral_001 for backward compatibility.
    """
    scenario_id = state.get("scenario_id", _DEFAULT_SCENARIO) or _DEFAULT_SCENARIO
    scenario = ATTACK_SCENARIOS.get(scenario_id)
    if not scenario:
        scenario = ATTACK_SCENARIOS[_DEFAULT_SCENARIO]
        scenario_id = _DEFAULT_SCENARIO

    gt = scenario.get("ground_truth", {})
    # Prefer explicit assets_to_isolate; fall back to affected_assets
    if "assets_to_isolate" in gt:
        to_isolate = set(gt["assets_to_isolate"])
    else:
        to_isolate = set(gt.get("affected_assets", []))
    attacker_ips = set(gt.get("attacker_ips", []))

    # Build assets_to_not_isolate: critical assets (≥3) NOT compromised
    do_not_isolate: set = set()
    criticality_map: Dict[str, float] = {}
    for asset in ASSET_INVENTORY:
        hostname = asset["hostname"]
        crit = asset["criticality"]
        if hostname not in to_isolate and crit >= 3:
            do_not_isolate.add(hostname)
            criticality_map[hostname] = min(crit / 5.0, 1.0)

    # Build key evidence sources
    if "key_evidence_sources" in gt:
        key_sources: set = set(gt["key_evidence_sources"])
    else:
        key_sources = set()
        for source, logs in scenario.get("logs", {}).items():
            if any(log.get("is_malicious", False) for log in logs):
                key_sources.add(source)

    return {
        "scenario_id": scenario_id,
        "attack_type": gt.get("attack_type", "unknown"),
        "attacker_ip": scenario.get("attacker_ip", "") or "",
        "assets_to_isolate": to_isolate,
        "assets_to_not_isolate": do_not_isolate,
        "criticality_map": criticality_map,
        "ips_to_block": attacker_ips,
        "key_evidence_sources": key_sources,
        "kill_chain": scenario.get("kill_chain", []),
        "report_required_fields": REPORT_REQUIRED_FIELDS,
    }


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 3 episode.

    state must contain:
      scenario_id: str
      agent_isolated_assets: List[str]
      agent_blocked_ips: List[str]
      agent_queried_sources: List[str]
      agent_report: Dict[str, Any]        — written report fields
      steps_taken: int
      max_steps: int
    """
    gt = _get_ground_truth(state)

    containment_score = _score_containment(state, gt)
    collateral_penalty = _score_collateral(state, gt)
    evidence_score = _score_evidence(state, gt)
    speed_score = _score_speed(state)
    report_score = _score_report(state, gt)

    total = (
        containment_score * CONTAINMENT_WEIGHT
        - collateral_penalty * COLLATERAL_WEIGHT
        + evidence_score * EVIDENCE_WEIGHT
        + speed_score * SPEED_WEIGHT
        + report_score * REPORT_WEIGHT
    )

    return round(min(max(total, 0.001), 0.999), 4)


def _score_containment(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Did the agent isolate the right assets and block the right IPs?"""
    isolated: Set[str] = set(state.get("agent_isolated_assets", []))
    blocked: Set[str] = set(state.get("agent_blocked_ips", []))

    must_isolate = gt["assets_to_isolate"]
    must_block = gt["ips_to_block"]

    isolation_score = len(isolated & must_isolate) / len(must_isolate) if must_isolate else 1.0
    block_score = len(blocked & must_block) / len(must_block) if must_block else 1.0

    return (isolation_score * 0.6 + block_score * 0.4)


def _score_collateral(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Penalty for isolating critical assets that weren't compromised."""
    isolated: Set[str] = set(state.get("agent_isolated_assets", []))
    should_not_isolate = gt["assets_to_not_isolate"]
    criticality_map = gt["criticality_map"]

    collateral = isolated & should_not_isolate
    if not collateral:
        return 0.0

    penalty = sum(criticality_map.get(h, 0.3) for h in collateral)
    return min(penalty, 1.0)


def _score_evidence(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Did the agent query the right log sources to find the kill chain?"""
    queried: Set[str] = set(state.get("agent_queried_sources", []))
    key_sources = gt["key_evidence_sources"]
    if not key_sources:
        return 1.0
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


def _score_report(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Score the incident report quality based on required fields."""
    report: Dict[str, Any] = state.get("agent_report", {})
    if not report:
        return 0.0

    required = gt["report_required_fields"]
    truth_ip = gt["attacker_ip"]

    # Count filled fields — special handling for attacker_ip on insider threats
    found = 0
    for field in required:
        if field == "attacker_ip" and not truth_ip:
            # No external attacker exists (insider threat).
            # Credit if agent omits IP, leaves blank, or identifies as internal.
            agent_ip = (report.get("attacker_ip") or "").strip().lower()
            if not agent_ip or "internal" in agent_ip or "insider" in agent_ip or agent_ip == "n/a":
                found += 1  # correct: recognized no external attacker
            else:
                found += 0.5  # partial: filled something, but scenario has no external IP
        elif report.get(field):
            found += 1
    field_score = found / len(required)

    # Bonus: correct attack_type in report
    agent_type = report.get("attack_type", "").lower().replace(" ", "_")
    truth_type = gt["attack_type"]
    if agent_type == truth_type:
        field_score = min(field_score + 0.15, 1.0)
    elif agent_type and (truth_type in agent_type or agent_type in truth_type):
        field_score = min(field_score + 0.10, 1.0)
    else:
        agent_words = set(agent_type.split("_"))
        truth_words = set(truth_type.split("_"))
        overlap = agent_words & truth_words
        if overlap and len(overlap) >= len(truth_words) * 0.5:
            field_score = min(field_score + 0.05, 1.0)

    # Bonus: correct attacker IP in report
    if truth_ip:
        # External attacker scenario — partial credit for subnet
        agent_ip = report.get("attacker_ip", "").strip()
        if agent_ip == truth_ip:
            field_score = min(field_score + 0.1, 1.0)
        else:
            agent_parts = agent_ip.split(".")
            truth_parts = truth_ip.split(".")
            if len(agent_parts) == 4 and len(truth_parts) == 4:
                if agent_parts[:3] == truth_parts[:3]:
                    field_score = min(field_score + 0.06, 1.0)
                elif agent_parts[:2] == truth_parts[:2]:
                    field_score = min(field_score + 0.03, 1.0)
    else:
        # Insider threat — award bonus for correctly identifying as internal
        agent_ip = (report.get("attacker_ip") or "").strip().lower()
        if not agent_ip or "internal" in agent_ip or "insider" in agent_ip or agent_ip == "n/a":
            field_score = min(field_score + 0.1, 1.0)

    return round(field_score, 4)


def compute_step_reward(action_type: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> float:
    """Dense per-step reward for Task 3."""
    gt = _get_ground_truth(state)
    reward = 0.0

    if action_type == "query_logs":
        source = parameters.get("log_source", "")
        previously_rewarded_sources = set(state.get("rewarded_sources", []))
        if source not in previously_rewarded_sources:
            if source in gt["key_evidence_sources"]:
                reward += 0.10
            else:
                reward -= 0.02
            state.setdefault("rewarded_sources", []).append(source)

    elif action_type == "isolate_asset":
        hostname = parameters.get("hostname", "")
        if hostname in gt["assets_to_isolate"]:
            reward += 0.25  # correct isolation
        elif hostname in gt["assets_to_not_isolate"]:
            # Collateral damage penalty scaled by criticality
            crit_penalty = gt["criticality_map"].get(hostname, 0.2)
            reward -= crit_penalty  # e.g. DC-01 = -0.40 if criticality 5

    elif action_type == "block_ip":
        ip = parameters.get("ip_address", "")
        if ip in gt["ips_to_block"]:
            reward += 0.20
        else:
            # Partial credit for blocking IP in same subnet as attacker
            partial = False
            for target_ip in gt["ips_to_block"]:
                ip_parts = ip.split(".")
                target_parts = target_ip.split(".")
                if len(ip_parts) == 4 and len(target_parts) == 4:
                    if ip_parts[:3] == target_parts[:3]:
                        reward += 0.10  # Same /24 subnet
                        partial = True
                        break
                    elif ip_parts[:2] == target_parts[:2]:
                        reward += 0.05  # Same /16 subnet
                        partial = True
                        break
            if not partial:
                reward -= 0.05  # blocking wrong IP

    elif action_type == "write_report":
        report = parameters.get("report", {})
        required = gt["report_required_fields"]
        filled = sum(1 for f in required if report.get(f))
        reward += (filled / len(required)) * 0.3

    elif action_type == "close_incident":
        # Reward only if proper containment already done
        isolated = set(state.get("agent_isolated_assets", []))
        if isolated & gt["assets_to_isolate"]:
            reward += 0.15

    return round(reward, 4)
