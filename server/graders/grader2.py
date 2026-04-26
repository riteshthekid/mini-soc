"""
Grader for Task 2: Incident Investigation (Medium)
Scores based on evidence gathering quality, timeline correlation, and verdict accuracy.

Supports dynamic scenario selection — ground truth is looked up from the
active scenario rather than hardcoded, so Tier 2/3 scenarios work automatically.
"""
from __future__ import annotations
from typing import Dict, Any, Set

# Dual-import for OpenEnv Docker compatibility
try:
    from server.simulator.attack_seeds import ATTACK_SCENARIOS
except ImportError:
    from ..simulator.attack_seeds import ATTACK_SCENARIOS


# Default ground truth (Tier 1 fallback)
_DEFAULT_SCENARIO = "brute_force_ssh_001"

# Scoring weights
VERDICT_WEIGHT = 0.35
ATTACK_TYPE_WEIGHT = 0.20
EVIDENCE_WEIGHT = 0.30
ATTACKER_ID_WEIGHT = 0.15


def _get_ground_truth(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build grader ground truth dynamically from the active scenario.
    Falls back to brute_force_ssh_001 for backward compatibility.
    """
    scenario_id = state.get("scenario_id", _DEFAULT_SCENARIO) or _DEFAULT_SCENARIO
    scenario = ATTACK_SCENARIOS.get(scenario_id)
    if not scenario:
        scenario = ATTACK_SCENARIOS[_DEFAULT_SCENARIO]

    gt = scenario.get("ground_truth", {})

    # Derive key evidence log IDs from malicious log entries
    key_evidence: set = set()
    for source, logs in scenario.get("logs", {}).items():
        for log in logs:
            if log.get("is_malicious", False) and "log_id" in log:
                key_evidence.add(log["log_id"])

    # Build key log sources — prefer explicit field, else derive from malicious logs
    if "key_evidence_sources" in gt:
        key_sources: Set[str] = set(gt["key_evidence_sources"])
    else:
        key_sources = set()
        for source, logs in scenario.get("logs", {}).items():
            if any(log.get("is_malicious", False) for log in logs):
                key_sources.add(source)

    return {
        "scenario_id": scenario_id,
        "verdict": gt.get("verdict", "true_positive"),
        "attack_type": gt.get("attack_type", "unknown"),
        "attacker_ip": scenario.get("attacker_ip", "") or "",
        "key_evidence_log_ids": key_evidence,
        "key_log_sources": key_sources,
    }


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 2 episode.

    state must contain:
      scenario_id: str                      — active scenario
      agent_verdict: str                    — "true_positive" | "false_positive"
      agent_attack_type: str                — e.g. "brute_force"
      agent_attacker_ip: str                — identified attacker IP
      agent_queried_log_ids: List[str]      — log IDs the agent retrieved
      agent_queried_sources: List[str]      — log sources queried
    """
    gt = _get_ground_truth(state)

    verdict_score = _score_verdict(state, gt)
    attack_type_score = _score_attack_type(state, gt)
    evidence_score = _score_evidence(state, gt)
    attacker_score = _score_attacker_id(state, gt)

    total = (
        verdict_score * VERDICT_WEIGHT
        + attack_type_score * ATTACK_TYPE_WEIGHT
        + evidence_score * EVIDENCE_WEIGHT
        + attacker_score * ATTACKER_ID_WEIGHT
    )

    return round(min(max(total, 0.001), 0.999), 4)


_VERDICT_ALIASES = {
    "true_positive": {"true_positive", "tp", "true_pos", "malicious", "confirmed"},
    "false_positive": {"false_positive", "fp", "false_pos", "benign", "not_malicious"},
}

def _score_verdict(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    agent = state.get("agent_verdict", "").lower().replace(" ", "_").strip()
    truth = gt["verdict"]
    
    if agent == truth:
        return 1.0
    
    aliases = _VERDICT_ALIASES.get(truth, set())
    if agent in aliases:
        return 1.0
    
    return 0.0


# Attack type canonical aliases — maps common LLM phrasings to ground truth
_ATTACK_TYPE_ALIASES = {
    "brute_force_ssh": {
        "brute_force_ssh", "ssh_brute_force", "brute_force", "credential_brute_force",
        "password_guessing", "credential_stuffing", "ssh_attack", "brute_force_attack",
    },
    "phishing_lateral_movement": {
        "phishing_lateral_movement", "phishing_lateral", "spearphishing",
        "phishing_attack", "lateral_movement", "phishing_with_lateral",
        "spearphishing_lateral", "phishing",
    },
    "ransomware": {
        "ransomware", "ransomware_attack", "crypto_ransomware", "encryption_attack",
        "file_encryption", "ransom",
    },
    "insider_threat": {
        "insider_threat", "insider", "internal_threat", "data_exfiltration",
        "insider_data_theft", "malicious_insider",
    },
    "supply_chain_attack": {
        "supply_chain_attack", "supply_chain", "dependency_attack",
        "third_party_compromise", "vendor_compromise",
    },
    "advanced_persistent_threat": {
        "advanced_persistent_threat", "apt", "multi_stage_apt",
        "multi_stage_attack", "persistent_threat", "apt_attack",
    },
}


def _normalize_attack_type(raw: str) -> str:
    """Map an LLM attack type output to the canonical form."""
    a = raw.strip().lower().replace(" ", "_").replace("-", "_")
    for canonical, aliases in _ATTACK_TYPE_ALIASES.items():
        if a in aliases:
            return canonical
    return a


def _score_attack_type(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    agent = state.get("agent_attack_type", "").lower().replace(" ", "_")
    truth = gt["attack_type"]

    # Exact match
    if agent == truth:
        return 1.0

    # Try canonical alias normalization
    norm_agent = _normalize_attack_type(agent)
    norm_truth = _normalize_attack_type(truth)
    if norm_agent == norm_truth:
        return 1.0

    # Substring containment
    if agent and (truth in agent or agent in truth):
        return 0.7  # Close match (e.g., brute_force vs brute_force_ssh)

    # Check if normalized forms share aliases
    for canonical, aliases in _ATTACK_TYPE_ALIASES.items():
        if norm_agent in aliases and norm_truth in aliases:
            return 0.9  # Same family

    # Keyword overlap check
    agent_words = set(agent.split("_"))
    truth_words = set(truth.split("_"))
    overlap = agent_words & truth_words
    if overlap and len(overlap) >= len(truth_words) * 0.5:
        return 0.4  # Partial keyword match

    return 0.0


def _score_evidence(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """
    Partial credit for each key piece of evidence retrieved.
    Also rewards querying the right log sources.
    Includes ordered strategy bonus (B6) — querying key sources
    first (before irrelevant ones) gets extra credit.
    """
    queried_ids: Set[str] = set(state.get("agent_queried_log_ids", []))
    queried_sources_ordered: list = state.get("agent_queried_sources", [])
    queried_sources: Set[str] = set(queried_sources_ordered)

    key_ids = gt["key_evidence_log_ids"]
    key_sources = gt["key_log_sources"]

    id_score = len(queried_ids & key_ids) / len(key_ids) if key_ids else 0.0
    source_score = len(queried_sources & key_sources) / len(key_sources) if key_sources else 0.0

    # B6: Ordered strategy bonus — reward querying key sources FIRST
    strategy_bonus = 0.0
    if queried_sources_ordered and key_sources:
        # Check if the first N queries (where N = len(key_sources)) hit key sources
        first_queries = queried_sources_ordered[:max(len(key_sources), 2)]
        relevant_first = [s for s in first_queries if s in key_sources]
        if len(relevant_first) >= 2:
            strategy_bonus = 0.15  # Queried 2+ key sources in first moves
        elif len(relevant_first) >= 1:
            strategy_bonus = 0.08  # At least 1 key source queried early

    # Penalize for querying too many irrelevant sources (thrashing)
    irrelevant = len(queried_sources - key_sources)
    noise_penalty = min(irrelevant * 0.03, 0.12)

    return max((id_score * 0.4 + source_score * 0.45 + strategy_bonus) - noise_penalty, 0.0)


def _score_attacker_id(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    agent_ip = state.get("agent_attacker_ip", "").strip()
    truth_ip = gt["attacker_ip"]
    # If scenario has no external attacker (e.g. insider threat)
    if not truth_ip:
        agent_lower = agent_ip.lower()
        if not agent_ip or "internal" in agent_lower or "insider" in agent_lower or agent_lower == "n/a":
            return 1.0  # correctly identified as insider / no external IP
        return 0.5  # provided an IP when none exists
    if agent_ip == truth_ip:
        return 1.0

    # Partial credit for same subnet
    agent_parts = agent_ip.split(".")
    truth_parts = truth_ip.split(".")
    if len(agent_parts) == 4 and len(truth_parts) == 4:
        if agent_parts[:3] == truth_parts[:3]:
            return 0.6  # Same /24 subnet
        if agent_parts[:2] == truth_parts[:2]:
            return 0.3  # Same /16 subnet

    return 0.0


def compute_step_reward(action_type: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> float:
    """
    Dense per-step reward for Task 2.
    """
    gt = _get_ground_truth(state)
    reward = 0.0

    if action_type == "query_logs":
        source = parameters.get("log_source", "")
        previously_rewarded_sources = set(state.get("rewarded_sources", []))
        
        if source not in previously_rewarded_sources:
            if source in gt["key_log_sources"]:
                reward += 0.15  # relevant log source
            else:
                reward -= 0.03  # irrelevant query costs time
            state.setdefault("rewarded_sources", []).append(source)

        # Bonus if key evidence log ID found
        queried_ids = set(state.get("agent_queried_log_ids", []))
        previously_rewarded = set(state.get("rewarded_log_ids", []))
        new_key_evidence = (queried_ids & gt["key_evidence_log_ids"]) - previously_rewarded
        reward += len(new_key_evidence) * 0.1
        state.setdefault("rewarded_log_ids", []).extend(new_key_evidence)

    elif action_type == "classify_alert":
        classification = parameters.get("classification", "")
        gt_classification = ATTACK_SCENARIOS.get(
            gt["scenario_id"], {}
        ).get("ground_truth", {}).get("classification", "critical")
        if classification == gt_classification:
            reward += 0.1  # correct classification

    elif action_type == "close_incident":
        verdict = parameters.get("verdict", "").lower().replace(" ", "_").strip()
        attack_type = parameters.get("attack_type", "").lower().replace(" ", "_")
        
        # Step reward for verdict
        if verdict == gt["verdict"] or verdict in _VERDICT_ALIASES.get(gt["verdict"], set()):
            reward += 0.3
            
        # Step reward for attack_type (fuzzy match via aliases)
        truth_type = gt["attack_type"]
        norm_agent_type = _normalize_attack_type(attack_type)
        norm_truth_type = _normalize_attack_type(truth_type)
        if attack_type == truth_type or norm_agent_type == norm_truth_type:
            reward += 0.2
        elif attack_type and (truth_type in attack_type or attack_type in truth_type):
            reward += 0.15
        else:
            agent_words = set(attack_type.split("_"))
            truth_words = set(truth_type.split("_"))
            overlap = agent_words & truth_words
            if overlap and len(overlap) >= len(truth_words) * 0.5:
                reward += 0.1

    elif action_type == "escalate":
        reward += 0.05  # reasonable action for high-severity

    return round(reward, 4)
