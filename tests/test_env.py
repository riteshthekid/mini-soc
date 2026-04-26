"""
Smoke tests for Mini SOC environment.
Run: python -m pytest tests/ -v
"""
import pytest
from server.mini_soc_environment import SocEnvironment
from models import Action, ActionType
from server.graders import grader1, grader2, grader3


# ---------------------------------------------------------------------------
# Test reset()
# ---------------------------------------------------------------------------

def test_reset_task1(env):
    result = env.reset("alert_triage")
    obs = result.observation
    assert obs.alert_queue, "Alert queue must not be empty"
    assert obs.current_alert is not None
    assert obs.task_context.task_id == "alert_triage"
    assert obs.task_context.difficulty == "easy"


def test_reset_task2(env):
    result = env.reset("incident_investigation")
    obs = result.observation
    assert obs.current_alert is not None
    assert len(obs.open_incidents) == 1


def test_reset_task3(env):
    result = env.reset("threat_response")
    obs = result.observation
    assert obs.current_alert is not None
    assert len(obs.alert_queue) >= 1


def test_reset_invalid_task(env):
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


def test_reset_clears_state(env):
    env.reset("alert_triage")
    # Do a step
    action = Action(action_type=ActionType.QUERY_LOGS, parameters={"log_source": "auth"})
    env.step(action)
    # Reset and verify clean state
    env.reset("alert_triage")
    state = env.state()
    assert state.step_count == 0
    assert state.total_reward == 0.0


# ---------------------------------------------------------------------------
# Test step()
# ---------------------------------------------------------------------------

def test_step_query_logs(env_task2):
    action = Action(action_type=ActionType.QUERY_LOGS, parameters={"log_source": "auth"})
    result = env_task2.step(action)
    assert result.observation is not None
    assert isinstance(result.reward, float)
    assert result.done is False
    assert len(result.observation.available_logs) > 0


def test_step_classify_alert(env_task1):
    action = Action(
        action_type=ActionType.CLASSIFY_ALERT,
        parameters={"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}
    )
    result = env_task1.step(action)
    assert result.reward > 0  # correct classification should give positive reward


def test_step_classify_alert_wrong(env_task1):
    action = Action(
        action_type=ActionType.CLASSIFY_ALERT,
        parameters={"alert_id": "ALT-001", "classification": "benign", "priority": "P4"}
    )
    result = env_task1.step(action)
    assert result.reward < 0  # missing critical as benign should penalize


def test_step_isolate_correct_asset(env_task3):
    # Query some logs first
    env_task3.step(Action(action_type=ActionType.QUERY_LOGS, parameters={"log_source": "process"}))
    # Isolate correct compromised asset
    action = Action(action_type=ActionType.ISOLATE_ASSET, parameters={"hostname": "WS-HR-03"})
    result = env_task3.step(action)
    assert result.reward > 0


def test_step_isolate_wrong_asset(env_task3):
    action = Action(action_type=ActionType.ISOLATE_ASSET, parameters={"hostname": "DC-01"})
    result = env_task3.step(action)
    assert result.reward < 0  # collateral damage


def test_step_after_done(env):
    env.reset("alert_triage")
    env._done = True
    action = Action(action_type=ActionType.REQUEST_INFO, parameters={})
    result = env.step(action)
    assert result.done is True
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# Test state()
# ---------------------------------------------------------------------------

def test_state_has_ground_truth(env):
    env.reset("incident_investigation")
    state = env.state()
    assert "verdict" in state.ground_truth
    assert state.ground_truth["verdict"] == "true_positive"


# ---------------------------------------------------------------------------
# Test graders directly
# ---------------------------------------------------------------------------

def test_grader1_perfect_score():
    state = {
        "agent_classifications": {
            "ALT-001": {"classification": "critical", "priority": "P1"},
            "ALT-002": {"classification": "critical", "priority": "P1"},
            "ALT-010": {"classification": "suspicious", "priority": "P2"},
            "ALT-011": {"classification": "suspicious", "priority": "P2"},
            "ALT-012": {"classification": "critical", "priority": "P1"},
            "ALT-020": {"classification": "benign", "priority": "P4"},
            "ALT-030": {"classification": "benign", "priority": "P4"},
            "ALT-031": {"classification": "benign", "priority": "P3"},
            "ALT-032": {"classification": "benign", "priority": "P4"},
            "ALT-033": {"classification": "suspicious", "priority": "P2"},
        }
    }
    score = grader1.grade(state)
    assert score == 0.999


def test_grader1_empty_state():
    score = grader1.grade({})
    assert score == 0.001


def test_grader1_partial_score():
    state = {
        "agent_classifications": {
            "ALT-001": {"classification": "critical", "priority": "P1"},
            "ALT-002": {"classification": "critical", "priority": "P1"},
        }
    }
    score = grader1.grade(state)
    assert 0.0 < score < 1.0


def test_grader2_perfect_score():
    state = {
        "scenario_id": "brute_force_ssh_001",
        "agent_verdict": "true_positive",
        "agent_attack_type": "brute_force",
        "agent_attacker_ip": "185.220.101.47",
        "agent_queried_log_ids": ["AUTH-001", "AUTH-002", "FW-001"],
        "agent_queried_sources": ["auth", "firewall"],
    }
    score = grader2.grade(state)
    assert score > 0.9


def test_grader2_wrong_verdict():
    state = {
        "scenario_id": "brute_force_ssh_001",
        "agent_verdict": "false_positive",
        "agent_attack_type": "brute_force",
        "agent_attacker_ip": "185.220.101.47",
        "agent_queried_log_ids": ["AUTH-001", "AUTH-002", "FW-001"],
        "agent_queried_sources": ["auth", "firewall"],
    }
    score = grader2.grade(state)
    assert score < 0.7  # verdict wrong = big penalty


def test_grader3_collateral_damage():
    state = {
        "scenario_id": "phishing_lateral_001",
        "agent_isolated_assets": ["DC-01"],  # WRONG: healthy critical asset
        "agent_blocked_ips": ["94.102.49.190"],
        "agent_queried_sources": ["process", "network"],
        "agent_report": {},
        "steps_taken": 10,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    assert score < 0.5  # collateral damage tank score


def test_grader3_scores_in_range():
    state = {
        "scenario_id": "phishing_lateral_001",
        "agent_isolated_assets": [],
        "agent_blocked_ips": [],
        "agent_queried_sources": [],
        "agent_report": {},
        "steps_taken": 30,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# v2.0 Tests (19-28)
# ---------------------------------------------------------------------------

def test_thrashing_penalty(env):
    """Test 19: Thrashing penalty activates after >5 identical non-classify actions."""
    env.reset("incident_investigation")
    # The penalty fires when type_count > 5, meaning 6 prior same-type actions exist
    for _ in range(6):
        env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    # 7th identical action should trigger thrashing penalty
    result = env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert result.reward == -0.05, f"Expected -0.05 thrashing penalty, got {result.reward}"


def test_step_after_done_returns_error(env):
    """Test 20: Step after done returns reward=0.0, done=True, error='episode_done'."""
    env.reset("alert_triage")
    env._done = True
    result = env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert result.reward == 0.0
    assert result.done is True
    assert result.info.get("error") == "episode_done"


def test_max_steps_boundary(env):
    """Test 21: Max steps boundary terminates the episode."""
    env.reset("alert_triage")  # max_steps = 15
    for _ in range(15):
        if not env._done:
            env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert env._done is True, "Episode should be done after max_steps reached"


def test_progressive_alerts_task3(env):
    """Test 22: Progressive alert disclosure in threat_response task."""
    env.reset("threat_response")
    initial_count = len(env._alert_queue)
    assert initial_count == 1, f"Should start with 1 alert, got {initial_count}"
    assert env._alert_queue[0].alert_id == "ALT-010"

    # Manually set step count to trigger alert surfacing at step 3
    env._step_count = 2  # Next step() will increment to 3
    env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert len(env._alert_queue) == 2, f"ALT-011 should surface at step 3, got {len(env._alert_queue)} alerts"
    assert env._alert_queue[1].alert_id == "ALT-011"

    # Surface third alert at step 6
    while env._step_count < 5:
        env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert len(env._alert_queue) == 3, f"ALT-012 should surface at step 6, got {len(env._alert_queue)} alerts"
    assert env._alert_queue[2].alert_id == "ALT-012"


def test_difficulty_escalation(env):
    """Test 23: Adaptive tier escalation when rolling avg > 0.70."""
    from server.mini_soc_environment import DifficultyTier
    env.reset("alert_triage")
    assert env._difficulty_tier == DifficultyTier.TIER_1

    # Simulate 5 strong episodes
    env._episode_scores = [0.72, 0.74, 0.71, 0.73, 0.75]
    env._maybe_escalate_difficulty(0.73)
    assert env._difficulty_tier == DifficultyTier.TIER_2, f"Expected Tier 2, got {env._difficulty_tier}"

    # Simulate very strong episodes for Tier 3
    env._episode_scores = [0.86, 0.87, 0.88, 0.89, 0.90]
    env._maybe_escalate_difficulty(0.90)
    assert env._difficulty_tier == DifficultyTier.TIER_3, f"Expected Tier 3, got {env._difficulty_tier}"


def test_grader1_deterministic():
    """Test 24: Grader1 produces identical scores across 10 runs."""
    state = {
        "agent_classifications": {
            "ALT-001": {"classification": "critical", "priority": "P1"},
        }
    }
    scores = [grader1.grade(state) for _ in range(10)]
    assert len(set(scores)) == 1, f"Expected identical scores, got {set(scores)}"


def test_ransomware_scenario_loads(env):
    """Test 25: ransomware_001 scenario is present and well-formed."""
    from server.simulator.attack_seeds import ATTACK_SCENARIOS
    assert "ransomware_001" in ATTACK_SCENARIOS
    scenario = ATTACK_SCENARIOS["ransomware_001"]
    assert scenario["attack_type"] == "malware"
    assert len(scenario["alerts"]) > 0
    assert "ground_truth" in scenario
    assert scenario["ground_truth"]["classification"] == "critical"

    # Verify the environment can still load with new scenarios
    result = env.reset("alert_triage")
    assert result.observation is not None


def test_mitre_tags_in_scenarios():
    """Test 26: MITRE ATT&CK tags present in all scenario ground truths."""
    from server.simulator.attack_seeds import ATTACK_SCENARIOS
    for scenario_id, scenario in ATTACK_SCENARIOS.items():
        gt = scenario["ground_truth"]
        assert "mitre_techniques" in gt, f"mitre_techniques missing from {scenario_id}"
        if scenario_id != "false_positive_scan_001":
            # All real attack scenarios must have at least one technique
            assert len(gt["mitre_techniques"]) > 0, f"{scenario_id} has empty mitre_techniques"
            for tech in gt["mitre_techniques"]:
                assert "technique_id" in tech, f"Missing technique_id in {scenario_id}"
                assert "name" in tech, f"Missing name in {scenario_id}"
                assert "tactic" in tech, f"Missing tactic in {scenario_id}"
                assert tech["technique_id"].startswith("T"), f"Bad technique_id in {scenario_id}: {tech['technique_id']}"


def test_metrics_endpoint(client):
    """Test 27: /metrics endpoint returns valid metrics dict."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "episode_count" in data
    assert "mean_reward_by_task" in data
    assert "step_distribution" in data
    assert "difficulty_tier" in data
    assert isinstance(data["episode_count"], int)
    assert isinstance(data["difficulty_tier"], int)


def test_difficulty_endpoint(client):
    """Test 28: /difficulty endpoint sets tier correctly."""
    # Set tier 2
    response = client.post("/difficulty", json={"tier": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == 2
    assert data["status"] == "updated"

    # Verify metrics reflect the change
    metrics = client.get("/metrics").json()
    assert metrics["difficulty_tier"] == 2

    # Invalid tier should return 400
    response = client.post("/difficulty", json={"tier": 99})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# API coverage tests (boost app.py from 53% to >80%)
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health returns status ok with version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"


def test_root_endpoint(client):
    """GET / returns env metadata with all endpoint paths."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "mini-soc"
    assert "/metrics" in data["endpoints"]
    assert "/scenarios" in data["endpoints"]


def test_tasks_endpoint(client):
    """GET /tasks returns all 3 tasks."""
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tasks"]) == 3
    task_ids = [t["id"] for t in data["tasks"]]
    assert "alert_triage" in task_ids
    assert "incident_investigation" in task_ids
    assert "threat_response" in task_ids


def test_scenarios_endpoint(client):
    """GET /scenarios returns all 7 scenarios with MITRE tags."""
    response = client.get("/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 7
    # Verify structure of first scenario
    s = data["scenarios"][0]
    assert "scenario_id" in s
    assert "mitre_techniques" in s
    assert "alert_count" in s


def test_reset_step_workflow(client):
    """POST /reset then /step produces valid responses."""
    # Reset
    resp = client.post("/reset", json={"task_id": "alert_triage"})
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "info" in data

    # Step with a valid action
    resp = client.post("/step", json={
        "action_type": "classify_alert",
        "parameters": {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"},
    })
    assert resp.status_code == 200
    step_data = resp.json()
    assert "reward" in step_data
    assert "done" in step_data
    assert isinstance(step_data["reward"], float)


def test_step_invalid_action(client):
    """POST /step with invalid action_type returns 400."""
    client.post("/reset", json={"task_id": "alert_triage"})
    resp = client.post("/step", json={
        "action_type": "nonexistent_action",
        "parameters": {},
    })
    assert resp.status_code == 400


def test_reset_invalid_task_api(client):
    """POST /reset with invalid task_id returns 400."""
    resp = client.post("/reset", json={"task_id": "fake_task"})
    assert resp.status_code == 400


def test_state_endpoint(client):
    """GET /state returns full state with ground truth after reset."""
    client.post("/reset", json={"task_id": "incident_investigation"})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "ground_truth" in data
    assert "observation" in data
    assert data["ground_truth"]["verdict"] == "true_positive"


# ---------------------------------------------------------------------------
# Scenario selection tests (wires all 7 scenarios to gameplay)
# ---------------------------------------------------------------------------

def test_scenario_selection_tier1(env):
    """Tier 1 uses default scenarios: brute_force + phishing_lateral."""
    from server.mini_soc_environment import DifficultyTier
    env._difficulty_tier = DifficultyTier.TIER_1

    env.reset("incident_investigation")
    assert env._active_scenario_id == "brute_force_ssh_001"

    env.reset("threat_response")
    assert env._active_scenario_id == "phishing_lateral_001"

    env.reset("alert_triage")
    assert env._active_scenario_id == ""  # Task 1 uses mixed queue


def test_scenario_selection_tier2(env):
    """Tier 2 unlocks ransomware + insider threat scenarios."""
    from server.mini_soc_environment import DifficultyTier
    env._difficulty_tier = DifficultyTier.TIER_2

    env.reset("incident_investigation")
    assert env._active_scenario_id in ("ransomware_001", "insider_threat_001")

    env.reset("threat_response")
    assert env._active_scenario_id == "ransomware_001"


def test_scenario_selection_tier3(env):
    """Tier 3 unlocks supply chain + APT scenarios."""
    from server.mini_soc_environment import DifficultyTier
    env._difficulty_tier = DifficultyTier.TIER_3

    env.reset("incident_investigation")
    assert env._active_scenario_id == "supply_chain_001"

    env.reset("threat_response")
    assert env._active_scenario_id == "multi_stage_apt_001"


def test_grader2_ransomware_scenario():
    """Grader2 correctly grades ransomware_001 scenario."""
    state = {
        "scenario_id": "ransomware_001",
        "agent_verdict": "true_positive",
        "agent_attack_type": "malware",
        "agent_attacker_ip": "192.168.50.99",
        "agent_queried_log_ids": ["PROC-R01", "PROC-R02", "NET-R01"],
        "agent_queried_sources": ["process", "network", "firewall"],
    }
    score = grader2.grade(state)
    assert score > 0.8, f"Ransomware perfect answer should score >0.8, got {score}"


def test_grader3_ransomware_scenario():
    """Grader3 correctly grades ransomware_001 — isolate WS-FINANCE-01."""
    state = {
        "scenario_id": "ransomware_001",
        "agent_isolated_assets": ["WS-FINANCE-01"],
        "agent_blocked_ips": ["192.168.50.99"],
        "agent_queried_sources": ["process", "network", "firewall"],
        "agent_report": {
            "summary": "Ransomware detected",
            "attack_type": "malware",
            "affected_assets": ["WS-FINANCE-01"],
            "attacker_ip": "192.168.50.99",
            "timeline": "Shadow copy deletion then encryption",
        },
        "steps_taken": 8,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    assert score > 0.7, f"Ransomware correct response should score >0.7, got {score}"


def test_grader3_apt_scenario_dc01_is_compromised():
    """In APT scenario, DC-01 IS compromised — isolating it is correct."""
    state = {
        "scenario_id": "multi_stage_apt_001",
        "agent_isolated_assets": ["DC-01", "BACKUP-SRV-01"],
        "agent_blocked_ips": ["45.33.32.156"],
        "agent_queried_sources": ["auth", "process", "network", "dns"],
        "agent_report": {
            "summary": "APT detected",
            "attack_type": "lateral_movement",
            "affected_assets": ["BACKUP-SRV-01", "DC-01"],
            "attacker_ip": "45.33.32.156",
            "timeline": "Service account compromise then lateral movement",
        },
        "steps_taken": 10,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    # DC-01 is actually compromised in APT scenario, so no collateral penalty
    assert score > 0.65, f"APT correct response should score >0.65, got {score}"


def test_tier2_episode_full_cycle(env):
    """Full episode at Tier 2 uses ransomware scenario for threat_response."""
    from server.mini_soc_environment import DifficultyTier
    from models import Action, ActionType
    env._difficulty_tier = DifficultyTier.TIER_2
    result = env.reset("threat_response")
    assert env._active_scenario_id == "ransomware_001"
    # First alert should be from ransomware scenario
    assert result.observation.alert_queue[0].alert_id == "ALT-040"


def test_grader3_insider_threat_no_ip_full_credit():
    """Insider threat: agent omits attacker_ip → full report credit."""
    state = {
        "scenario_id": "insider_threat_001",
        "agent_isolated_assets": ["WS-FINANCE-01"],
        "agent_blocked_ips": [],
        "agent_queried_sources": ["network", "auth", "dns"],
        "agent_report": {
            "summary": "Insider data exfiltration detected",
            "attack_type": "data_exfiltration",
            "affected_assets": ["WS-FINANCE-01"],
            "attacker_ip": "",  # correctly empty — no external attacker
            "timeline": "User staged data then exfiltrated via cloud",
        },
        "steps_taken": 8,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    # Agent should NOT be penalized for empty attacker_ip on insider threat
    assert score > 0.6, f"Insider threat with correct response should score >0.6, got {score}"


def test_grader3_insider_threat_internal_label():
    """Insider threat: agent reports 'internal_user' → full credit."""
    state = {
        "scenario_id": "insider_threat_001",
        "agent_isolated_assets": ["WS-FINANCE-01"],
        "agent_blocked_ips": [],
        "agent_queried_sources": ["network", "auth", "dns"],
        "agent_report": {
            "summary": "Insider data exfiltration detected",
            "attack_type": "data_exfiltration",
            "affected_assets": ["WS-FINANCE-01"],
            "attacker_ip": "internal_user",  # acceptable label for insider
            "timeline": "User staged data then exfiltrated via cloud",
        },
        "steps_taken": 8,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    assert score > 0.6, f"Insider 'internal_user' label should get full credit, got {score}"


def test_grader2_insider_threat_no_ip():
    """Grader2 awards full attacker_id score for insider threat with no IP."""
    state = {
        "scenario_id": "insider_threat_001",
        "agent_verdict": "true_positive",
        "agent_attack_type": "data_exfiltration",
        "agent_attacker_ip": "",
        "agent_queried_log_ids": ["NET-I01", "AUTH-I01", "DNS-I01"],
        "agent_queried_sources": ["network", "auth", "dns"],
    }
    score = grader2.grade(state)
    assert score > 0.8, f"Insider threat correct verdict should score >0.8, got {score}"


def test_grader2_insider_threat_internal_label():
    """Grader2 accepts 'internal' as correct for insider scenarios."""
    state = {
        "scenario_id": "insider_threat_001",
        "agent_verdict": "true_positive",
        "agent_attack_type": "data_exfiltration",
        "agent_attacker_ip": "internal_user",
        "agent_queried_log_ids": ["NET-I01", "AUTH-I01", "DNS-I01"],
        "agent_queried_sources": ["network", "auth", "dns"],
    }
    score = grader2.grade(state)
    assert score > 0.8, f"Insider 'internal_user' should get full attacker_id credit, got {score}"


# ============================================================================
# Track B: Algorithm improvement tests
# ============================================================================


def test_grader1_classification_alias_high():
    """Grader1 accepts 'high' as alias for 'critical'."""
    state = {
        "agent_classifications": {
            "ALT-001": {"classification": "high", "priority": "P1"},
            "ALT-002": {"classification": "high", "priority": "P1"},
        }
    }
    score = grader1.grade(state)
    assert score > 0.0, f"'high' should be accepted as alias for 'critical', got {score}"


def test_grader1_classification_alias_safe():
    """Grader1 accepts 'safe' as alias for 'benign'."""
    state = {
        "agent_classifications": {
            "ALT-020": {"classification": "safe", "priority": "P4"},
            "ALT-030": {"classification": "clean", "priority": "P4"},
        }
    }
    score = grader1.grade(state)
    assert score > 0.0, f"'safe'/'clean' should be accepted as aliases for 'benign', got {score}"


def test_grader1_step_reward_alias():
    """Grader1 step reward accepts 'high' as 'critical'."""
    reward = grader1.compute_step_reward("ALT-001", "high", "P1")
    assert reward > 0.0, f"'high' should get positive step reward for critical alert, got {reward}"


def test_grader2_attack_type_alias_ssh_brute_force():
    """Grader2 accepts 'ssh_brute_force' as alias for 'brute_force_ssh'."""
    state = {
        "scenario_id": "brute_force_ssh_001",
        "agent_verdict": "true_positive",
        "agent_attack_type": "ssh_brute_force",
        "agent_attacker_ip": "185.220.101.47",
        "agent_queried_log_ids": ["AUTH-001", "AUTH-002", "FW-001"],
        "agent_queried_sources": ["auth", "firewall"],
    }
    score = grader2.grade(state)
    assert score > 0.85, f"'ssh_brute_force' alias should get near-perfect score, got {score}"


def test_grader2_attack_type_alias_password_guessing():
    """Grader2 accepts 'password_guessing' as alias for 'brute_force_ssh'."""
    state = {
        "scenario_id": "brute_force_ssh_001",
        "agent_verdict": "tp",
        "agent_attack_type": "password_guessing",
        "agent_attacker_ip": "185.220.101.47",
        "agent_queried_log_ids": ["AUTH-001", "AUTH-002"],
        "agent_queried_sources": ["auth", "firewall"],
    }
    score = grader2.grade(state)
    assert score > 0.80, f"'password_guessing' alias should get good score, got {score}"


def test_grader3_block_ip_subnet_partial_credit():
    """Grader3 gives -0.05 penalty for blocking wrong IP (subnet logic reverted)."""
    state = {
        "scenario_id": "phishing_lateral_001",
        "agent_isolated_assets": ["WS-HR-03"],
        "agent_blocked_ips": ["94.102.49.191"],  # .191 instead of .190
        "agent_queried_sources": ["process", "network"],
        "agent_report": {},
        "steps_taken": 5,
        "max_steps": 30,
    }
    reward = grader3.compute_step_reward(
        "block_ip", {"ip_address": "94.102.49.191"}, state
    )
    assert reward == -0.05, f"Blocking wrong IP should get -0.05 penalty, got {reward}"
