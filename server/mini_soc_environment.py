"""
Mini SOC Environment — Core Episode Logic
Implements the OpenEnv state machine: reset(), step(), state().
"""
from __future__ import annotations
import uuid
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

# Dual-import pattern: supports both package-mode and Docker-mode
try:
    from models import (
        Action, ActionType, Alert, Asset, Incident,
        Observation, ResetResult, Reward, StateResult,
        StepResult, TaskContext, LogEntry,
    )
except ImportError:
    from ..models import (
        Action, ActionType, Alert, Asset, Incident,
        Observation, ResetResult, Reward, StateResult,
        StepResult, TaskContext, LogEntry,
    )

try:
    from server.simulator.attack_seeds import (
        ATTACK_SCENARIOS, TASK1_ALERT_QUEUE, ASSET_INVENTORY,
    )
except ImportError:
    from .simulator.attack_seeds import (
        ATTACK_SCENARIOS, TASK1_ALERT_QUEUE, ASSET_INVENTORY,
    )

try:
    from server.simulator.log_gen import (
        get_logs_for_source, build_asset_inventory,
        sanitize_for_agent, get_benign_log_noise,
    )
except ImportError:
    from .simulator.log_gen import (
        get_logs_for_source, build_asset_inventory,
        sanitize_for_agent, get_benign_log_noise,
    )

try:
    from server.graders import grader1, grader2, grader3
except ImportError:
    from .graders import grader1, grader2, grader3


class DifficultyTier:
    """Adaptive difficulty levels. Higher tiers unlock harder scenarios."""
    TIER_1 = 1  # Default: single attacker, 2-3 log sources needed
    TIER_2 = 2  # Auto-unlock at rolling_avg > 0.70: decoy IPs, tighter SLA
    TIER_3 = 3  # Auto-unlock at rolling_avg > 0.85: APT, 14-day log window


TASK_CONFIG = {
    "alert_triage": {
        "max_steps": 15,
        "scenario_id": None,
        "grader": grader1,
        "objective": (
            "Classify all 10 alerts in the queue as benign/suspicious/critical "
            "and assign correct priority (P1–P4) to each."
        ),
    },
    "incident_investigation": {
        "max_steps": 20,
        "scenario_id": "brute_force_ssh_001",
        "grader": grader2,
        "objective": (
            "Investigate the active incident: query relevant log sources, "
            "identify the attacker IP, and submit a verdict with attack type."
        ),
    },
    "threat_response": {
        "max_steps": 30,
        "scenario_id": "phishing_lateral_001",
        "grader": grader3,
        "objective": (
            "A multi-stage attack is active. Gather evidence, isolate compromised assets, "
            "block attacker IPs, and write a full incident report."
        ),
    },
}


class SocEnvironment:
    """
    Mini SOC OpenEnv Environment.
    Call reset(task_id) to start an episode, then step(action) repeatedly.
    """

    def __init__(self):
        self._episode_id: str = ""
        self._task_id: str = ""
        self._active_scenario_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0

        # Adaptive difficulty tracking (persists across episodes)
        self._difficulty_tier: int = DifficultyTier.TIER_1
        self._episode_scores: List[float] = []
        self._episode_count: int = 0
        self._metrics: Dict[str, Any] = {
            "episode_count": 0,
            "mean_reward_by_task": {},
            "step_distribution": [],
            "difficulty_tier": DifficultyTier.TIER_1,
        }

        # Episode state
        self._alert_queue: List[Alert] = []
        self._current_alert: Optional[Alert] = None
        self._available_logs: List[LogEntry] = []
        self._asset_inventory: List[Asset] = []
        self._open_incidents: List[Incident] = []
        self._actions_taken: List[str] = []
        self._task_context: Optional[TaskContext] = None

        # Grader tracking state
        self._agent_classifications: Dict[str, Dict[str, str]] = {}
        self._agent_queried_log_ids: List[str] = []
        self._agent_queried_sources: List[str] = []
        self._agent_isolated_assets: List[str] = []
        self._agent_blocked_ips: List[str] = []
        self._agent_verdict: str = ""
        self._agent_attack_type: str = ""
        self._agent_attacker_ip: str = ""
        self._agent_report: Dict[str, Any] = {}
        self._rewarded_log_ids: List[str] = []
        self._rewarded_sources: List[str] = []

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "alert_triage") -> ResetResult:
        """Start a fresh episode for the given task."""
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIG.keys())}")

        config = TASK_CONFIG[task_id]
        self._episode_id = str(uuid.uuid4())[:8]
        self._task_id = task_id
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._actions_taken = []

        # Reset grader state
        self._agent_classifications = {}
        self._agent_queried_log_ids = []
        self._agent_queried_sources = []
        self._agent_isolated_assets = []
        self._agent_blocked_ips = []
        self._agent_verdict = ""
        self._agent_attack_type = ""
        self._agent_attacker_ip = ""
        self._agent_report = {}

        # Build episode state — select scenario based on difficulty tier
        scenario_id = self._select_scenario(task_id)
        self._active_scenario_id = scenario_id
        self._asset_inventory = build_asset_inventory(scenario_id)
        self._available_logs = []
        self._open_incidents = []

        if task_id == "alert_triage":
            self._alert_queue = [Alert(**{k: v for k, v in a.items() if k != "ground_truth_classification" and k != "ground_truth_priority"})
                                  for a in TASK1_ALERT_QUEUE]
            self._current_alert = self._alert_queue[0] if self._alert_queue else None

        elif task_id == "incident_investigation":
            scenario = ATTACK_SCENARIOS[scenario_id]
            self._alert_queue = [Alert(**a) for a in scenario["alerts"]]
            self._current_alert = self._alert_queue[0]
            incident = Incident(
                incident_id=f"INC-{self._episode_id}",
                alert_ids=[a.alert_id for a in self._alert_queue],
                status="open",
            )
            self._open_incidents = [incident]

        elif task_id == "threat_response":
            scenario = ATTACK_SCENARIOS[scenario_id]
            # Start with only first alert visible; more surface as steps progress
            self._alert_queue = [Alert(**scenario["alerts"][0])]
            self._current_alert = self._alert_queue[0]
            incident = Incident(
                incident_id=f"INC-{self._episode_id}",
                alert_ids=[scenario["alerts"][0]["alert_id"]],
                status="open",
            )
            self._open_incidents = [incident]

        self._task_context = TaskContext(
            task_id=task_id,
            task_name=task_id.replace("_", " ").title(),
            difficulty={"alert_triage": "easy", "incident_investigation": "medium", "threat_response": "hard"}[task_id],
            objective=config["objective"],
            max_steps=config["max_steps"],
        )

        obs = self._build_observation(message=f"Episode started. Task: {task_id}. {config['objective']}")
        return ResetResult(observation=obs, info={"episode_id": self._episode_id, "task_id": task_id})

    def step(self, action: Action) -> StepResult:
        """Execute one agent action and return next observation + reward."""
        if self._done:
            obs = self._build_observation(message="Episode already done. Call reset().")
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "episode_done"})

        self._step_count += 1
        config = TASK_CONFIG[self._task_id]
        self._task_context.current_step = self._step_count

        # Reveal new alerts in threat_response as episode progresses
        if self._task_id == "threat_response":
            self._surface_new_alerts()

        # Process action
        reward, message, error = self._process_action(action)
        self._total_reward += reward
        self._actions_taken.append(f"step={self._step_count} {action.action_type.value}")

        # Check terminal conditions
        done = self._check_done(config["max_steps"])
        self._done = done

        obs = self._build_observation(message=message)
        info = {
            "step": self._step_count,
            "total_reward": round(self._total_reward, 4),
            "error": error,
        }
        if done:
            final_score = self._compute_final_score()
            info["final_score"] = final_score
            info["difficulty_tier"] = self._difficulty_tier
            self._maybe_escalate_difficulty(final_score)

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> StateResult:
        """Full internal state including ground truth. Used by graders."""
        return StateResult(
            observation=self._build_observation(),
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step_count,
            total_reward=self._total_reward,
            done=self._done,
            ground_truth=self._build_ground_truth(),
        )

    # -----------------------------------------------------------------------
    # Action processing
    # -----------------------------------------------------------------------

    def _process_action(self, action: Action) -> Tuple[float, str, Optional[str]]:
        """Returns (reward, message, error_string)."""
        at = action.action_type
        params = action.parameters
        reward = 0.0
        error = None

        # Penalize repeated identical actions (thrashing)
        action_sig = f"{at.value}:{str(sorted(params.items()))}"
        past_sigs = [f"{a.split(' ', 1)[1] if ' ' in a else a}" for a in self._actions_taken]
        type_count = sum(1 for a in self._actions_taken if at.value in a)
        # Only penalize if same action TYPE repeated >5 times AND it's not classify_alert
        # (classify_alert legitimately needs 10 calls with different alert_ids)
        if type_count > 5 and at.value != "classify_alert":
            reward -= 0.05
            return reward, "Warning: repeated action type detected.", "thrashing_penalty"

        if at == ActionType.QUERY_LOGS:
            reward, message = self._handle_query_logs(params)

        elif at == ActionType.CLASSIFY_ALERT:
            reward, message = self._handle_classify_alert(params)

        elif at == ActionType.ISOLATE_ASSET:
            reward, message = self._handle_isolate_asset(params)

        elif at == ActionType.BLOCK_IP:
            reward, message = self._handle_block_ip(params)

        elif at == ActionType.ESCALATE:
            reward, message = self._handle_escalate(params)

        elif at == ActionType.WRITE_REPORT:
            reward, message = self._handle_write_report(params)

        elif at == ActionType.CLOSE_INCIDENT:
            reward, message = self._handle_close_incident(params)

        elif at == ActionType.REQUEST_INFO:
            reward = 0.0
            message = "Info request noted. Consult available logs and asset inventory."

        else:
            reward = -0.05
            message = f"Unknown action type: {at}"
            error = "unknown_action"

        return round(reward, 4), message, error

    def _handle_query_logs(self, params: Dict) -> Tuple[float, str]:
        source = params.get("log_source", "")
        filter_ip = params.get("filter_ip")
        filter_user = params.get("filter_user")
        scenario_id = TASK_CONFIG[self._task_id]["scenario_id"]

        if not scenario_id:
            return -0.02, "No scenario logs available for this task."

        logs = get_logs_for_source(scenario_id, source, filter_ip, filter_user)
        noise = get_benign_log_noise(source, count=2)
        all_logs = logs + noise

        # Track for grader
        for log in logs:
            if log.log_id not in self._agent_queried_log_ids:
                self._agent_queried_log_ids.append(log.log_id)
        if source not in self._agent_queried_sources:
            self._agent_queried_sources.append(source)

        # Merge into available_logs (deduplicate)
        existing_ids = {l.log_id for l in self._available_logs}
        for log in all_logs:
            if log.log_id not in existing_ids:
                self._available_logs.append(log)
                existing_ids.add(log.log_id)

        # Compute reward based on task
        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward(
            "query_logs", params, self._build_grader_state()
        )
        return reward, f"Retrieved {len(logs)} logs from {source} source. {len(noise)} background entries also loaded."

    def _handle_classify_alert(self, params: Dict) -> Tuple[float, str]:
        alert_id = params.get("alert_id", "")
        classification = params.get("classification", "")
        priority = params.get("priority", "P3")

        if not alert_id or not classification:
            return -0.05, "classify_alert requires alert_id and classification."

        self._agent_classifications[alert_id] = {
            "classification": classification,
            "priority": priority,
        }

        if self._task_id == "alert_triage":
            reward = grader1.compute_step_reward(alert_id, classification, priority)
            # Advance to next unclassified alert
            classified_ids = set(self._agent_classifications.keys())
            for alert in self._alert_queue:
                if alert.alert_id not in classified_ids:
                    self._current_alert = alert
                    break
            return reward, f"Alert {alert_id} classified as {classification} (priority {priority})."

        elif self._task_id == "incident_investigation":
            reward = grader2.compute_step_reward("classify_alert", params, self._build_grader_state())
            return reward, f"Alert {alert_id} classified as {classification}."

        return 0.05, f"Alert {alert_id} classified."

    def _handle_isolate_asset(self, params: Dict) -> Tuple[float, str]:
        hostname = params.get("hostname", "")
        if not hostname:
            return -0.05, "isolate_asset requires hostname."

        for asset in self._asset_inventory:
            if asset.hostname == hostname:
                asset.is_isolated = True
                break

        if hostname not in self._agent_isolated_assets:
            self._agent_isolated_assets.append(hostname)

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("isolate_asset", params, self._build_grader_state())
        return reward, f"Asset {hostname} isolated from network."

    def _handle_block_ip(self, params: Dict) -> Tuple[float, str]:
        ip = params.get("ip_address", "")
        if not ip:
            return -0.05, "block_ip requires ip_address."

        if ip not in self._agent_blocked_ips:
            self._agent_blocked_ips.append(ip)

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("block_ip", params, self._build_grader_state())
        return reward, f"IP {ip} blocked on perimeter firewall."

    def _handle_escalate(self, params: Dict) -> Tuple[float, str]:
        reason = params.get("reason", "no reason provided")
        reward = 0.05 if self._task_id != "alert_triage" else -0.02
        return reward, f"Alert escalated to Tier-2. Reason: {reason}"

    def _handle_write_report(self, params: Dict) -> Tuple[float, str]:
        report = params.get("report", params)  # accept report as nested or flat
        self._agent_report = report

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("write_report", {"report": report}, self._build_grader_state())
        return reward, "Incident report submitted."

    def _handle_close_incident(self, params: Dict) -> Tuple[float, str]:
        verdict = params.get("verdict", "")
        attack_type = params.get("attack_type", "")
        self._agent_verdict = verdict
        self._agent_attack_type = attack_type

        attacker_ip = params.get("attacker_ip", "")
        if attacker_ip:
            self._agent_attacker_ip = attacker_ip

        for inc in self._open_incidents:
            inc.status = "resolved"
            inc.verdict = verdict
            inc.attack_type = attack_type

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("close_incident", params, self._build_grader_state())

        self._done = True  # closing incident ends episode
        return reward, f"Incident closed. Verdict: {verdict}. Attack type: {attack_type}."

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _surface_new_alerts(self):
        """Reveal additional alerts progressively in threat_response task."""
        if not self._active_scenario_id:
            return
        scenario = ATTACK_SCENARIOS.get(self._active_scenario_id)
        if not scenario:
            return
        all_alerts = scenario["alerts"]
        reveal_at_steps = {3: 1, 6: 2}  # step_count → reveal up to index
        idx = reveal_at_steps.get(self._step_count)
        if idx is not None and idx < len(all_alerts):
            existing_ids = {a.alert_id for a in self._alert_queue}
            alert = Alert(**all_alerts[idx])
            if alert.alert_id not in existing_ids:
                self._alert_queue.append(alert)
                if self._open_incidents:
                    self._open_incidents[0].alert_ids.append(alert.alert_id)

    def _check_done(self, max_steps: int) -> bool:
        if self._done:
            return True
        if self._step_count >= max_steps:
            return True
        # Task 1: done when all alerts classified
        if self._task_id == "alert_triage":
            classified = set(self._agent_classifications.keys())
            all_ids = {a.alert_id for a in self._alert_queue}
            if all_ids and all_ids.issubset(classified):
                return True
        return False

    def _compute_final_score(self) -> float:
        grader = TASK_CONFIG[self._task_id]["grader"]
        return grader.grade(self._build_grader_state())

    def _maybe_escalate_difficulty(self, final_score: float) -> None:
        """
        Track episode scores and auto-escalate difficulty tier.
        Rolling window of last 5 episodes:
          - Tier 2 unlocks at rolling avg > 0.70
          - Tier 3 unlocks at rolling avg > 0.85
        Also updates internal metrics for the /metrics endpoint.
        """
        self._episode_scores.append(final_score)
        if len(self._episode_scores) > 5:
            self._episode_scores.pop(0)

        self._episode_count += 1

        # Update metrics
        task_key = self._task_id
        task_rewards = self._metrics.get("mean_reward_by_task", {})
        if task_key not in task_rewards:
            task_rewards[task_key] = {"total": 0.0, "count": 0}
        task_rewards[task_key]["total"] += final_score
        task_rewards[task_key]["count"] += 1
        self._metrics["mean_reward_by_task"] = task_rewards
        self._metrics["episode_count"] = self._episode_count
        self._metrics["step_distribution"].append(self._step_count)

        # Auto-escalate difficulty
        avg = sum(self._episode_scores) / len(self._episode_scores)
        if avg > 0.85 and self._difficulty_tier < DifficultyTier.TIER_3:
            self._difficulty_tier = DifficultyTier.TIER_3
        elif avg > 0.70 and self._difficulty_tier < DifficultyTier.TIER_2:
            self._difficulty_tier = DifficultyTier.TIER_2

        self._metrics["difficulty_tier"] = self._difficulty_tier

    def set_difficulty_tier(self, tier: int) -> None:
        """Manually set difficulty tier (for testing via /difficulty endpoint)."""
        if tier in (DifficultyTier.TIER_1, DifficultyTier.TIER_2, DifficultyTier.TIER_3):
            self._difficulty_tier = tier
            self._metrics["difficulty_tier"] = tier

    def _select_scenario(self, task_id: str) -> str:
        """
        Select attack scenario based on task and current difficulty tier.
        Tier 1 uses the original defaults. Higher tiers unlock additional
        scenarios so all 7 scenario seeds see actual gameplay.
        """
        if task_id == "alert_triage":
            return ""  # Task 1 uses mixed alert queue, not a single scenario

        if task_id == "incident_investigation":
            if self._difficulty_tier >= DifficultyTier.TIER_3:
                return "supply_chain_001"
            elif self._difficulty_tier >= DifficultyTier.TIER_2:
                return random.choice(["ransomware_001", "insider_threat_001"])
            return "brute_force_ssh_001"

        if task_id == "threat_response":
            if self._difficulty_tier >= DifficultyTier.TIER_3:
                return "multi_stage_apt_001"
            elif self._difficulty_tier >= DifficultyTier.TIER_2:
                return "ransomware_001"
            return "phishing_lateral_001"

        return ""

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics for the /metrics endpoint."""
        metrics = dict(self._metrics)
        # Compute mean rewards from accumulators
        mean_by_task = {}
        for task_id, data in metrics.get("mean_reward_by_task", {}).items():
            count = data.get("count", 0)
            mean_by_task[task_id] = round(data["total"] / count, 4) if count > 0 else 0.0
        metrics["mean_reward_by_task"] = mean_by_task
        return metrics

    def _build_grader_state(self) -> Dict[str, Any]:
        return {
            "scenario_id": self._active_scenario_id,
            "agent_classifications": self._agent_classifications,
            "agent_queried_log_ids": self._agent_queried_log_ids,
            "agent_queried_sources": self._agent_queried_sources,
            "agent_isolated_assets": self._agent_isolated_assets,
            "agent_blocked_ips": self._agent_blocked_ips,
            "agent_verdict": self._agent_verdict,
            "agent_attack_type": self._agent_attack_type,
            "agent_attacker_ip": self._agent_attacker_ip,
            "agent_report": self._agent_report,
            "rewarded_log_ids": self._rewarded_log_ids,
            "rewarded_sources": self._rewarded_sources,
            "steps_taken": self._step_count,
            "max_steps": TASK_CONFIG[self._task_id]["max_steps"],
        }

    def _build_ground_truth(self) -> Dict[str, Any]:
        scenario_id = self._active_scenario_id
        if scenario_id:
            return ATTACK_SCENARIOS.get(scenario_id, {}).get("ground_truth", {})
        return {}

    def _build_observation(self, message: str = "") -> Observation:
        """Build agent-safe observation (ground truth hidden)."""
        safe_assets = sanitize_for_agent(self._asset_inventory)
        safe_logs = [
            LogEntry(
                log_id=l.log_id, log_source=l.log_source, timestamp=l.timestamp,
                source_ip=l.source_ip, dest_ip=l.dest_ip, user=l.user,
                event_type=l.event_type, details=l.details, is_malicious=False
            )
            for l in self._available_logs
        ]
        ctx = self._task_context
        if ctx:
            ctx = ctx.model_copy(update={"current_step": self._step_count})
        return Observation(
            current_alert=self._current_alert,
            alert_queue=self._alert_queue,
            available_logs=safe_logs,
            asset_inventory=safe_assets,
            open_incidents=self._open_incidents,
            actions_taken=self._actions_taken,
            time_elapsed=self._step_count * 5,
            task_context=ctx,
            message=message,
        )
