"""
Mini SOC — TRL Reward Wrapper
==============================
Bridges the Mini SOC environment to HuggingFace TRL's GRPOTrainer.
The GRPOTrainer calls `soc_reward_function` for each group of K completions.
Each completion is parsed as a JSON action and executed in the environment
via HTTP, returning the episode reward.

Usage:
    from train.reward_wrapper import soc_reward_function, build_soc_dataset
"""
from __future__ import annotations

import json
import os
import random
import time
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("mini_soc.train")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOC_ENV_URL = os.environ.get(
    "SOC_ENV_URL",
    "https://riteshp30-mini-soc.hf.space",  # Default to live HF Space for Colab
)
REQUEST_TIMEOUT = float(os.environ.get("SOC_TIMEOUT", "30"))

# Task IDs available in the environment
TASK_IDS = ["alert_triage", "incident_investigation", "threat_response"]

# Reward normalization — raw step rewards range [-0.40, +0.30].
# Scaling improves GRPO convergence.
REWARD_SCALE = 2.5
REWARD_CLIP_MIN = -1.0
REWARD_CLIP_MAX = 1.0

# System prompt template given to the model
SYSTEM_PROMPT = (
    "You are a SOC analyst AI. You operate a security environment by issuing "
    "JSON actions. Each action must be a JSON object with 'action_type' and "
    "'parameters' keys.\n\n"
    "Available action_types: query_logs, classify_alert, isolate_asset, "
    "block_ip, escalate, write_report, close_incident, request_info.\n\n"
    "Respond with ONLY valid JSON. No explanations, no markdown, no backticks.\n"
    "Example: {\"action_type\": \"classify_alert\", \"parameters\": "
    "{\"alert_id\": \"ALT-001\", \"classification\": \"critical\", \"priority\": \"P1\"}}"
)


# ---------------------------------------------------------------------------
# Reward normalization
# ---------------------------------------------------------------------------

def normalize_reward(raw: float) -> float:
    """Scale and clip raw step reward for GRPO stability."""
    scaled = raw * REWARD_SCALE
    return max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, scaled))


# ---------------------------------------------------------------------------
# HTTP helpers with retry logic
# ---------------------------------------------------------------------------

def _request_with_retry(
    method: str,
    url: str,
    payload: Optional[Dict] = None,
    retries: int = 3,
    delay: float = 2.0,
) -> Dict[str, Any]:
    """Make an HTTP request with exponential backoff retry."""
    for attempt in range(retries):
        try:
            if method == "GET":
                r = httpx.get(url, timeout=REQUEST_TIMEOUT)
            else:
                r = httpx.post(url, json=payload or {}, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning("Request failed after %d retries: %s %s → %s", retries, method, url, e)
                raise
            wait = delay * (attempt + 1)
            logger.debug("Request attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, e)
            time.sleep(wait)
    return {}  # unreachable, but satisfies type checker


def _check_env_health() -> bool:
    """Verify the environment server is reachable."""
    try:
        health = _request_with_retry("GET", f"{SOC_ENV_URL}/health", retries=2)
        return health.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reward function — called by GRPOTrainer
# ---------------------------------------------------------------------------

def soc_reward_function(
    prompts,
    completions,
    **kwargs: Any,
) -> List[float]:
    """
    Reward function compatible with TRL GRPOTrainer.

    TRL passes:
      prompts:     list of prompt strings (or chat messages)
      completions: list of completion messages, each is list[dict] like
                   [{"role": "assistant", "content": "..."}]

    For each completion:
      1. Extract the text content from the chat message
      2. Parse it as JSON to extract action_type + parameters
      3. Reset the environment for the assigned task
      4. Execute a multi-step episode
      5. Return the episode reward (normalized)

    Returns:
        List of float rewards, one per completion.
    """
    # Health check — fail fast if env is down
    if not _check_env_health():
        logger.warning("Environment unreachable at %s — returning fallback rewards", SOC_ENV_URL)
        # Return VARIED penalties to avoid zero-gradient issue
        return [-(0.05 + 0.02 * (i % 5)) for i in range(len(completions))]

    rewards = []
    prompt_list = prompts if isinstance(prompts, list) else []

    for i, completion in enumerate(completions):
        try:
            # Extract text from TRL's completion format
            # TRL passes completions as list[dict] like [{"role": "assistant", "content": "..."}]
            if isinstance(completion, list) and len(completion) > 0:
                # Chat message format: [{"role": "assistant", "content": "..."}]
                text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
            elif isinstance(completion, dict):
                text = completion.get("content", str(completion))
            elif isinstance(completion, str):
                text = completion
            else:
                text = str(completion)

            reward = _run_single_episode(text, prompt_list, i)
        except Exception as e:
            logger.warning("Reward computation failed for completion %d: %s", i, e)
            # Varied penalties to maintain gradient signal
            reward = -(0.05 + 0.03 * (i % 4))
        rewards.append(reward)

    return rewards


def _run_single_episode(
    completion: str,
    prompts: List[str],
    idx: int,
) -> float:
    """
    Execute a single episode from one model completion.

    Supports two formats:
      1. Single action: {"action_type": "...", "parameters": {...}}
      2. Multi-step plan: [{"action_type": "...", ...}, ...]

    Returns the total episode reward (normalized).
    """
    # Determine which task to run based on prompt content
    task_id = _extract_task_id(prompts, idx)

    # Reset the environment
    _request_with_retry("POST", f"{SOC_ENV_URL}/reset", {"task_id": task_id})

    # Parse completion into action(s)
    actions = _parse_completion(completion)
    if not actions:
        return -0.1  # Empty or unparseable

    # Execute actions sequentially
    total_reward = 0.0
    for action in actions:
        action_type = action.get("action_type", "")
        parameters = action.get("parameters", {})

        try:
            result = _request_with_retry(
                "POST",
                f"{SOC_ENV_URL}/step",
                {"action_type": action_type, "parameters": parameters},
            )
        except Exception:
            # Connection failed mid-episode — return what we have
            return total_reward if total_reward != 0.0 else -0.1

        step_reward = float(result.get("reward", 0.0))
        total_reward += normalize_reward(step_reward)

        if result.get("done", False):
            # Add final score bonus (weighted)
            final_score = result.get("info", {}).get("final_score", 0.0)
            total_reward += final_score * 0.5
            break

    return round(total_reward, 4)


def _parse_completion(completion: str) -> List[Dict[str, Any]]:
    """
    Parse a model completion string into a list of actions.

    Handles:
      - Clean JSON object: {"action_type": "...", "parameters": {...}}
      - JSON array: [{"action_type": "...", ...}, ...]
      - Markdown-wrapped JSON: ```json ... ```
      - Multiple JSON objects on separate lines
    """
    text = completion.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try parsing as a single JSON object or array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, list):
            return [a for a in parsed if isinstance(a, dict)]
    except json.JSONDecodeError:
        pass

    # Try parsing line-by-line (for multi-line outputs)
    actions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "action_type" in obj:
                actions.append(obj)
        except json.JSONDecodeError:
            continue

    # Last resort: try to find JSON embedded in natural language
    if not actions:
        import re
        # Pattern allows one level of nested braces (for "parameters": {...})
        json_pattern = re.compile(r'\{(?:[^{}]|\{[^{}]*\})*"action_type"(?:[^{}]|\{[^{}]*\})*\}')
        matches = json_pattern.findall(text)
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and "action_type" in obj:
                    actions.append(obj)
                    break  # Take the first valid match
            except json.JSONDecodeError:
                continue

    return actions


def _extract_task_id(prompts: List[str], idx: int) -> str:
    """
    Extract the task_id from the prompt text, or cycle through tasks.
    """
    if idx < len(prompts):
        prompt = prompts[idx].lower()
        for tid in TASK_IDS:
            if tid.replace("_", " ") in prompt or tid in prompt:
                return tid

    # Round-robin fallback
    return TASK_IDS[idx % len(TASK_IDS)]


# ---------------------------------------------------------------------------
# Dataset builder — creates the prompt dataset for GRPOTrainer
# ---------------------------------------------------------------------------

# Warmup actions used to diversify starting observations
_WARMUP_ACTIONS = {
    "alert_triage": [
        {"action_type": "request_info", "parameters": {}},
    ],
    "incident_investigation": [
        {"action_type": "query_logs", "parameters": {"log_source": "auth"}},
        {"action_type": "query_logs", "parameters": {"log_source": "firewall"}},
        {"action_type": "query_logs", "parameters": {"log_source": "dns"}},
        {"action_type": "query_logs", "parameters": {"log_source": "process"}},
        {"action_type": "query_logs", "parameters": {"log_source": "network"}},
    ],
    "threat_response": [
        {"action_type": "query_logs", "parameters": {"log_source": "process"}},
        {"action_type": "query_logs", "parameters": {"log_source": "network"}},
        {"action_type": "query_logs", "parameters": {"log_source": "auth"}},
        {"action_type": "request_info", "parameters": {}},
    ],
}


def build_soc_dataset(num_samples: int = 60):
    """
    Build a HuggingFace Dataset of SOC task prompts for GRPO training.

    Each sample contains a 'prompt' field formatted as a chat message
    with the system prompt and a task-specific user message.

    To prevent overfitting on identical observations, each prompt is
    generated from a slightly different starting state by executing
    0–3 warm-up actions before capturing the observation.

    Args:
        num_samples: Total number of prompt samples (divided across 3 tasks).

    Returns:
        HuggingFace Dataset with 'prompt' column.
    """
    prompts = []
    samples_per_task = max(num_samples // len(TASK_IDS), 1)

    for task_id in TASK_IDS:
        for i in range(samples_per_task):
            try:
                # Reset to get a fresh observation
                obs_data = _request_with_retry(
                    "POST",
                    f"{SOC_ENV_URL}/reset",
                    {"task_id": task_id},
                )

                # Take 0–3 warmup actions to diversify the observation state
                n_warmup = i % 4  # cycle: 0, 1, 2, 3, 0, 1, ...
                warmup_pool = _WARMUP_ACTIONS.get(task_id, [])
                for w in range(n_warmup):
                    if not warmup_pool:
                        break
                    warmup_action = warmup_pool[w % len(warmup_pool)]
                    try:
                        step_result = _request_with_retry(
                            "POST",
                            f"{SOC_ENV_URL}/step",
                            warmup_action,
                        )
                        if step_result.get("done", False):
                            break
                        # Use the post-warmup observation
                        obs_data = step_result
                    except Exception:
                        break

                # Build the prompt from the (possibly advanced) observation
                user_msg = _format_prompt(task_id, obs_data, step=n_warmup)
                # TRL GRPOTrainer expects prompts as chat messages
                chat_prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                prompts.append({"prompt": chat_prompt})

            except Exception as e:
                logger.warning("Failed to build prompt for %s: %s", task_id, e)
                # Fallback static prompt in chat format
                prompts.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _static_prompt(task_id)},
                    ],
                })

    random.shuffle(prompts)

    # Lazy import — only needed when actually building a dataset for training
    from datasets import Dataset
    return Dataset.from_list(prompts)


def _format_prompt(task_id: str, obs_data: Dict[str, Any], step: int = 0) -> str:
    """Format a training prompt from a reset observation."""
    obs = obs_data.get("observation", {})
    message = obs.get("message", "")
    task_ctx = obs.get("task_context", {})
    objective = task_ctx.get("objective", "")
    difficulty = task_ctx.get("difficulty", "")

    # Include alert info for context
    alerts_summary = ""
    alert_queue = obs.get("alert_queue", [])
    if alert_queue:
        alert_lines = []
        for a in alert_queue[:5]:  # Limit to 5 for prompt length
            alert_lines.append(
                f"  - {a.get('alert_id', '?')}: {a.get('alert_type', '?')} "
                f"(severity={a.get('severity', '?')})"
            )
        alerts_summary = "\nAlerts:\n" + "\n".join(alert_lines)

    # Include asset info
    assets_summary = ""
    assets = obs.get("asset_inventory", [])
    if assets:
        asset_lines = []
        for a in assets[:5]:
            asset_lines.append(
                f"  - {a.get('hostname', '?')} ({a.get('asset_type', '?')}, "
                f"criticality={a.get('criticality', '?')})"
            )
        assets_summary = "\nAssets:\n" + "\n".join(asset_lines)

    # Include available logs (if warmup produced some)
    logs_summary = ""
    logs = obs.get("available_logs", [])
    if logs:
        log_lines = []
        for l in logs[:5]:
            log_lines.append(
                f"  - [{l.get('log_source', '?')}] {l.get('timestamp', '')[:16]} "
                f"{l.get('event_type', '?')} src={l.get('source_ip', '?')}"
            )
        logs_summary = "\nRetrieved Logs:\n" + "\n".join(log_lines)

    # Include open incidents
    incidents_summary = ""
    incidents = obs.get("open_incidents", [])
    if incidents:
        inc_lines = [f"  - {inc.get('incident_id', '?')}: {inc.get('status', '?')}" for inc in incidents]
        incidents_summary = "\nOpen Incidents:\n" + "\n".join(inc_lines)

    prompt = (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[TASK]\nTask: {task_id} (difficulty: {difficulty})\n"
        f"Objective: {objective}\n"
        f"Step: {step}\n"
        f"{message}"
        f"{alerts_summary}"
        f"{assets_summary}"
        f"{logs_summary}"
        f"{incidents_summary}\n\n"
        f"[RESPOND]\nIssue your next action as a JSON object:"
    )
    return prompt


def _static_prompt(task_id: str) -> str:
    """Fallback static prompt when environment is unavailable."""
    objectives = {
        "alert_triage": (
            "Classify all 10 alerts in the queue as benign/suspicious/critical "
            "and assign correct priority (P1-P4)."
        ),
        "incident_investigation": (
            "Investigate the active incident: query relevant log sources, "
            "identify the attacker IP, and submit a verdict with attack type."
        ),
        "threat_response": (
            "A multi-stage attack is active. Gather evidence, isolate compromised "
            "assets, block attacker IPs, and write a full incident report."
        ),
    }
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[TASK]\nTask: {task_id}\n"
        f"Objective: {objectives.get(task_id, 'Complete the task.')}\n\n"
        f"[RESPOND]\nIssue your next action as a JSON object:"
    )
