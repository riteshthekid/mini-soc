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
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("mini_soc.train")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOC_ENV_URL = os.environ.get("SOC_ENV_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.environ.get("SOC_TIMEOUT", "15"))

# Task IDs available in the environment
TASK_IDS = ["alert_triage", "incident_investigation", "threat_response"]

# System prompt template given to the model
SYSTEM_PROMPT = (
    "You are a SOC analyst AI. You operate a security environment by issuing "
    "JSON actions. Each action must be a JSON object with 'action_type' and "
    "'parameters' keys.\n\n"
    "Available action_types: query_logs, classify_alert, isolate_asset, "
    "block_ip, escalate, write_report, close_incident, request_info.\n\n"
    "Respond with ONLY valid JSON. No explanations."
)


# ---------------------------------------------------------------------------
# Reward function — called by GRPOTrainer
# ---------------------------------------------------------------------------

def soc_reward_function(
    completions: List[str],
    **kwargs: Any,
) -> List[float]:
    """
    Reward function compatible with TRL GRPOTrainer.

    For each completion string:
      1. Parse it as JSON to extract action_type + parameters
      2. Reset the environment for the assigned task
      3. Execute a multi-step episode (up to 10 steps per completion)
      4. Return the cumulative episode reward

    If the completion is malformed JSON, return -0.1 penalty.

    Args:
        completions: List of model-generated completion strings.
        **kwargs: May contain 'prompts' or other GRPOTrainer metadata.

    Returns:
        List of float rewards, one per completion.
    """
    rewards = []
    prompts = kwargs.get("prompts", [])

    for i, completion in enumerate(completions):
        try:
            reward = _run_single_episode(completion, prompts, i)
        except Exception as e:
            logger.warning("Reward computation failed for completion %d: %s", i, e)
            reward = -0.1
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

    Returns the total episode reward.
    """
    client = httpx.Client(timeout=REQUEST_TIMEOUT)

    try:
        # Determine which task to run based on prompt content
        task_id = _extract_task_id(prompts, idx)

        # Reset the environment
        reset_resp = client.post(
            f"{SOC_ENV_URL}/reset",
            json={"task_id": task_id},
        )
        reset_resp.raise_for_status()

        # Parse completion into action(s)
        actions = _parse_completion(completion)
        if not actions:
            return -0.1  # Empty or unparseable

        # Execute actions sequentially
        total_reward = 0.0
        for action in actions:
            action_type = action.get("action_type", "")
            parameters = action.get("parameters", {})

            step_resp = client.post(
                f"{SOC_ENV_URL}/step",
                json={
                    "action_type": action_type,
                    "parameters": parameters,
                },
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            total_reward += float(result.get("reward", 0.0))

            if result.get("done", False):
                # Add final score bonus (weighted)
                final_score = result.get("info", {}).get("final_score", 0.0)
                total_reward += final_score * 0.5
                break

        return round(total_reward, 4)

    except httpx.HTTPError as e:
        logger.warning("HTTP error during episode: %s", e)
        return -0.1
    except json.JSONDecodeError:
        return -0.1
    finally:
        client.close()


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

def build_soc_dataset(num_samples: int = 60):
    """
    Build a HuggingFace Dataset of SOC task prompts for GRPO training.

    Each sample contains a 'prompt' field formatted as a chat message
    with the system prompt and a task-specific user message.

    Args:
        num_samples: Total number of prompt samples (divided across 3 tasks).

    Returns:
        HuggingFace Dataset with 'prompt' column.
    """
    client = httpx.Client(timeout=REQUEST_TIMEOUT)
    prompts = []

    samples_per_task = max(num_samples // len(TASK_IDS), 1)

    for task_id in TASK_IDS:
        for _ in range(samples_per_task):
            try:
                # Reset to get a fresh observation
                reset_resp = client.post(
                    f"{SOC_ENV_URL}/reset",
                    json={"task_id": task_id},
                )
                reset_resp.raise_for_status()
                obs_data = reset_resp.json()

                # Build the prompt from the observation
                prompt = _format_prompt(task_id, obs_data)
                prompts.append({"prompt": prompt})

            except Exception as e:
                logger.warning("Failed to build prompt for %s: %s", task_id, e)
                # Fallback static prompt
                prompts.append({
                    "prompt": _static_prompt(task_id),
                })

    client.close()
    random.shuffle(prompts)

    # Lazy import — only needed when actually building a dataset for training
    from datasets import Dataset
    return Dataset.from_list(prompts)


def _format_prompt(task_id: str, obs_data: Dict[str, Any]) -> str:
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

    prompt = (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[TASK]\nTask: {task_id} (difficulty: {difficulty})\n"
        f"Objective: {objective}\n"
        f"{message}"
        f"{alerts_summary}"
        f"{assets_summary}\n\n"
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
