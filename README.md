---
title: Mini SOC
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
license: mit
tags:
  - openenv
  - cybersecurity
  - reinforcement-learning
  - grpo
  - soc-analyst
---

# 🛡️ Mini SOC — AI Security Analyst RL Environment

> Train AI agents to perform real SOC (Security Operations Center) work: triaging alerts, investigating incidents, and responding to active threats.

**🔗 Links:**
- 🌐 **Live Environment:** [https://huggingface.co/spaces/riteshp30/mini-soc](https://huggingface.co/spaces/riteshp30/mini-soc)
- 📓 **Training Notebook (Colab):** [Open in Google Colab](https://colab.research.google.com/drive/1dszsl5Z50asbUKmtfdfjNqqPsmNFgeP5?usp=sharing)
- 📊 **Training Results:** [See Results Section](#training-results--grpo)
- 📝 **Blog Post:** [The Alert Storm: How Mini SOC is Training AI to Think Like a Cyber Defender](./blog.md)

Built on the [OpenEnv](https://github.com/pytorch/openenv) framework.

---

## Why This Environment?

SOC analyst shortage is a **$10B+ industry problem**. Security teams are overwhelmed with alert volume — analysts spend 40% of their time on false positives. This environment trains and evaluates AI agents on the exact multi-step decision-making SOC analysts perform daily:

- 🔍 Reading and classifying security alerts (easy → hard)
- 📋 Querying forensic log sources across 5 categories
- 🔗 Correlating events to identify attack kill chains
- 🛑 Containing threats by isolating assets and blocking IPs
- 📝 Writing structured incident reports with MITRE ATT&CK tags

**What makes this challenging:** The agent must balance thoroughness with efficiency, avoid collateral damage (isolating healthy servers = severe penalty), and handle evolving attack scenarios with adaptive difficulty.

---

## Tasks

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `alert_triage` | 🟢 Easy | 15 | Classify 10 alerts as benign/suspicious/critical with priority |
| `incident_investigation` | 🟡 Medium | 20 | Query logs, identify attacker, submit verdict with attack type |
| `threat_response` | 🔴 Hard | 30 | Multi-stage attack: gather evidence, isolate, block, write report |

### Task 1 — Alert Triage (Easy)
The agent receives a queue of 10 security alerts (sampled from a pool of 20) including port scans, brute-force attempts, suspicious logins, and false positives. Must classify each as `benign`, `suspicious`, or `critical`, and assign priority (`P1`–`P4`). Grader uses fuzzy matching for classification aliases.

### Task 2 — Incident Investigation (Medium)
A suspicious alert is open. The agent must query log sources (`auth`, `firewall`, `dns`, `process`, `network`), correlate events, identify the attacker IP, and close the incident with a verdict and attack type. Includes ordered strategy bonus — querying relevant sources first earns extra credit.

### Task 3 — Active Threat Response (Hard)
A multi-stage attack is unfolding in real time. New alerts surface as the episode progresses. The agent must gather evidence, isolate **only** compromised assets (isolating healthy critical assets like `DC-01` triggers severe penalty), block attacker IPs, and write a structured incident report.

---

## Key Features

### 🎯 7 Attack Scenarios with MITRE ATT&CK Tags
| Scenario | Attack Type | Difficulty Tier | MITRE Techniques |
|---|---|---|---|
| `brute_force_ssh_001` | Brute Force | Tier 1 | T1110.001, T1078 |
| `phishing_lateral_001` | Phishing + Lateral Movement | Tier 1 | T1566.002, T1021.001 |
| `false_positive_scan_001` | False Positive | Tier 1 | — |
| `ransomware_001` | Ransomware | Tier 2 | T1486, T1490 |
| `insider_threat_001` | Insider Data Theft | Tier 2 | T1567, T1083 |
| `supply_chain_001` | Supply Chain Compromise | Tier 3 | T1195.002, T1059 |
| `multi_stage_apt_001` | Advanced Persistent Threat | Tier 3 | T1190, T1003 |

### 📈 Adaptive Difficulty
Three-tier system with automatic escalation based on rolling average performance:
- **Tier 1** (default): Basic scenarios
- **Tier 2** (unlocks at avg > 0.70): Ransomware, insider threats
- **Tier 3** (unlocks at avg > 0.85): Supply chain, multi-stage APT

### 🎲 Extended Alert Pool
20 diverse alerts covering DNS tunneling, DLP violations, privilege escalation, lateral movement, and benign noise. 10 randomly sampled per episode to prevent memorization.

### 🏆 Dense Reward Shaping
| Signal | Condition | Value |
|---|---|---|
| Correct classification | `classify_alert` matches ground truth | +0.20 |
| Correct priority | Priority matches ground truth | +0.10 |
| Missing critical as benign | Dangerous misclassification | −0.30 |
| Relevant log query | Source contains key evidence | +0.10–0.15 |
| Strategy bonus | Key sources queried first | +0.08–0.15 |
| Correct asset isolation | Hostname is actually compromised | +0.25 |
| Collateral damage (DC) | Isolating healthy domain controller | −0.40 |
| Correct IP block | Attacker IP blocked | +0.20 |
| Correct verdict | Verdict matches ground truth | +0.30 |
| Report completeness | All required fields present | up to +0.30 |

---

## Training Results — GRPO

We trained a **Qwen2.5-1.5B-Instruct** model using [TRL's GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) with LoRA adapters on a Google Colab T4 GPU.

### Configuration
| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | GRPO + LoRA (rank=16, alpha=32) |
| Training Steps | 200 |
| Trainable Parameters | 4.3M |
| Training Time | ~1h 49min |
| Hardware | Colab T4 (16GB) |

### Training Loss Curve
```
Step  | Loss      | Phase
------|-----------|---------------------------
5     | +0.1112   | Learning JSON format
25    | +0.1786   | Peak exploration
70    | +0.0269   | Best format accuracy
90    | -0.0524   | First negative (refining)
150   | -0.0691   | Consistent refinement
180   | -0.2656   | Strongest correction
200   | -0.0766   | Converging
```

### Evaluation Results
| Task | Random Baseline | GRPO-Trained | Improvement |
|---|---|---|---|
| `alert_triage` | 0.001 | **0.070** | **70×** |
| `incident_investigation` | 0.001 | 0.001 | — |
| `threat_response` | 0.000 | 0.000 | — |
| JSON Success Rate | ~0% | **100%** | ∞ |

**Key achievement:** The model learned to produce **100% valid JSON actions** (from 0%) and achieved measurable score improvement on alert triage. Tasks 2/3 need extended training (500+ steps) for task-specific strategy learning.

### Multi-Level Reward Innovation
Our custom reward function guarantees non-zero GRPO gradients even when all completions are invalid:
```
+0.05 if text contains { and }
+0.05 if text contains "action_type"
+0.03 if text contains "parameters"
+0.10 if parseable as JSON
+0.15 if correct JSON structure
+0.10 if valid action name
```
This prevents the common zero-gradient problem in LLM RL training.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start episode. Body: `{"task_id": "alert_triage"}` |
| `POST` | `/step` | Submit action. Body: `{"action_type": "...", "parameters": {...}}` |
| `GET` | `/state` | Full internal state (includes ground truth) |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Training metrics: episode count, mean reward, difficulty tier |
| `POST` | `/difficulty` | Set difficulty tier (1, 2, or 3) |
| `GET` | `/scenarios` | All attack scenarios with MITRE tags |

---

## Quick Start

### Try the Live Environment

```bash
# Health check
curl https://riteshp30-mini-soc.hf.space/health

# Start an episode
curl -X POST https://riteshp30-mini-soc.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "alert_triage"}'

# Submit an action
curl -X POST https://riteshp30-mini-soc.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify_alert", "parameters": {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}}'
```

### Using Docker

```bash
docker build -t mini-soc .
docker run -p 8000:8000 mini-soc
```

### Using the Python Client

```python
from client import MiniSocEnv

with MiniSocEnv(base_url="https://riteshp30-mini-soc.hf.space") as env:
    result = env.reset(task_id="alert_triage")
    print(result.observation.alert_queue)

    step = env.step("classify_alert", {
        "alert_id": "ALT-001",
        "classification": "critical",
        "priority": "P1",
    })
    print(f"Reward: {step.reward}, Done: {step.done}")
```

### Train with GRPO (Colab)

Open [`train/train_colab.ipynb`](./train/train_colab.ipynb) in Google Colab with a T4 GPU and run all cells. Training takes ~2 hours for 200 steps.

### Run Tests

```bash
python -m pytest tests/ -v  # 54 tests, all passing
```

---

## Observation Space

```python
class Observation(BaseModel):
    current_alert: Optional[Alert]        # Active alert under investigation
    alert_queue: List[Alert]              # Pending alerts
    available_logs: List[LogEntry]        # Logs retrieved this episode
    asset_inventory: List[Asset]          # Network assets (criticality 1–5)
    open_incidents: List[Incident]        # Active incident cases
    actions_taken: List[str]              # Episode action history
    time_elapsed: int                     # Simulated minutes elapsed
    task_context: Optional[TaskContext]   # Task objective and progress
    message: str                          # Human-readable status
```

## Action Space

```python
class Action(BaseModel):
    action_type: Literal[
        "query_logs",      # Retrieve logs from a source
        "classify_alert",  # Label alert severity + priority
        "escalate",        # Escalate to Tier-2
        "isolate_asset",   # Remove asset from network
        "block_ip",        # Block IP at perimeter firewall
        "close_incident",  # Submit final verdict
        "write_report",    # Write structured incident report
        "request_info",    # Ask for clarification
    ]
    parameters: Dict[str, Any]
```

---

## Project Structure

```
mini-soc/
├── models.py                 # Pydantic data models (Action, Observation, etc.)
├── client.py                 # MiniSocEnv(EnvClient) — agent-facing client
├── inference.py              # LLM baseline evaluation
├── run_agent.py              # Multi-task agent runner
├── Dockerfile                # Production container
├── server/
│   ├── app.py                # FastAPI app (8 endpoints)
│   ├── mini_soc_environment.py  # Core environment + adaptive difficulty
│   ├── simulator/
│   │   ├── attack_seeds.py   # 7 scenarios + 20-alert pool
│   │   └── log_gen.py        # Synthetic log generator
│   └── graders/
│       ├── grader1.py        # Alert Triage scorer (fuzzy matching)
│       ├── grader2.py        # Investigation scorer (strategy bonus)
│       └── grader3.py        # Threat Response scorer
├── train/
│   ├── train_grpo.py         # GRPO training script (TRL + Unsloth)
│   ├── reward_wrapper.py     # Multi-level reward function
│   ├── train_colab.ipynb     # 5-cell Colab notebook
│   └── plot_rewards.py       # Training visualization
└── tests/
    └── test_env.py           # 54 comprehensive tests
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SOC_ENV_URL` | No | `http://localhost:8000` | Environment server URL |
| `HF_TOKEN` | No | — | HuggingFace token (for model downloads) |
| `WANDB_API_KEY` | No | — | Weights & Biases key (optional tracking) |

---

## License

MIT
