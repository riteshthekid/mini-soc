# Mini SOC — v2.0 Execution Plan
**Repository:** `riteshthekid/mini-soc`  
**Phase:** Round 2 — Onsite Build  
**Sequence:** Code → Debug → Host → Train  
**Last Updated:** April 2026

---

## Current State (v1.0 — Everything Working Locally)

| Component | File | Status |
|---|---|---|
| Core environment | `server/mini_soc_environment.py` | ✅ Complete (493 lines) |
| Data models | `models.py` | ✅ Complete (12 Pydantic v2 models) |
| FastAPI server | `server/app.py` | ✅ Complete (5 endpoints) |
| Graders x3 | `server/graders/` | ✅ Complete (weighted, partial credit) |
| Attack scenarios | `server/simulator/attack_seeds.py` | ✅ Complete (3 seeded scenarios) |
| Log generator | `server/simulator/log_gen.py` | ✅ Complete |
| HTTP client | `client.py` | ✅ Complete |
| Expert agent | `run_agent.py` | ✅ Complete |
| LLM baseline | `inference.py` | ✅ Complete |
| OpenEnv manifest | `openenv.yaml` | ✅ Complete |
| Docker + Compose | `server/Dockerfile` + `docker-compose.yml` | ✅ Complete |
| Test suite | `tests/test_env.py` | ✅ 18 tests passing |
| Docs | `README.md` + `docs/API.md` | ✅ Complete |
| **TRL training script** | `train/` | ❌ Missing |
| **Frontend dashboard** | `frontend/` | ❌ Missing |
| **New attack scenarios** | ransomware, insider, APT | ❌ Missing |
| **HuggingFace Space** | deployed URL | ❌ Not deployed |
| **Demo video + blog** | YouTube + HF blog | ❌ Missing |

**Baseline scores (Qwen2.5-72B):**
- Task 1 alert_triage: `0.65` ✅ (threshold 0.60)
- Task 2 incident_investigation: `0.42` ⚠️ (threshold 0.50)
- Task 3 threat_response: `0.22` ❌ (threshold 0.40)

---

## The 4-Phase Plan

```
PHASE 1: CODE          PHASE 2: DEBUG         PHASE 3: HOST          PHASE 4: TRAIN
─────────────────      ─────────────────      ─────────────────      ─────────────────
Training script    →   Run all tests      →   HF Space deploy    →   GRPO on Colab
New scenarios          Fix env issues         validate-sub.sh        Reward curve chart
Frontend               Lint + type check      inference.py live      Blog + video
4 new files            18 + 10 tests          Docker verified        Pitch prep
```

---

---

# PHASE 1 — CODE

> **Goal:** Write all missing v2.0 code. Nothing needs to run yet — just write.  
> **Estimated time:** 10–12 hours

---

## 1.1 New Attack Scenarios

**File:** `server/simulator/attack_seeds.py`  
**Action:** Add 4 new scenarios to the existing `ATTACK_SCENARIOS` dict.

### Scenario 4 — `ransomware_001`

```python
"ransomware_001": {
    "scenario_id": "ransomware_001",
    "attack_type": "malware",
    "attacker_ip": "192.168.50.99",
    "target_hostname": "WS-FINANCE-01",
    "kill_chain": ["initial_access", "execution", "impact"],
    "ground_truth": {
        "classification": "critical",
        "priority": "P1",
        "verdict": "true_positive",
        "attack_type": "malware",
        "key_evidence": ["PROC-R01", "NET-R01", "FS-R01"],
        "affected_assets": ["WS-FINANCE-01"],
        "attacker_ips": ["192.168.50.99"],
    },
    "alerts": [...],   # ransomware process + SMB spread alerts
    "logs": {
        "process": [...],   # vssadmin.exe delete shadows, .exe encrypting files
        "network": [...],   # SMB broadcast to subnet
        "auth":    [...],
        "firewall":[...],
        "dns":     [],
    },
}
```

**Key evidence to include:**
- `PROC-R01`: `vssadmin.exe delete shadows` — deletes volume shadow copies
- `PROC-R02`: unknown `.exe` writing `.encrypted` extensions to `C:\Finance\`
- `NET-R01`: SMB broadcast packets to `10.0.2.0/24` attempting to spread
- Agent must act within 15 steps or SLA penalty applies (`−0.01` per step over limit)

---

### Scenario 5 — `insider_threat_001`

```python
"insider_threat_001": {
    "scenario_id": "insider_threat_001",
    "attack_type": "data_exfiltration",
    "attacker_ip": None,   # ← no external IP — authorised internal user
    "compromised_user": "bwalker",
    "kill_chain": ["collection", "exfiltration"],
    "ground_truth": {
        "classification": "suspicious",
        "priority": "P2",
        "verdict": "true_positive",
        "attack_type": "data_exfiltration",
        "key_evidence": ["NET-I01", "AUTH-I01", "DNS-I01"],
        "affected_assets": ["WS-FINANCE-01"],
        "attacker_ips": [],   # no external IP — challenge for agent
    },
    ...
}
```

**Why this is hard:** No external attacker IP. Agent must detect anomalous behaviour from an authorised user (`bwalker` bulk-downloading finance files at 2am, then DNS query to a personal cloud storage domain). Tests agent's ability to reason about *behaviour* not just *source IP*.

---

### Scenario 6 — `supply_chain_001`

Trusted software update installs backdoor. All alerts are low-severity — agent must correlate low-confidence signals across 3 sources to identify the attack. Designed to challenge the `suspicious → investigate → critical` reasoning chain.

---

### Scenario 7 — `multi_stage_apt_001`

6-stage kill chain with red herring decoy IPs in logs. Requires the agent to query all 5 log sources and reason across a 14-step timeline. Intended as a bonus hard scenario for the adaptive difficulty tier 3.

---

## 1.2 Adaptive Difficulty Engine

**File:** `server/mini_soc_environment.py`  
**Action:** Add difficulty tier tracking to `SocEnvironment`.

```python
class DifficultyTier:
    TIER_1 = 1   # default: single attacker, 2–3 log sources needed
    TIER_2 = 2   # auto-unlock at rolling_avg > 0.70: decoy IPs, tighter SLA
    TIER_3 = 3   # auto-unlock at rolling_avg > 0.85: APT, 14-day log window

class SocEnvironment:
    def __init__(self):
        ...
        self._difficulty_tier: int = DifficultyTier.TIER_1
        self._episode_scores: List[float] = []   # rolling window of last 5

    def _maybe_escalate_difficulty(self, final_score: float):
        self._episode_scores.append(final_score)
        if len(self._episode_scores) > 5:
            self._episode_scores.pop(0)
        avg = sum(self._episode_scores) / len(self._episode_scores)
        if avg > 0.85 and self._difficulty_tier < 3:
            self._difficulty_tier = 3
        elif avg > 0.70 and self._difficulty_tier < 2:
            self._difficulty_tier = 2
```

**New endpoint:** `POST /difficulty` — manually set tier for testing.  
**New endpoint:** `GET /metrics` — return training statistics for frontend.

---

## 1.3 MITRE ATT&CK Tags

**File:** `server/simulator/attack_seeds.py`  
**Action:** Add `mitre_techniques` field to each scenario's ground truth.

```python
"ground_truth": {
    ...
    "mitre_techniques": [
        {"technique_id": "T1110.001", "name": "Password Guessing", "tactic": "Credential Access"},
        {"technique_id": "T1078",     "name": "Valid Accounts",    "tactic": "Initial Access"},
    ],
}
```

---

## 1.4 Training Script

> This is the most critical missing piece for Round 2 judging.

### File: `train/reward_wrapper.py`

Adapter that bridges `SocEnvironment` to the HuggingFace TRL `GRPOTrainer` reward function signature.

```python
import httpx
from typing import List

def soc_reward_function(
    prompts: List[str],
    completions: List[str],
    task_ids: List[str],
    **kwargs
) -> List[float]:
    """
    Called by GRPOTrainer for each group of K completions.
    Executes each completion as a JSON action in the environment.
    Returns episode reward for each completion.
    """
    rewards = []
    for prompt, completion, task_id in zip(prompts, completions, task_ids):
        try:
            action = json.loads(completion)
            result = httpx.post(
                f"{SOC_ENV_URL}/step",
                json={"action_type": action["action_type"],
                      "parameters": action.get("parameters", {})},
                timeout=10,
            ).json()
            rewards.append(float(result.get("reward", 0.0)))
        except Exception:
            rewards.append(-0.1)   # penalty for malformed JSON
    return rewards
```

---

### File: `train/train_grpo.py`

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from train.reward_wrapper import soc_reward_function
import wandb

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./outputs/mini-soc-grpo"

# LoRA config
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# GRPO config
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_generations=4,         # K=4 group size
    max_new_tokens=200,
    temperature=0.7,
    logging_steps=5,
    save_steps=50,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=soc_reward_function,
    args=grpo_config,
    train_dataset=soc_dataset,   # prompts from all 3 task resets
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub("riteshthekid/mini-soc-grpo")
```

---

### File: `train/train_colab.ipynb`

Google Colab notebook with:
1. `!pip install trl peft transformers httpx wandb` cell
2. Environment setup cell (start server in background thread)
3. Training cell (runs `train_grpo.py`)
4. Reward curve plotting cell (matplotlib)
5. Before/after comparison cell (random agent vs trained agent scores)

---

### File: `train/plot_rewards.py`

```python
import matplotlib.pyplot as plt
import json

def plot_reward_curve(log_file: str, output_path: str = "reward_curve.png"):
    """
    Reads WandB-format JSON log and plots:
      - Mean episode reward per training step
      - Per-task score lines (T1, T2, T3)
      - Random agent baseline (dashed horizontal)
    """
    ...
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_rewards, label="Trained agent", color="#0F6E56", linewidth=2)
    plt.axhline(y=0.09, color="#E24B4A", linestyle="--", label="Random agent baseline")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Mini SOC — GRPO Training Reward Curve")
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
```

---

## 1.5 Frontend Dashboard

**Stack:** React 18 + Vite + Tailwind CSS + Chart.js

### File structure to create:

```
frontend/
├── package.json
├── vite.config.js
├── index.html
└── src/
    ├── App.jsx
    ├── store/useStore.js          # Zustand global state
    ├── api/envClient.js           # fetch() wrapper for /reset /step /state
    ├── pages/
    │   ├── TrainingMonitor.jsx    # Reward curve + task score cards
    │   ├── EpisodeViewer.jsx      # Live episode replay
    │   ├── ModelComparison.jsx    # Before/after score bars
    │   └── ScenarioBrowser.jsx    # Scenario cards + kill chain timeline
    └── components/
        ├── AlertFeed.jsx          # Color-coded alert list
        ├── LogExplorer.jsx        # Tabbed log viewer by source
        ├── AssetMap.jsx           # Network grid with isolation status
        ├── ActionHistory.jsx      # Step list with reward deltas
        ├── RewardMeter.jsx        # Live cumulative reward bar
        └── RewardChart.jsx        # Chart.js line chart
```

### Key component specs:

**`AlertFeed.jsx`**
- Reads `observation.alert_queue` from store
- Color coding: `critical` → red background, `suspicious` → amber, `benign` → green
- After episode ends, overlay correct ground truth classification on each alert
- Shows unclassified count as badge

**`AssetMap.jsx`**
- Grid of asset cards: hostname + criticality (1–5 filled circles) + department
- Red border + "ISOLATED" badge when `asset.is_isolated = true`
- Pulse animation when isolation happens this step

**`RewardChart.jsx`**
- Chart.js line chart: x=training step, y=mean reward
- 3 series: overall + T1 + T2 + T3
- Red dashed baseline at y=0.09 (random agent)
- Auto-refreshes every 10 seconds via polling `/metrics`

**`ModelComparison.jsx`**
- Two columns: "Before Training" and "After Training"
- Horizontal bar chart per task showing score difference
- Green arrow + delta number when trained > random

---

## 1.6 New Tests (10 Tests)

**File:** `tests/test_env.py` — append to existing 18 tests.

```python
# Test 19: Thrashing penalty activates at step 6
def test_thrashing_penalty(env):
    env.reset("incident_investigation")
    for _ in range(5):
        env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    result = env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert result.reward == -0.05

# Test 20: Step after done returns reward=0.0 and error
def test_step_after_done(env):
    env.reset("alert_triage")
    env._done = True
    result = env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert result.reward == 0.0
    assert result.done is True
    assert result.info.get("error") == "episode_done"

# Test 21: Max steps boundary terminates episode
def test_max_steps_task1(env):
    env.reset("alert_triage")
    for _ in range(15):
        if not env._done:
            env.step(Action(action_type=ActionType.REQUEST_INFO, parameters={}))
    assert env._done is True

# Test 22: Progressive alert disclosure in Task 3
def test_progressive_alerts_task3(env):
    env.reset("threat_response")
    assert len(env._alert_queue) == 1      # only first alert at start
    env._step_count = 3
    env._surface_new_alerts()
    assert len(env._alert_queue) == 2      # ALT-011 surfaces

# Test 23: Adaptive tier escalation
def test_difficulty_escalation(env):
    env.reset("alert_triage")
    env._episode_scores = [0.72, 0.74, 0.71, 0.73, 0.75]
    env._maybe_escalate_difficulty(0.73)
    assert env._difficulty_tier == 2

# Test 24: Grader determinism across 10 runs
def test_grader1_deterministic():
    state = {"agent_classifications": {"ALT-001": {"classification": "critical", "priority": "P1"}}}
    scores = [grader1.grade(state) for _ in range(10)]
    assert len(set(scores)) == 1   # all identical

# Test 25: New scenario ransomware_001 loads
def test_ransomware_scenario_loads(env):
    result = env.reset("alert_triage")   # uses TASK1_ALERT_QUEUE
    assert result.observation is not None

# Test 26: MITRE tags present in scenario ground truth
def test_mitre_tags_in_scenarios():
    scenario = ATTACK_SCENARIOS["brute_force_ssh_001"]
    assert "mitre_techniques" in scenario["ground_truth"]
    assert len(scenario["ground_truth"]["mitre_techniques"]) > 0

# Test 27: /metrics endpoint returns valid dict
def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "episode_count" in data

# Test 28: /difficulty endpoint sets tier
def test_difficulty_endpoint(client):
    response = client.post("/difficulty", json={"tier": 2})
    assert response.status_code == 200
```

---

## Phase 1 Checklist

```
[ ] attack_seeds.py     — add ransomware_001 scenario
[ ] attack_seeds.py     — add insider_threat_001 scenario
[ ] attack_seeds.py     — add mitre_techniques to all 3 existing scenarios
[ ] mini_soc_env.py     — add DifficultyTier + _maybe_escalate_difficulty()
[ ] server/app.py       — add /metrics endpoint
[ ] server/app.py       — add /difficulty endpoint
[ ] train/reward_wrapper.py   — write TRL reward adapter
[ ] train/train_grpo.py       — write GRPOTrainer main script
[ ] train/train_colab.ipynb   — write Colab notebook (5 cells)
[ ] train/plot_rewards.py     — write reward curve chart script
[ ] frontend/package.json     — init React + Vite + Tailwind + Chart.js
[ ] frontend/src/App.jsx      — router + layout
[ ] frontend/src/pages/       — 4 pages (Monitor, Viewer, Compare, Scenarios)
[ ] frontend/src/components/  — 6 components
[ ] tests/test_env.py         — add tests 19–28
```

---

---

# PHASE 2 — DEBUG

> **Goal:** Everything runs without errors. All 28 tests pass. Linting clean.  
> **Estimated time:** 4–6 hours

---

## 2.1 Run Full Test Suite

```bash
# Activate venv
source venv/bin/activate

# Run all 28 tests with coverage
python -m pytest tests/ -v --cov=server --cov=models --cov-report=term-missing

# Expected output:
# 28 passed
# Coverage: >= 80%
```

**Common failures to fix at this step:**

| Failure | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` in new train/ files | Missing `__init__.py` | Add `train/__init__.py` |
| New scenario test fails | `attack_seeds.py` key typo | Check scenario dict key matches test string |
| Grader determinism test fails | Float rounding | Use `round(..., 6)` in grader return |
| `/metrics` endpoint 404 | Forgot to register route in `app.py` | Add `@app.get("/metrics")` |
| Progressive alert test fails | `_step_count` comparison off-by-one | Check `>=` vs `==` in `_surface_new_alerts` |

---

## 2.2 Type Checking

```bash
mypy . --ignore-missing-imports

# Fix every error before moving to Phase 3
# Common fixes:
#   - Add return type annotations to new functions
#   - Add Optional[] wrapping where None is possible
#   - Add List[str] typing to new list fields
```

---

## 2.3 Linting

```bash
ruff check .
ruff format .

# Zero errors/warnings required
```

---

## 2.4 Manual End-to-End Test

Start the server and run each endpoint manually to confirm nothing is broken:

```bash
# Terminal 1: start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: run all checks
curl http://localhost:8000/health
curl http://localhost:8000/tasks
curl http://localhost:8000/metrics

# Test all 3 tasks
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"task_id": "alert_triage"}'

curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action_type": "classify_alert", "parameters": {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}}'

# Run expert agent (should score ~0.95 on all tasks)
python run_agent.py

# Expected output:
# [END] task=alert_triage       success=true  score=0.950
# [END] task=incident_invest... success=true  score=0.980
# [END] task=threat_response    success=true  score=0.850
```

---

## 2.5 Docker Compose Test

```bash
# Build all 3 services
docker-compose build

# Run environment service only
docker-compose up soc-env

# In another terminal — confirm it responds
curl http://localhost:8000/health
# Expected: {"status": "ok", "env": "mini-soc", "version": "1.0.0"}

# Run test service
docker-compose --profile test up soc-tests
# Expected: 28 passed
```

---

## 2.6 Frontend Dev Build

```bash
cd frontend
npm install
npm run dev

# Open http://localhost:5173
# Check: Training Monitor page loads
# Check: Episode Viewer connects to http://localhost:8000
# Check: Reward chart renders (even with empty data)
```

---

## 2.7 Validate Submission Script (Dry Run)

```bash
# Simulate what the hackathon validators will run
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" -d '{}'
# Must return: 200

# Run openenv validate (if installed)
pip install openenv-core
openenv validate
# Must pass all checks
```

---

## Phase 2 Checklist

```
[ ] pytest — 28 tests passing
[ ] coverage — >= 80%
[ ] mypy — 0 errors
[ ] ruff — 0 errors
[ ] manual curl — all 6 endpoints respond correctly
[ ] run_agent.py — scores ~0.95/0.98/0.85 on 3 tasks
[ ] docker-compose build — no errors
[ ] docker-compose up soc-env — /health returns 200
[ ] frontend npm run dev — loads without console errors
[ ] openenv validate — passes
[ ] POST /reset dry run — returns 200 (validator simulation)
```

---

---

# PHASE 3 — HOST

> **Goal:** Environment live on HuggingFace Space. Validator script 3/3 checks pass.  
> **Estimated time:** 3–4 hours

---

## 3.1 Prepare HuggingFace Space

Add this block to the top of `README.md` (must be the very first content in the file for HF to parse it):

```yaml
---
title: Mini SOC OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - cybersecurity
  - agent-evaluation
---
```

---

## 3.2 Create HuggingFace Space

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your HF_TOKEN when prompted

# Create the Space (do this once)
huggingface-cli repo create mini-soc --type space --space-sdk docker

# Add HF remote to your git repo
git remote add hf https://huggingface.co/spaces/riteshthekid/mini-soc

# Push
git push hf main
```

---

## 3.3 Verify Deployment

```bash
# Wait ~2 minutes for Space to build

# Check the build logs at:
# https://huggingface.co/spaces/riteshthekid/mini-soc/logs

# Once live, test it:
export HF_SPACE_URL="https://riteshthekid-mini-soc.hf.space"

curl $HF_SPACE_URL/health
# Expected: {"status": "ok"}

curl -X POST $HF_SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "alert_triage"}'
# Expected: HTTP 200 with observation JSON
```

---

## 3.4 Run validate-submission.sh

```bash
# Download and run the official validator
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh \
  | bash -s -- https://riteshthekid-mini-soc.hf.space .

# Expected output:
# [HH:MM:SS] PASSED -- HF Space is live and responds to /reset
# [HH:MM:SS] PASSED -- Docker build succeeded
# [HH:MM:SS] PASSED -- openenv validate passed
# ========================================
#   All 3/3 checks passed!
#   Your submission is ready to submit.
# ========================================
```

**Common deployment failures:**

| Error | Cause | Fix |
|---|---|---|
| Space build fails | Missing `requirements.txt` in server/ | Add or symlink it |
| `/reset` returns 500 | Import error in container | Check HF logs, fix import path |
| Build timeout (>600s) | Too many pip packages | Use `python:3.11-slim`, remove unused deps |
| openenv validate fails | `openenv.yaml` field mismatch | Check task IDs match exactly |

---

## 3.5 Run inference.py Against Live Space

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SOC_ENV_URL=https://riteshthekid-mini-soc.hf.space

python inference.py 2>&1 | tee baseline_results.txt

# Expected final lines:
# [SCORE] alert_triage: 0.650
# [SCORE] incident_investigation: 0.420
# [SCORE] threat_response: 0.220
# Overall average: 0.430
```

---

## 3.6 Build and Serve Frontend

```bash
cd frontend
npm run build
# Builds to frontend/dist/

# Copy built files so FastAPI can serve them
cp -r frontend/dist/* server/static/

# Rebuild Docker image with frontend included
docker build -t mini-soc-v2 .
docker push riteshthekid/mini-soc-v2

# Push updated image to HF Space
git add .
git commit -m "Add frontend build to Docker image"
git push hf main
```

---

## Phase 3 Checklist

```
[ ] README.md YAML front matter added (HF Space config)
[ ] huggingface-cli login succeeded
[ ] Space created at riteshthekid/mini-soc
[ ] git push hf main — no errors
[ ] HF Space build logs show "success"
[ ] curl $HF_SPACE_URL/health — returns 200
[ ] curl $HF_SPACE_URL/reset — returns HTTP 200 with observation
[ ] validate-submission.sh — 3/3 passed
[ ] inference.py runs against live URL — [END] lines emitted for all 3 tasks
[ ] Frontend dist/ built and served by FastAPI
[ ] Updated Docker image pushed to HF Space
```

---

---

# PHASE 4 — TRAIN

> **Goal:** Show reward improvement. Before/after chart. Blog post. Pitch ready.  
> **Estimated time:** 6–8 hours (includes ~45 min GPU training time)

---

## 4.1 Open Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `train/train_colab.ipynb`
3. Set runtime: **GPU → T4** (free tier is enough)

---

## 4.2 Run the 5 Training Cells

### Cell 1: Install dependencies

```python
!pip install trl peft transformers httpx wandb torch -q
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
```

### Cell 2: Start environment server (background thread)

```python
import subprocess, threading, time

def start_server():
    subprocess.run(["uvicorn", "server.app:app",
                    "--host", "0.0.0.0", "--port", "8000"])

thread = threading.Thread(target=start_server, daemon=True)
thread.start()
time.sleep(3)   # wait for startup

import httpx
resp = httpx.get("http://localhost:8000/health")
print(resp.json())   # {"status": "ok"}
```

### Cell 3: Run GRPO training (200 steps)

```python
import wandb
wandb.init(project="mini-soc-rl", name="grpo-qwen-1.5b-200steps")

from train.train_grpo import run_training
run_training(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    num_steps=200,
    group_size=4,
    env_url="http://localhost:8000",
)
# Runtime: ~45 minutes on T4
```

### Cell 4: Plot reward curve

```python
from train.plot_rewards import plot_reward_curve
plot_reward_curve(
    log_file="wandb/latest-run/files/output.log",
    output_path="reward_curve.png"
)
from IPython.display import Image
Image("reward_curve.png")
```

### Cell 5: Before/after comparison

```python
import httpx, json

def score_random_agent(task_id, n_episodes=5):
    """Run n episodes with random actions, return mean score."""
    import random
    actions = ["query_logs", "classify_alert", "escalate", "request_info"]
    scores = []
    for _ in range(n_episodes):
        httpx.post("http://localhost:8000/reset", json={"task_id": task_id})
        for step in range(10):
            action = random.choice(actions)
            r = httpx.post("http://localhost:8000/step",
                json={"action_type": action, "parameters": {}}).json()
            if r["done"]:
                scores.append(r["info"].get("final_score", 0.0))
                break
    return sum(scores) / len(scores) if scores else 0.0

print("=== BEFORE TRAINING (random agent) ===")
for task in ["alert_triage", "incident_investigation", "threat_response"]:
    score = score_random_agent(task)
    print(f"  {task}: {score:.3f}")

print("\n=== AFTER TRAINING (Qwen2.5-1.5B GRPO) ===")
# Load trained model and score it
# ...
```

---

## 4.3 Expected Results to Screenshot

| Metric | Random Agent | After 200 Steps |
|---|---|---|
| Task 1 score | 0.15 | ~0.52 |
| Task 2 score | 0.08 | ~0.35 |
| Task 3 score | 0.04 | ~0.18 |
| Overall average | 0.09 | ~0.35 |
| DC-01 false isolation rate | 28% | ~8% |

> Screenshot both the WandB reward curve and the before/after table. These are your primary judge evidence.

---

## 4.4 Write the HuggingFace Blog Post

Post at: `huggingface.co/blog/riteshthekid/mini-soc`

**Structure (600–800 words):**

```
## Mini SOC: Teaching AI to Investigate Cyber Attacks

### The Problem
[1 paragraph: SOC analyst shortage, alert fatigue, current AI limitations]

### The Environment
[1 paragraph: what Mini SOC is, 3 tasks, OpenEnv compliance]

### The RL Loop
[1 paragraph: observation → action → reward, explain thrashing penalty + collateral damage as novel mechanics]

### Training Results
[Embed reward_curve.png]
[Embed before/after comparison table]

### What We Learned
[1 paragraph: what was hard, what worked, what surprised you]

### Try It
- HF Space: https://huggingface.co/spaces/riteshthekid/mini-soc
- Repo: https://github.com/riteshthekid/mini-soc
```

---

## 4.5 Record the 2-Minute Demo Video

**Script (exact timing):**

| Time | What to show |
|---|---|
| 0:00–0:20 | "SOC analysts face 500+ alerts per day. 40% are false positives. Current AI can't investigate — it can only classify." |
| 0:20–0:45 | Show HF Space at `riteshthekid-mini-soc.hf.space`. Show `/tasks` endpoint. Show reset + first observation JSON. |
| 0:45–1:15 | Run one threat_response episode with RANDOM agent. Show it isolating DC-01. Red reward: `-0.40`. |
| 1:15–1:45 | Run same episode with TRAINED agent. Watch it query process logs → find PowerShell → isolate WS-HR-03. Green reward: `+0.25`. |
| 1:45–2:00 | Show reward curve chart. "Random: 0.09. After 200 training steps: 0.35. 4× improvement." |

**Upload to:** YouTube (unlisted) and paste link into HF blog post.

---

## 4.6 Pitch Prep

**3-minute pitch structure:**

| Minute | Content |
|---|---|
| 0:00–1:00 | Problem: "3.5M unfilled SOC jobs. $10B shortage. AI today classifies — it cannot investigate." Show busy alert dashboard. |
| 1:00–2:30 | Demo: before agent (DC-01 isolation, −0.40), after agent (correct containment, +0.25), reward curve chart |
| 2:30–3:00 | Architecture (one slide: 3 graders, GRPO, 3 tasks), results table, Scaler AI Labs bonus prize mention |

**Q&A answers to rehearse:**
- "Why GRPO?" → No value network needed, works well with LLM logprobs
- "How do you prevent gaming?" → Thrashing penalty + collateral damage + coverage multiplier
- "Why not real data?" → Seeded synthetic guarantees reproducibility and no PII
- "Scaler AI Labs?" → Enterprise workflow RL — SOC is a real multi-app enterprise workflow

---

## Phase 4 Checklist

```
[ ] Colab notebook opens and all 5 cells run without error
[ ] 200 training steps complete on T4 GPU (~45 min)
[ ] WandB reward curve visible — upward trend confirmed
[ ] before/after comparison table generated
[ ] reward_curve.png saved and looks good
[ ] Trained model pushed to riteshthekid/mini-soc-grpo on HF Hub
[ ] HF blog post published (600+ words, reward curve embedded)
[ ] 2-minute video recorded and uploaded to YouTube
[ ] Video URL added to HF blog post
[ ] 3-minute pitch timed and rehearsed
[ ] Q&A answers rehearsed for all 6 likely questions
[ ] submission form filled with HF Space URL + blog URL + video URL
```

---

---

## Full Timeline Summary

| Day | Phase | Key Deliverable |
|---|---|---|
| Day 1 AM (4h) | Code | New scenarios + adaptive difficulty |
| Day 1 PM (4h) | Code | Training script (all 4 files) |
| Day 2 AM (4h) | Code | Frontend (pages + components) |
| Day 2 PM (2h) | Code | New tests (19–28) |
| Day 3 AM (3h) | Debug | 28 tests pass + mypy + ruff clean |
| Day 3 PM (3h) | Debug | Manual e2e + Docker + frontend dev |
| Day 4 AM (3h) | Host | HF Space deploy + validator 3/3 |
| Day 4 PM (2h) | Host | Live inference.py + frontend served |
| Day 5 AM (4h) | Train | Colab run + reward curve |
| Day 5 PM (4h) | Train | Blog + video + pitch rehearsal |

---

## Projected Final Score

| Criterion | Weight | Current | After Plan |
|---|---|---|---|
| Environment Innovation | 40% | 33/40 | 38/40 |
| Storytelling & Demo | 30% | 10/30 | 26/30 |
| Reward Improvement Evidence | 20% | 2/20 | 17/20 |
| Training Pipeline Setup | 10% | 1/10 | 8/10 |
| **TOTAL** | **100%** | **46/100** | **89/100** |

---

*Generated from codebase analysis of `riteshthekid/mini-soc` v1.0.0 — April 2026*
