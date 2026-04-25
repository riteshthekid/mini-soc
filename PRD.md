# Product Requirements Document (PRD) — Mini SOC v2.0.0

## 1. Executive Summary

**Product Name:** Mini Security Operations Center (Mini SOC)
**Version:** 2.0.0
**License:** MIT
**Target Platforms:** PyTorch OpenEnv Framework, Hugging Face Spaces (Docker SDK), Standalone Docker
**Repository:** `github.com/riteshthekid/mini-soc`

### 1.1 Objective

Mini SOC is a fully deterministic Reinforcement Learning (RL) environment that trains and evaluates Large Language Models (LLMs) as autonomous Tier-1 SOC analysts. It simulates the complete multi-step decision-making workflow of real-world cybersecurity operations: triaging security alerts, conducting forensic log investigations, containing active threats, and writing structured incident reports.

### 1.2 Problem Statement

SOC analyst shortage is a **$10B+ industry problem**. Analysts spend ~40% of their time on false positives. This environment provides a standardized benchmark for measuring an LLM's ability to perform the exact sequential reasoning that human SOC analysts execute daily.

### 1.3 Target Audience

- **AI/ML Researchers:** Training RL agents (specifically via GRPO) on complex, sequential cybersecurity decision-making tasks.
- **Cybersecurity Engineers:** Evaluating the forensic reasoning and threat response capabilities of foundation models.
- **OpenEnv Ecosystem:** Providing a submission-ready environment for the PyTorch OpenEnv framework.

---

## 2. System Architecture

### 2.1 High-Level Design

The system follows a **decoupled client-server architecture** with strict separation between the environment backend (state machine + grading) and the agent-facing SDK.

```
┌─────────────────────┐         HTTP/REST          ┌──────────────────────────┐
│   Agent / Client    │ ◄──────────────────────►   │   FastAPI Server         │
│   (client.py)       │   /reset /step /state      │   (server/app.py)        │
│   (inference.py)    │   /tasks /health            │                          │
│   (run_agent.py)    │   /metrics /scenarios       │   ┌────────────────────┐ │
│   (train/)          │   /difficulty               │   │ SocEnvironment     │ │
└─────────────────────┘                             │   │ (mini_soc_env.py)  │ │
                                                    │   └────────┬───────────┘ │
                                                    │            │             │
                                                    │   ┌────────▼───────────┐ │
                                                    │   │ Simulator          │ │
                                                    │   │ • attack_seeds.py  │ │
                                                    │   │ • log_gen.py       │ │
                                                    │   └────────┬───────────┘ │
                                                    │            │             │
                                                    │   ┌────────▼───────────┐ │
                                                    │   │ Grading Engine     │ │
                                                    │   │ • grader1.py       │ │
                                                    │   │ • grader2.py       │ │
                                                    │   │ • grader3.py       │ │
                                                    │   └────────────────────┘ │
                                                    └──────────────────────────┘
```

### 2.2 Backend Server (`server/`)

| Component | File | Responsibility |
|---|---|---|
| **FastAPI Engine** | `server/app.py` | HTTP server with factory pattern (`create_app()`). Exposes 9 REST endpoints. CORS-enabled. Global exception handler. |
| **Environment Core** | `server/mini_soc_environment.py` | Stateful `SocEnvironment` class (602 LOC). Manages episodes, step counts, action processing, difficulty escalation, and grader invocation. |
| **Log Generator** | `server/simulator/log_gen.py` | Retrieves scenario-specific evidence logs. Injects 2–3 benign noise entries per query. Sanitizes `is_compromised` ground truth from agent observations. |
| **Attack Seeds** | `server/simulator/attack_seeds.py` | 7 deterministic attack scenarios (893 LOC), 10-alert triage queue, 7-asset network inventory. All ground truth hardcoded for reproducibility. |
| **Grading Engine** | `server/graders/grader[1-3].py` | Three task-specific scoring modules with both final `grade()` and dense `compute_step_reward()` functions. |

### 2.3 Client Interface & SDK

| Component | File | Responsibility |
|---|---|---|
| **Python SDK** | `client.py` | `MiniSocEnv` class wrapping REST API with `httpx`. Context manager support. Methods: `reset()`, `step()`, `state()`, `health()`, `tasks()`. |
| **Data Schemas** | `models.py` | 15 Pydantic v2 models: `Alert`, `LogEntry`, `Asset`, `Incident`, `TaskContext`, `Observation`, `Action`, `ActionType`, `Reward`, `StepResult`, `ResetResult`, `StateResult`, plus enums (`AlertSeverity`, `AlertClassification`, `AlertPriority`, `AttackType`). |

### 2.4 Agent Scripts

| Script | Purpose |
|---|---|
| `run_agent.py` | Deterministic expert agent (350 LOC). Hardcoded perfect-play strategies for all 3 tasks. Used for Docker Compose testing. |
| `inference.py` | LLM-powered baseline agent (322 LOC). Uses OpenAI-compatible API with task-specific system prompts. Produces OpenEnv-formatted STDOUT logs. |

---

## 3. API Specification

### 3.1 Endpoints

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/` | — | `{name, version, endpoints, openenv_spec}` | Root metadata |
| `GET` | `/health` | — | `{status, env, version}` | Health check |
| `POST` | `/reset` | `{task_id?: string}` | `ResetResult` | Start new episode |
| `POST` | `/step` | `{action_type: string, parameters: object}` | `StepResult` | Execute one action |
| `GET` | `/state` | — | `StateResult` (incl. ground truth) | Full internal state |
| `GET` | `/tasks` | — | `{tasks: [{id, name, difficulty, max_steps, description}]}` | Task catalog |
| `GET` | `/metrics` | — | `{episode_count, mean_reward_by_task, step_distribution, difficulty_tier}` | Training metrics |
| `POST` | `/difficulty` | `{tier: 1\|2\|3}` | `{tier, status}` | Set difficulty tier |
| `GET` | `/scenarios` | — | `{scenarios: [...], count}` | Scenario catalog with MITRE tags |

### 3.2 Request/Response Schemas (Pydantic v2)

**Observation** — returned in every `/reset` and `/step` response:

| Field | Type | Description |
|---|---|---|
| `current_alert` | `Alert?` | Active alert under investigation |
| `alert_queue` | `List[Alert]` | Pending alerts awaiting triage |
| `available_logs` | `List[LogEntry]` | Forensic evidence retrieved this episode |
| `asset_inventory` | `List[Asset]` | Network assets (criticality 1–5, `is_compromised` always hidden) |
| `open_incidents` | `List[Incident]` | Active case files |
| `actions_taken` | `List[str]` | Action history log |
| `time_elapsed` | `int` | Simulated minutes (`step_count × 5`) |
| `task_context` | `TaskContext?` | Task objective, difficulty, step tracking |
| `message` | `str` | Human-readable status message |

---

## 4. Action Space

8 discrete action types, each with specific parameter schemas:

| Action Type | Parameters | Applicable Tasks | Description |
|---|---|---|---|
| `query_logs` | `{log_source, filter_ip?, filter_user?}` | T2, T3 | Query `auth\|firewall\|dns\|process\|network` logs |
| `classify_alert` | `{alert_id, classification, priority}` | T1, T2 | Label alert as `benign\|suspicious\|critical` with `P1–P4` |
| `isolate_asset` | `{hostname}` | T3 | Disconnect host from network |
| `block_ip` | `{ip_address}` | T3 | Block IP at perimeter firewall |
| `close_incident` | `{incident_id?, verdict, attack_type, attacker_ip?}` | T2, T3 | Submit final verdict |
| `write_report` | `{report: {summary, attack_type, affected_assets, attacker_ip, timeline}}` | T3 | Submit structured incident report |
| `escalate` | `{reason}` | T2, T3 | Escalate to Tier-2 analyst |
| `request_info` | `{question?}` | All | No-op information request |

---

## 5. Evaluation Tasks & Grading Logic

All scores are clamped to `(0.001, 0.999)` to prevent OpenEnv validation rejection.

### 5.1 Task 1: Alert Triage (Easy)

- **Objective:** Classify 10 security alerts with correct severity and priority.
- **Max Steps:** 15
- **Success Threshold:** 0.60
- **Grader (`grader1.py`):**
  - Classification accuracy: **70%** weight (exact match required, case-insensitive)
  - Priority accuracy: **30%** weight (off-by-one = 50% partial credit)
  - Coverage multiplier penalizes incomplete attempts
- **Ground Truth:** 10 alerts — 3 critical/P1, 3 suspicious/P2, 1 benign/P3, 3 benign/P4
- **Step Rewards:**
  - Correct classification: **+0.20**
  - Correct priority: **+0.10**, off-by-one: **+0.05**
  - Missing critical as benign: **−0.30**
  - False alarm (benign → critical): **−0.10**

### 5.2 Task 2: Incident Investigation (Medium)

- **Objective:** Query forensic logs, identify attacker IP, submit verdict with attack type.
- **Max Steps:** 20
- **Success Threshold:** 0.50
- **Grader (`grader2.py`):** Weighted scoring with dynamic ground truth lookup:

| Component | Weight | Scoring |
|---|---|---|
| Verdict correctness | 35% | Exact or alias match (`tp`, `true_pos`, `malicious`, etc.) |
| Attack type | 20% | Exact=1.0, substring=0.7, keyword overlap≥50%=0.4 |
| Evidence gathering | 30% | Log ID recall × 0.4 + source recall × 0.45 + strategy bonus (0.15) − noise penalty |
| Attacker IP | 15% | Exact=1.0, same /24=0.6, same /16=0.3, insider `internal`=1.0 |

- **Step Rewards:** New relevant source: **+0.15**, irrelevant: **−0.03**, key evidence ID: **+0.10**, correct verdict: **+0.30**, correct attack type: **+0.20**

### 5.3 Task 3: Active Threat Response (Hard)

- **Objective:** Gather evidence, contain threat (isolate + block), write incident report, close incident.
- **Max Steps:** 30
- **Success Threshold:** 0.40
- **Progressive Alert Disclosure:** Starts with 1 alert; additional alerts surface at steps 3 and 6.
- **Grader (`grader3.py`):**

| Component | Weight | Scoring |
|---|---|---|
| Containment | 30% | Correct isolations × 0.6 + correct IP blocks × 0.4 |
| Collateral damage | 20% | **Subtracted.** Penalty per wrongly-isolated critical asset (scaled by `criticality/5`) |
| Evidence gathering | 20% | Source recall against key evidence sources |
| Speed | 10% | Linear decay: full score at ≤50% step budget, zero at 100% |
| Report quality | 20% | Required fields: `summary`, `attack_type`, `affected_assets`, `attacker_ip`, `timeline` + bonuses for correct values |

- **Step Rewards:** Correct isolation: **+0.25**, collateral DC-01: up to **−0.40**, correct IP block: **+0.20**, report fields: up to **+0.30**
- **Report Required Fields:** `{summary, attack_type, affected_assets, attacker_ip, timeline}`

---

## 6. Simulation Layer

### 6.1 Scenario Catalog (7 Deterministic Seeds)

| # | Scenario ID | Attack Type | MITRE Techniques | Tier | Attacker IP |
|---|---|---|---|---|---|
| 1 | `brute_force_ssh_001` | Brute Force | T1110.001, T1078 | 1 | 185.220.101.47 |
| 2 | `phishing_lateral_001` | Lateral Movement | T1566.001, T1059.001, T1071.001, T1550.002, T1078.002 | 1 | 94.102.49.190 |
| 3 | `false_positive_scan_001` | False Positive (Benign) | — | 1 | — |
| 4 | `ransomware_001` | Malware (Ransomware) | T1486, T1490, T1021.002 | 2 | 192.168.50.99 |
| 5 | `insider_threat_001` | Data Exfiltration | T1074.001, T1567.002 | 2 | None (insider) |
| 6 | `supply_chain_001` | Malware (Backdoor) | T1195.002, T1543.002, T1071.001 | 3 | 203.0.113.42 |
| 7 | `multi_stage_apt_001` | Lateral Movement (APT) | T1078.001, T1134.001, T1053.005, T1572, T1021.002 | 3 | 45.33.32.156 |

### 6.2 Network Asset Inventory (7 Assets)

| Hostname | IP | Type | Criticality | Department |
|---|---|---|---|---|
| WEB-SERVER-01 | 10.0.1.20 | Server | 4 | Engineering |
| DC-01 | 10.0.0.5 | Domain Controller | **5** | IT |
| WS-HR-03 | 10.0.2.15 | Workstation | 2 | HR |
| DB-FINANCE-01 | 10.0.0.30 | Database | **5** | Finance |
| IT-SCANNER-01 | 10.0.0.100 | Workstation | 1 | IT |
| BACKUP-SRV-01 | 10.0.0.20 | Server | 3 | IT |
| WS-FINANCE-01 | 10.0.2.50 | Workstation | 3 | Finance |

### 6.3 Log Sources

5 forensic log sources: `auth`, `firewall`, `dns`, `process`, `network`. Each query returns scenario-specific entries plus 2 benign noise entries. The `is_malicious` ground truth field is **always hidden** from agent observations (set to `False`).

### 6.4 Adaptive Difficulty System

Rolling window of last 5 episode scores drives automatic tier escalation:

| Tier | Unlock Condition | Investigation Scenario | Threat Response Scenario |
|---|---|---|---|
| Tier 1 (Default) | — | `brute_force_ssh_001` | `phishing_lateral_001` |
| Tier 2 | Rolling avg > 0.70 | `ransomware_001` or `insider_threat_001` (random) | `ransomware_001` |
| Tier 3 | Rolling avg > 0.85 | `supply_chain_001` | `multi_stage_apt_001` |

Manual override available via `POST /difficulty {tier: 1|2|3}`.

---

## 7. Anti-Reward-Hacking Mechanisms

| Mechanism | Implementation | Location |
|---|---|---|
| **Thrashing penalty** | Same action type >5 repeats (except `classify_alert`) → −0.05 | `mini_soc_environment.py:276` |
| **Log ID deduplication** | `rewarded_log_ids` / `rewarded_sources` prevent double-counting | `grader2.py:190-204`, `grader3.py:233-239` |
| **Source deduplication** | Only first query of a source earns step reward | `grader2.py:192`, `grader3.py:234` |
| **Noise injection** | 2–3 benign log entries mixed into every `query_logs` result | `log_gen.py:91-132` |
| **Ground truth hiding** | `is_compromised` and `is_malicious` sanitized from observations | `log_gen.py:79-88`, `mini_soc_environment.py:580-587` |
| **Collateral damage scaling** | Isolating healthy critical assets penalized proportional to criticality | `grader3.py:130-141` |
| **Fuzzy matching** | Verdict/attack-type aliases prevent brittle exact-match gaming | `grader2.py:95-130` |
| **Score clamping** | All final scores clamped to `(0.001, 0.999)` | All graders |

---

## 8. Training Pipeline (GRPO)

### 8.1 Architecture

```
┌─────────────────┐      prompts       ┌──────────────────┐
│ build_soc_      │ ──────────────►    │ GRPOTrainer      │
│ dataset()       │                    │ (TRL ≥0.15)      │
│ (reward_wrapper)│                    │                  │
└─────────────────┘                    │  K completions   │
                                       │  per prompt      │
┌─────────────────┐    rewards[]       │                  │
│ soc_reward_     │ ◄──────────────    │                  │
│ function()      │ ──────────────►    │                  │
│ (reward_wrapper)│                    └──────────────────┘
│                 │
│  For each completion:
│  1. Parse JSON action(s)
│  2. POST /reset → /step loop
│  3. Return cumulative reward
└─────────────────┘
```

### 8.2 Components

| Component | File | Details |
|---|---|---|
| **Training Script** | `train/train_grpo.py` | HuggingFace TRL `GRPOTrainer`. LoRA (r=16, α=32) via PEFT. Optional Unsloth 2× acceleration. WandB integration. CLI + programmatic API. |
| **Reward Wrapper** | `train/reward_wrapper.py` | Bridges TRL ↔ environment. Parses completions as JSON actions. Supports single-action and multi-step plans. Malformed JSON → −0.1 penalty. |
| **Dataset Builder** | `train/reward_wrapper.py` | `build_soc_dataset()` generates HF `Dataset` by resetting environment and formatting observations as chat-style prompts. |
| **Reward Plotter** | `train/plot_rewards.py` | Publication-quality matplotlib charts. Dual-axis (reward + loss). Rolling average smoothing. Dark theme. Comparison bar charts. |
| **Smoke Tests** | `train/test_smoke.py` | Validates reward wrapper, dataset builder, and completion parsing. |
| **Colab Notebook** | `train/train_colab.ipynb` | Ready-to-run notebook for Google Colab training. |

### 8.3 Default Hyperparameters

| Parameter | Default | CLI Flag |
|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` | `--model` |
| Training steps | 200 | `--steps` |
| Batch size | 1 | `--batch-size` |
| Learning rate | 2e-5 | `--lr` |
| Group size (K) | 4 | `--K` |
| Max new tokens | 200 | — |
| Temperature | 0.7 | — |
| Training prompts | 60 (20/task) | `--prompts` |
| LoRA rank | 16 | — |
| LoRA alpha | 32 | — |

---

## 9. Testing & Quality Assurance

### 9.1 Test Suite (`tests/test_env.py`)

**35+ tests** across 6 categories:

| Category | Tests | Coverage |
|---|---|---|
| Reset behavior | 5 | All 3 tasks, invalid task, state clearing |
| Step mechanics | 6 | Log queries, correct/wrong classifications, correct/wrong asset isolation, post-done behavior |
| State endpoint | 1 | Ground truth presence and correctness |
| Grader correctness | 12 | Perfect scores, empty states, partial credit, collateral damage, insider threat handling, determinism, ransomware/APT scenarios |
| Adaptive difficulty | 2 | Tier escalation thresholds, manual tier setting |
| API endpoint coverage | 8 | All 9 HTTP endpoints via FastAPI TestClient |
| Scenario wiring | 4 | Tier 1/2/3 scenario selection for both investigation and response tasks |

### 9.2 Configuration

- **Framework:** pytest ≥ 8.0 with `pytest-cov` ≥ 5.0
- **Coverage target:** ≥ 80% (`fail_under = 80`)
- **Linter:** Ruff (target: Python 3.10, line-length: 120)
- **Type checking:** mypy with `check_untyped_defs = true`

---

## 10. Deployment & Infrastructure

### 10.1 Docker

- **Base image:** `python:3.11-slim`
- **Build:** Multi-stage — Stage 1 caches pip dependencies, Stage 2 copies application code
- **Health check:** `curl -f http://localhost:8000/health` every 30s
- **Default CMD:** `uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1`

### 10.2 Docker Compose (3 services)

| Service | Container | Command | Profile |
|---|---|---|---|
| `soc-env` | `mini-soc-env` | uvicorn (default) | default |
| `soc-agent` | `mini-soc-agent` | `python run_agent.py` | default |
| `soc-tests` | `mini-soc-tests` | `pytest tests/ -v` | `test` |

### 10.3 OpenEnv Compatibility

- **Manifest:** `openenv.yaml` — fully compliant with OpenEnv 1.0.0 spec
- **Required Python:** ≥ 3.10
- **Core dependencies:** `fastapi`, `uvicorn`, `pydantic`, `httpx`, `openai`, `pyyaml`
- **Baseline model:** `Qwen/Qwen2.5-72B-Instruct`
- **Baseline script:** `inference.py`

### 10.4 Hugging Face Spaces

Configured via README YAML frontmatter: `sdk: docker`, `app_port: 8000`, tags: `openenv`, `cybersecurity`, `reinforcement-learning`.

---

## 11. Baseline Performance

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Expected Score | Success Threshold | Max Steps |
|---|---|---|---|
| Alert Triage | 0.65 | 0.60 | 15 |
| Incident Investigation | 0.42 | 0.50 | 20 |
| Threat Response | 0.22 | 0.40 | 30 |
| **Average** | **0.43** | — | — |

---

## 12. File Inventory

```
mini-soc/                          # Root
├── __init__.py                    # OpenEnv package exports
├── models.py                      # 15 Pydantic models + 4 enums (199 LOC)
├── client.py                      # MiniSocEnv SDK client (97 LOC)
├── inference.py                   # LLM baseline agent (322 LOC)
├── run_agent.py                   # Deterministic expert agent (350 LOC)
├── openenv.yaml                   # OpenEnv manifest (112 LOC)
├── pyproject.toml                 # Package config + tooling (123 LOC)
├── requirements.txt               # Pip dependencies
├── Dockerfile                     # Multi-stage Docker build (64 LOC)
├── docker-compose.yml             # 3-service orchestration (62 LOC)
├── Makefile                       # Build automation
├── README.md                      # User-facing documentation (292 LOC)
├── PRD.md                         # This document
├── CHANGELOG.md                   # Version history
├── LICENSE                        # MIT
├── server/
│   ├── __init__.py
│   ├── app.py                     # FastAPI application factory (249 LOC)
│   ├── mini_soc_environment.py    # Core environment engine (602 LOC)
│   ├── logging_config.py          # Structured logging
│   ├── simulator/
│   │   ├── attack_seeds.py        # 7 scenarios + alert queue + assets (893 LOC)
│   │   └── log_gen.py             # Log retrieval + noise injection (133 LOC)
│   └── graders/
│       ├── grader1.py             # Task 1 scoring (121 LOC)
│       ├── grader2.py             # Task 2 scoring (239 LOC)
│       └── grader3.py             # Task 3 scoring (270 LOC)
├── train/
│   ├── __init__.py
│   ├── train_grpo.py              # GRPO training script (306 LOC)
│   ├── reward_wrapper.py          # TRL reward bridge (328 LOC)
│   ├── plot_rewards.py            # Reward curve plotter (311 LOC)
│   ├── test_smoke.py              # Training smoke tests
│   └── train_colab.ipynb          # Colab notebook
├── tests/
│   ├── conftest.py                # 4 shared fixtures (42 LOC)
│   └── test_env.py                # 35+ tests (635 LOC)
├── docs/
│   ├── API.md                     # API reference
│   └── architecture.svg           # Architecture diagram
└── outputs/                       # Runtime artifacts (gitignored)
```

**Total production codebase:** ~4,500 LOC across 20 source files.

---

## 13. Future Roadmap

1. **Procedural Scenario Generation:** Replace 7 static seeds with parameterized, randomized log volumes to prevent agent overfitting.
2. **Multi-Agent Orchestration:** Tier-1 Triager hands off state to specialized Tier-2 Investigator agent.
3. **Graph-Based Network Topology:** Replace flat asset list with dynamic graph structure for lateral movement mapping.
4. **Cloud-Scale GRPO Validation:** Distributed RL training at scale to establish official LLM performance leaderboards.
5. **Real-Time Log Streaming:** WebSocket-based log delivery simulating actual SIEM feed cadence.
6. **Extended MITRE Coverage:** Expand from 17 unique techniques to 50+ across all 14 tactics.