"""
Mini SOC — FastAPI Application (OpenEnv Server)
Implements OpenEnv HTTP API: /reset, /step, /state, /tasks, /health

Uses the create_app factory pattern for session isolation.
Supports dual-import for both package-mode and Docker-mode.
"""
import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Dual-import pattern
try:
    from models import Action, ActionType, ResetResult, StepResult, StateResult
except ImportError:
    from ..models import Action, ActionType, ResetResult, StepResult, StateResult

try:
    from server.mini_soc_environment import SocEnvironment, DifficultyTier
except ImportError:
    from .mini_soc_environment import SocEnvironment, DifficultyTier

try:
    from server.simulator.attack_seeds import ATTACK_SCENARIOS
except ImportError:
    from .simulator.attack_seeds import ATTACK_SCENARIOS

try:
    from server.logging_config import logger
except ImportError:
    from .logging_config import logger


# ---------------------------------------------------------------------------
# Request schemas (module-level so Pydantic can resolve annotations)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "alert_triage"


class StepRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


class DifficultyRequest(BaseModel):
    tier: int


# ---------------------------------------------------------------------------
# App factory — OpenEnv standard pattern
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Factory function that creates and configures the FastAPI app.
    Each call returns a fresh app with its own environment instance.
    """
    application = FastAPI(
        title="Mini SOC — OpenEnv Environment",
        description="AI SOC Analyst environment for RL agent training and evaluation.",
        version="2.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single environment instance (stateful per session)
    _env = SocEnvironment()

    # Global exception handler
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # -------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------

    @application.get("/health")
    def health():
        return {"status": "ok", "env": "mini-soc", "version": "2.0.0"}

    @application.post("/reset", response_model=Dict[str, Any])
    def reset(request: ResetRequest = ResetRequest()):
        """
        Reset the environment and start a new episode.
        Returns initial observation.
        """
        task = request.task_id or "alert_triage"
        try:
            logger.info("Resetting environment — task=%s", task)
            result: ResetResult = _env.reset(task_id=task)
            logger.info("Episode started — task=%s episode=%s", task, result.info.get("episode_id"))
            return result.model_dump(mode="json")
        except ValueError as e:
            logger.warning("Invalid reset request: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Reset failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

    @application.post("/step", response_model=Dict[str, Any])
    def step(request: StepRequest):
        """
        Submit one agent action. Returns next observation, reward, done flag.
        """
        try:
            action = Action(
                action_type=ActionType(request.action_type),
                parameters=request.parameters,
            )
            result: StepResult = _env.step(action)
            logger.debug("Step — action=%s reward=%.4f done=%s", request.action_type, result.reward, result.done)
            return result.model_dump(mode="json")
        except ValueError as e:
            logger.warning("Invalid action: %s", e)
            raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
        except Exception as e:
            logger.error("Step failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Step failed: {e}")

    @application.get("/state", response_model=Dict[str, Any])
    def state():
        """
        Returns current full environment state.
        Used by graders and debugging tools.
        """
        try:
            result: StateResult = _env.state()
            return result.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"State failed: {e}")

    @application.get("/tasks")
    def tasks():
        """Lists all available tasks with metadata."""
        return {
            "tasks": [
                {
                    "id": "alert_triage",
                    "name": "Alert Triage",
                    "difficulty": "easy",
                    "max_steps": 15,
                    "description": "Classify 10 security alerts with correct priority.",
                },
                {
                    "id": "incident_investigation",
                    "name": "Incident Investigation",
                    "difficulty": "medium",
                    "max_steps": 20,
                    "description": "Query logs, correlate evidence, submit verdict.",
                },
                {
                    "id": "threat_response",
                    "name": "Active Threat Response",
                    "difficulty": "hard",
                    "max_steps": 30,
                    "description": "Detect kill chain, isolate assets, write report.",
                },
            ]
        }

    # -------------------------------------------------------------------
    # v2.0 Endpoints — Metrics, Difficulty, Scenarios
    # -------------------------------------------------------------------

    @application.get("/metrics")
    def metrics():
        """
        Training metrics: episode count, mean reward by task,
        step distribution, current difficulty tier.
        """
        return _env.get_metrics()

    @application.post("/difficulty")
    def set_difficulty(request: DifficultyRequest):
        """
        Manually set the adaptive difficulty tier (1, 2, or 3).
        Used for testing and evaluation.
        """
        if request.tier not in (1, 2, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier {request.tier}. Must be 1, 2, or 3.",
            )
        _env.set_difficulty_tier(request.tier)
        logger.info("Difficulty tier set to %d", request.tier)
        return {"tier": request.tier, "status": "updated"}

    @application.get("/scenarios")
    def scenarios():
        """
        List all attack scenarios with metadata:
        ID, type, difficulty tier, kill chain, MITRE techniques.
        """
        result = []
        for scenario_id, data in ATTACK_SCENARIOS.items():
            gt = data.get("ground_truth", {})
            result.append({
                "scenario_id": scenario_id,
                "attack_type": data.get("attack_type", "unknown"),
                "classification": gt.get("classification", ""),
                "priority": gt.get("priority", ""),
                "kill_chain": data.get("kill_chain", []),
                "mitre_techniques": gt.get("mitre_techniques", []),
                "affected_assets": gt.get("affected_assets", []),
                "alert_count": len(data.get("alerts", [])),
            })
        return {"scenarios": result, "count": len(result)}

    @application.get("/")
    def root():
        return {
            "name": "mini-soc",
            "description": "Mini Security Operations Center RL environment",
            "version": "2.0.0",
            "endpoints": [
                "/reset", "/step", "/state", "/tasks",
                "/health", "/metrics", "/difficulty", "/scenarios",
            ],
            "openenv_spec": "1.0.0",
        }

    return application


# Create the app instance using factory pattern
app = create_app()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
