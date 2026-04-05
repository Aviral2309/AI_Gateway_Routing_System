"""
Base Agent — foundation for the agentic AI layer.

The agent pipeline:
  PlannerAgent   → decomposes complex queries into sub-tasks
  RouterAgent    → selects best model per sub-task
  ExecutorAgent  → runs sub-tasks (in parallel when possible)
  EvaluatorAgent → scores each result, flags failures
  RetryAgent     → re-routes failed sub-tasks to better models

This is the architecture that separates Nexus from a simple proxy.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    RETRYING  = "retrying"


@dataclass
class SubTask:
    """A single unit of work produced by the PlannerAgent."""
    id: str
    description: str          # what needs to be answered
    intent: str               # code / reasoning / factual / creative
    complexity: str           # simple / medium / complex
    depends_on: List[str]     # IDs of sub-tasks that must complete first
    result: Optional[str]     = None
    model_used: Optional[str] = None
    status: AgentStatus       = AgentStatus.PENDING
    latency_ms: float         = 0.0
    cost_usd: float           = 0.0
    quality_score: Optional[float] = None


@dataclass
class AgentTrace:
    """Full execution trace — shown in API response for transparency."""
    plan: List[SubTask]
    final_answer: str
    total_latency_ms: float
    total_cost_usd: float
    models_used: List[str]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    was_decomposed: bool = False


class BaseAgent:
    """Shared utilities for all agents."""

    def __init__(self, name: str):
        self.name = name

    def log(self, msg: str, level: str = "info"):
        getattr(logger, level)(f"[{self.name}] {msg}")

    def _now_ms(self) -> float:
        return time.monotonic() * 1000
