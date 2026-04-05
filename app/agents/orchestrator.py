"""
AgentOrchestrator — the top-level coordinator of the agentic pipeline.

Flow:
  1. PlannerAgent   → decompose query into sub-tasks
  2. ExecutorAgent  → run sub-tasks (parallel where possible)
  3. SynthesizerAgent → merge results into final answer

Falls back to direct single-model execution if agents are disabled
or if the query is simple enough not to need decomposition.
"""
import time
import logging
from typing import List, Dict, Optional

from app.agents.base_agent import AgentTrace, AgentStatus
from app.agents.planner_agent import PlannerAgent
from app.agents.executor_agent import ExecutorAgent
from app.agents.synthesizer_agent import SynthesizerAgent
from app.classifier.query_classifier import ClassificationResult, QueryClassifier
from app.models.base_model import BaseLLM, Message
from app.router.routing_rules import RoutingRules, RoutingStrategy
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentOrchestrator:

    def __init__(self, models: Dict[str, BaseLLM], classifier: QueryClassifier, rules: RoutingRules):
        self.models      = models
        self.classifier  = classifier
        self.rules       = rules
        self.planner     = PlannerAgent()
        self.executor    = ExecutorAgent(models, rules, classifier)
        self.synthesizer = SynthesizerAgent(models)

    async def run(
        self,
        query: str,
        classification: ClassificationResult,
        available_models: List[str],
        strategy: RoutingStrategy,
        budget_remaining: float,
        history: Optional[List[Message]] = None,
    ) -> AgentTrace:
        t0 = time.monotonic()

        # 1. Plan
        tasks = self.planner.plan(query, classification)
        was_decomposed = len(tasks) > 1

        if was_decomposed:
            logger.info(f"[Orchestrator] Decomposed into {len(tasks)} sub-tasks")

        # 2. Execute
        executed_tasks = await self.executor.execute_plan(
            tasks=tasks,
            available_models=available_models,
            strategy=strategy,
            budget_remaining=budget_remaining,
            original_query=query,
            history=history,
        )

        # 3. Synthesize
        final_answer = await self.synthesizer.synthesize(
            original_query=query,
            tasks=executed_tasks,
            available_models=available_models,
        )

        total_ms   = (time.monotonic() - t0) * 1000
        total_cost = sum(t.cost_usd for t in executed_tasks)
        models_used = list({t.model_used for t in executed_tasks if t.model_used})

        steps = [
            {
                "task_id":    t.id,
                "description": t.description[:80],
                "intent":     t.intent,
                "model_used": t.model_used,
                "status":     t.status.value,
                "latency_ms": round(t.latency_ms, 1),
                "cost_usd":   t.cost_usd,
            }
            for t in executed_tasks
        ]

        return AgentTrace(
            plan=executed_tasks,
            final_answer=final_answer,
            total_latency_ms=round(total_ms, 1),
            total_cost_usd=round(total_cost, 8),
            models_used=models_used,
            steps=steps,
            was_decomposed=was_decomposed,
        )
