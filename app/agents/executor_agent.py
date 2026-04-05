"""
ExecutorAgent — runs sub-tasks respecting dependency order.

Sub-tasks with no dependencies run in parallel (asyncio.gather).
Sub-tasks with dependencies run after their predecessors complete,
receiving the predecessor results as context.

This is what makes the system genuinely agentic — not just routing,
but orchestrating multi-step execution with data flow between steps.
"""
import asyncio
import time
import logging
from typing import List, Dict, Optional

from app.agents.base_agent import BaseAgent, SubTask, AgentStatus
from app.models.base_model import BaseLLM, Message
from app.router.routing_rules import RoutingRules, RoutingStrategy, RoutingDecision
from app.classifier.query_classifier import QueryClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):

    def __init__(
        self,
        models: Dict[str, BaseLLM],
        rules: RoutingRules,
        classifier: QueryClassifier,
    ):
        super().__init__("Executor")
        self.models     = models
        self.rules      = rules
        self.classifier = classifier

    async def execute_plan(
        self,
        tasks: List[SubTask],
        available_models: List[str],
        strategy: RoutingStrategy,
        budget_remaining: float,
        original_query: str,
        history: Optional[List[Message]] = None,
    ) -> List[SubTask]:
        """Execute all sub-tasks in dependency order."""

        completed: Dict[str, SubTask] = {}
        remaining = list(tasks)

        while remaining:
            # Find tasks whose dependencies are all completed
            ready = [
                t for t in remaining
                if all(dep in completed for dep in t.depends_on)
            ]

            if not ready:
                # Should never happen with valid plan, but handle gracefully
                self.log("Dependency deadlock — forcing remaining tasks", "warning")
                ready = remaining[:1]

            # Run ready tasks in parallel
            results = await asyncio.gather(
                *[self._run_task(t, completed, available_models, strategy, budget_remaining) for t in ready]
            )

            for task in results:
                completed[task.id] = task
                remaining = [t for t in remaining if t.id != task.id]

        return list(completed.values())

    async def _run_task(
        self,
        task: SubTask,
        completed: Dict[str, SubTask],
        available_models: List[str],
        strategy: RoutingStrategy,
        budget_remaining: float,
    ) -> SubTask:
        task.status = AgentStatus.RUNNING
        t0 = time.monotonic()

        try:
            # Build context from dependency results
            context = self._build_context(task, completed)

            # Route this specific sub-task
            cls = self.classifier.classify(task.description)
            decision = self.rules.decide(
                cls,
                budget_remaining_usd=budget_remaining,
                available_models=available_models,
                strategy=strategy,
            )

            # Execute with fallback
            messages = [Message(role="user", content=task.description)]
            if context:
                messages.insert(0, Message(
                    role="system",
                    content=f"Previous analysis context:\n{context}\n\nNow answer this specific part:"
                ))

            response = await self._call_with_fallback(messages, decision)

            task.result     = response.content
            task.model_used = response.model_id
            task.cost_usd   = response.cost_usd
            task.status     = AgentStatus.SUCCESS if response.success else AgentStatus.FAILED

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = AgentStatus.FAILED
            task.result = f"[Task failed: {str(e)[:100]}]"

        task.latency_ms = (time.monotonic() - t0) * 1000
        return task

    def _build_context(self, task: SubTask, completed: Dict[str, SubTask]) -> str:
        """Concatenate results from dependency tasks as context."""
        if not task.depends_on:
            return ""
        parts = []
        for dep_id in task.depends_on:
            dep = completed.get(dep_id)
            if dep and dep.result:
                parts.append(f"[{dep.description[:60]}]:\n{dep.result[:800]}")
        return "\n\n".join(parts)

    async def _call_with_fallback(self, messages, decision: RoutingDecision):
        order = [decision.primary_model_id] + decision.fallback_model_ids
        for mid in order:
            model = self.models.get(mid)
            if not model:
                continue
            resp = await model.complete(messages)
            if resp.success:
                return resp
        # Return last attempt (failed)
        return list(self.models.values())[0]._make_error_response("All models failed")
