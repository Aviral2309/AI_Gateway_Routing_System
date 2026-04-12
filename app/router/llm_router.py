"""
LLMRouter — central orchestrator.

Key improvements over previous version:
- Always has a fallback: if all health checks fail, still tries Ollama directly
- Never crashes with a 500: every error is caught and returned gracefully
- Clear fallback chain: Gemini Flash → Gemini Lite → Ollama → error message
- Works with zero API keys (Ollama only mode)
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.classifier.query_classifier import QueryClassifier, ClassificationResult
from app.models.base_model import BaseLLM, LLMResponse, Message
from app.models.gemini_model import GeminiModel
from app.models.ollama_model import OllamaModel
from app.router.routing_rules import RoutingRules, RoutingStrategy, RoutingDecision, OLLAMA
from app.evaluator.response_evaluator import ResponseEvaluator, EvaluationResult
from app.tracking.tracker import Tracker, RequestRecord
from app.cache.redis_cache import cache, CacheEntry
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base_agent import AgentTrace
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RouterResponse:
    content: str
    model_used: str
    classification: ClassificationResult
    decision: RoutingDecision
    evaluation: Optional[EvaluationResult]
    latency_ms: float
    cost_usd: float
    total_tokens: int
    success: bool
    error: Optional[str]
    attempts: int
    from_cache: bool = False
    agent_trace: Optional[AgentTrace] = None


class LLMRouter:

    def __init__(self):
        self.classifier  = QueryClassifier()
        self.rules       = RoutingRules()
        self.evaluator   = ResponseEvaluator()
        self.tracker     = Tracker()
        self._models: Dict[str, BaseLLM] = {}
        self._register_models()
        self.agent_orchestrator = AgentOrchestrator(
            models=self._models,
            classifier=self.classifier,
            rules=self.rules,
        )
        logger.info(f"LLMRouter ready with {len(self._models)} models: {list(self._models.keys())}")

    # ─── Main entry point ────────────────────────────────────────────────────

    async def route(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        use_agents: Optional[bool] = None,
    ) -> RouterResponse:

        history    = history or []
        session_id = session_id or "default"

        # 1. Classify
        classification = self.classifier.classify(query, context_turns=len(history))
        logger.info(f"[Router] {classification.reasoning}")

        # 2. Cache check
        if not history and classification.tier == "simple":
            cached = await cache.get(query, strategy.value)
            if cached:
                await self._track(
                    session_id, query, cached.content, cached.model_used,
                    classification, strategy, 0, 0, 0.0, 0.0,
                    True, False, 0, 0,
                )
                return self._cached_response(cached, classification, strategy)

        # 3. Get available models (with guaranteed fallback)
        available = await self._get_available_models()

        # 4. Budget
        budget = await self.tracker.get_session_budget_remaining_db(
            session_id, settings.default_budget_usd
        )

        # 5. Agentic routing for complex queries
        agents_on = use_agents if use_agents is not None else settings.enable_agents
        if agents_on and classification.tier == "complex" and not history:
            return await self._agent_route(
                query, classification, available, strategy, budget, session_id, history
            )

        # 6. Standard routing
        decision  = self.rules.decide(classification, budget, available, strategy)
        messages  = history + [Message(role="user", content=query)]
        response, attempts = await self._execute_with_fallback(
            messages, system_prompt, decision, temperature, max_tokens
        )

        # 7. Evaluate
        evaluation: Optional[EvaluationResult] = None
        if settings.enable_evaluation and not decision.skip_evaluation and response.success:
            try:
                evaluation = await self.evaluator.evaluate(
                    query, response.content, classification.intent
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        # 8. Cache
        if response.success and not history and classification.tier == "simple":
            await cache.set(query, strategy.value, CacheEntry(
                content=response.content, model_used=response.model_id,
                cost_usd=response.cost_usd, latency_ms=response.latency_ms,
            ))

        # 9. Track
        await self._track(
            session_id, query,
            response.content if response.success else None,
            response.model_id, classification, strategy,
            response.input_tokens, response.output_tokens,
            response.latency_ms, response.cost_usd,
            False, False, 1, attempts - 1,
            quality_score=evaluation.score if evaluation else None,
            quality_reason=evaluation.reason if evaluation else None,
            error=response.error,
        )

        return RouterResponse(
            content=response.content, model_used=response.model_id,
            classification=classification, decision=decision,
            evaluation=evaluation, latency_ms=response.latency_ms,
            cost_usd=response.cost_usd, total_tokens=response.total_tokens,
            success=response.success, error=response.error,
            attempts=attempts, from_cache=False,
        )

    # ─── Agentic route ───────────────────────────────────────────────────────

    async def _agent_route(
        self, query, classification, available, strategy, budget, session_id, history
    ) -> RouterResponse:
        logger.info("[Router] Agent pipeline activated")
        try:
            trace = await self.agent_orchestrator.run(
                query=query, classification=classification,
                available_models=available, strategy=strategy,
                budget_remaining=budget, history=history,
            )
        except Exception as e:
            logger.error(f"Agent pipeline failed: {e} — falling back to direct")
            # Fall back to direct routing
            decision = self.rules.decide(classification, budget, available, strategy)
            messages = [Message(role="user", content=query)]
            response, attempts = await self._execute_with_fallback(
                messages, None, decision, 0.7, 1024
            )
            return RouterResponse(
                content=response.content, model_used=response.model_id,
                classification=classification, decision=decision,
                evaluation=None, latency_ms=response.latency_ms,
                cost_usd=response.cost_usd, total_tokens=response.total_tokens,
                success=response.success, error=response.error,
                attempts=attempts, from_cache=False,
            )

        decision   = self.rules.decide(classification, budget, available, strategy)
        models_str = ", ".join(set(trace.models_used)) if trace.models_used else "multi-agent"

        evaluation: Optional[EvaluationResult] = None
        if settings.enable_evaluation:
            try:
                evaluation = await self.evaluator.evaluate(
                    query, trace.final_answer, classification.intent
                )
            except Exception:
                pass

        await self._track(
            session_id, query, trace.final_answer, models_str,
            classification, strategy, 0, 0,
            trace.total_latency_ms, trace.total_cost_usd,
            False, True, len(trace.plan), 0,
            quality_score=evaluation.score if evaluation else None,
        )

        return RouterResponse(
            content=trace.final_answer, model_used=models_str,
            classification=classification, decision=decision,
            evaluation=evaluation, latency_ms=trace.total_latency_ms,
            cost_usd=trace.total_cost_usd, total_tokens=0,
            success=True, error=None, attempts=len(trace.plan),
            from_cache=False, agent_trace=trace,
        )

    # ─── Model registry ──────────────────────────────────────────────────────

    def register_model(self, model: BaseLLM):
        self._models[model.model_id] = model

    def _register_models(self):
        # Register Gemini models (will show as unreachable if no API key — that's fine)
        self.register_model(GeminiModel(settings.gemini_primary_model))
        self.register_model(GeminiModel(settings.gemini_lite_model))
        # Always register Ollama — it's the guaranteed local fallback
        self.register_model(OllamaModel())

    async def _get_available_models(self) -> List[str]:
        """
        Check which models are actually reachable.
        CRITICAL: Always returns at least ['ollama-local'] even if health check
        fails — we try Ollama regardless because health check can be flaky.
        """
        checks  = {mid: m.health_check() for mid, m in self._models.items()}
        results = await asyncio.gather(*checks.values(), return_exceptions=True)
        available = [mid for mid, r in zip(checks.keys(), results) if r is True]

        # Safety guarantee: if nothing passes health check, still include ollama
        # The actual request will surface the real error if Ollama is truly down
        if not available:
            logger.warning("No models passed health check — defaulting to ollama-local")
            available = [OLLAMA]

        logger.info(f"Available models: {available}")
        return available

    # ─── Execution ───────────────────────────────────────────────────────────

    async def _execute_with_fallback(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        decision: RoutingDecision,
        temperature: float,
        max_tokens: int,
    ) -> tuple[LLMResponse, int]:
        """
        Try primary model, then all fallbacks, then all registered models.
        Never crashes — always returns something.
        """
        # Build ordered list: primary → fallbacks → everything else
        order = [decision.primary_model_id] + decision.fallback_model_ids
        # Add any remaining models not yet in the list
        for mid in self._models:
            if mid not in order:
                order.append(mid)

        last_response = None
        for attempt, mid in enumerate(order, start=1):
            model = self._models.get(mid)
            if not model:
                continue

            logger.info(f"[Attempt {attempt}] Calling {mid}")
            try:
                resp = await model.complete(
                    messages, system_prompt, temperature, max_tokens
                )
            except Exception as e:
                logger.error(f"Model {mid} raised exception: {e}")
                resp = model._err(str(e)[:200])

            if resp.success:
                logger.info(f"[Success] {mid} responded in {resp.latency_ms:.0f}ms")
                return resp, attempt

            logger.warning(f"[Fail {attempt}] {mid}: {resp.error}")
            last_response = resp

            if attempt >= settings.max_retries + len(order):
                break

        # Absolute last resort — return a helpful error message
        if last_response:
            return last_response, len(order)

        # Construct a clean error response
        err_resp = list(self._models.values())[0]._err(
            "All models unavailable. Check: 1) Gemini API key in .env  "
            "2) Ollama running (ollama serve)  3) At least one model pulled (ollama pull llama3.2:latest)"
        )
        return err_resp, len(order)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    async def _track(
        self, session_id, query, response_text, model_used,
        classification, strategy, input_tokens, output_tokens,
        latency_ms, cost_usd, from_cache, was_decomposed, agent_steps, retry_count,
        quality_score=None, quality_reason=None, error=None,
    ):
        try:
            record = RequestRecord(
                session_id=session_id, query=query, response=response_text,
                model_used=model_used, tier=classification.tier,
                intent=classification.intent,
                complexity_score=classification.complexity_score,
                routing_strategy=strategy.value,
                input_tokens=input_tokens, output_tokens=output_tokens,
                latency_ms=latency_ms, cost_usd=cost_usd,
                success=response_text is not None or from_cache,
                retry_count=retry_count, from_cache=from_cache,
                was_decomposed=was_decomposed, agent_steps=agent_steps,
                quality_score=quality_score, quality_reason=quality_reason,
                error_message=error,
            )
            await self.tracker.record_async(record)
        except Exception as e:
            logger.error(f"Tracking failed: {e}")

    def _cached_response(self, cached: CacheEntry, classification, strategy) -> RouterResponse:
        decision = RoutingDecision(
            primary_model_id=cached.model_used, fallback_model_ids=[],
            strategy=strategy, reason="cache hit",
            estimated_cost_usd=0.0, skip_evaluation=True,
        )
        return RouterResponse(
            content=cached.content,
            model_used=f"{cached.model_used}[cache]",
            classification=classification, decision=decision,
            evaluation=None, latency_ms=0.0, cost_usd=0.0,
            total_tokens=0, success=True, error=None,
            attempts=0, from_cache=True,
        )
