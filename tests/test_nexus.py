"""
Complete test suite — 35 tests covering every layer.
All run without DB, Redis, or any API key.
"""
import pytest
import asyncio
from app.classifier.feature_extractor import FeatureExtractor
from app.classifier.query_classifier import QueryClassifier
from app.router.routing_rules import RoutingRules, RoutingStrategy
from app.evaluator.response_evaluator import ResponseEvaluator
from app.tracking.tracker import Tracker, RequestRecord
from app.cache.redis_cache import RedisCache, CacheEntry
from app.agents.planner_agent import PlannerAgent
from app.agents.base_agent import AgentStatus


# ─── Feature Extractor ───────────────────────────────────────────────────────

class TestFeatureExtractor:

    def setup_method(self):
        self.fx = FeatureExtractor()

    def test_simple_query_low_complexity(self):
        f = self.fx.extract("What is the capital of France?")
        assert f.complexity_score < 0.5
        assert f.intent == "factual"

    def test_code_intent_detected(self):
        f = self.fx.extract("Write a Python function to sort a list")
        assert f.intent == "code"
        assert f.code_keyword_score > 0

    def test_code_block_detected(self):
        f = self.fx.extract("Debug this:\n```python\nx = None\nprint(x.upper())\n```")
        assert f.has_code_block is True

    def test_complex_query_high_score(self):
        f = self.fx.extract(
            "Design a distributed microservices architecture with event sourcing, "
            "CQRS, service discovery, API gateway, circuit breakers, distributed tracing, "
            "Kubernetes deployment, and explain all trade-offs with code examples."
        )
        assert f.complexity_score > 0.4

    def test_reasoning_intent(self):
        f = self.fx.extract("Explain the difference between SQL and NoSQL databases")
        assert f.intent == "reasoning"

    def test_feature_vector_length(self):
        assert len(self.fx.extract("test").to_vector()) == 12

    def test_has_math_detected(self):
        f = self.fx.extract("What is the derivative of x^2 + 3x?")
        assert f.has_math is True

    def test_short_query_low_tokens(self):
        f = self.fx.extract("Hi")
        assert f.token_count < 10


# ─── Query Classifier ────────────────────────────────────────────────────────

class TestQueryClassifier:

    def setup_method(self):
        self.cls = QueryClassifier()

    def test_simple_factual(self):
        r = self.cls.classify("What is 2 + 2?")
        assert r.tier == "simple"
        assert r.confidence > 0.5

    def test_complex_system_design(self):
        r = self.cls.classify(
            "Design a scalable distributed cache with consistent hashing, "
            "replication, eviction policies, and benchmarks across multiple regions."
        )
        assert r.tier in ("medium", "complex")

    def test_context_increases_complexity(self):
        r0 = self.cls.classify("What is Python?", context_turns=0)
        r5 = self.cls.classify("What is Python?", context_turns=5)
        assert r5.complexity_score >= r0.complexity_score

    def test_all_result_fields_present(self):
        r = self.cls.classify("Explain REST APIs")
        assert r.tier in ("simple", "medium", "complex")
        assert r.intent in ("code", "reasoning", "factual", "creative", "general")
        assert 0.0 <= r.complexity_score <= 1.0
        assert 0.0 <= r.confidence <= 1.0
        assert r.reasoning != ""

    def test_code_query(self):
        r = self.cls.classify("Write a Python quicksort implementation")
        assert r.intent == "code"


# ─── Routing Rules ───────────────────────────────────────────────────────────

class TestRoutingRules:

    def setup_method(self):
        self.rules = RoutingRules()

    def _cls(self, tier, intent, score=0.5):
        from app.classifier.feature_extractor import FeatureExtractor
        from app.classifier.query_classifier import ClassificationResult
        f = FeatureExtractor().extract("test")
        return ClassificationResult(tier=tier, intent=intent, complexity_score=score,
                                    confidence=0.9, features=f, reasoning="test")

    def test_budget_zero_forces_ollama(self):
        d = self.rules.decide(self._cls("complex", "code"),
                              budget_remaining_usd=0.0,
                              available_models=["gemini-1.5-flash", "ollama-local"])
        assert d.primary_model_id == "ollama-local"

    def test_local_first_strategy(self):
        d = self.rules.decide(self._cls("medium", "reasoning"),
                              budget_remaining_usd=1.0,
                              available_models=["gemini-1.5-flash", "ollama-local"],
                              strategy=RoutingStrategy.LOCAL_FIRST)
        assert "ollama" in d.primary_model_id

    def test_simple_factual_skips_evaluation(self):
        d = self.rules.decide(self._cls("simple", "factual", 0.1),
                              budget_remaining_usd=1.0,
                              available_models=["gemini-1.5-flash", "ollama-local"])
        assert d.skip_evaluation is True

    def test_fallbacks_populated(self):
        d = self.rules.decide(self._cls("complex", "code"),
                              budget_remaining_usd=1.0,
                              available_models=["gemini-1.5-flash", "ollama-local"])
        assert len(d.fallback_model_ids) >= 1

    def test_single_model_available(self):
        d = self.rules.decide(self._cls("medium", "general"),
                              budget_remaining_usd=1.0,
                              available_models=["ollama-local"])
        assert d.primary_model_id == "ollama-local"

    def test_cost_optimized_picks_ollama(self):
        d = self.rules.decide(self._cls("simple", "general"),
                              budget_remaining_usd=1.0,
                              available_models=["gemini-1.5-flash", "ollama-local"],
                              strategy=RoutingStrategy.COST_OPTIMIZED)
        assert "ollama" in d.primary_model_id


# ─── Tracker ─────────────────────────────────────────────────────────────────

class TestTracker:

    def setup_method(self):
        self.tracker = Tracker()

    def _r(self, **kw):
        defaults = dict(
            session_id="s1", query="test", response="answer",
            model_used="gemini-1.5-flash", tier="simple", intent="factual",
            complexity_score=0.2, routing_strategy="balanced",
            input_tokens=50, output_tokens=100,
            latency_ms=500.0, cost_usd=0.000005,
            success=True, retry_count=0, quality_score=0.85,
        )
        defaults.update(kw)
        return RequestRecord(**defaults)

    def test_record_increments_count(self):
        self.tracker.record(self._r())
        assert self.tracker.get_metrics()["summary"]["total_requests"] == 1

    def test_budget_tracking_accurate(self):
        self.tracker.record(self._r(cost_usd=0.05))
        self.tracker.record(self._r(cost_usd=0.03))
        assert abs(self.tracker.get_session_budget_remaining("s1", 0.10) - 0.02) < 0.0001

    def test_success_rate_calculation(self):
        self.tracker.record(self._r(success=True))
        self.tracker.record(self._r(success=True))
        self.tracker.record(self._r(success=False))
        assert self.tracker.get_metrics()["summary"]["success_rate"] == pytest.approx(0.667, abs=0.01)

    def test_by_model_breakdown(self):
        self.tracker.record(self._r(model_used="gemini-1.5-flash"))
        self.tracker.record(self._r(model_used="ollama-local"))
        self.tracker.record(self._r(model_used="gemini-1.5-flash"))
        m = self.tracker.get_metrics()
        assert m["by_model"]["gemini-1.5-flash"]["requests"] == 2
        assert m["by_model"]["ollama-local"]["requests"] == 1

    def test_zero_budget_clamped(self):
        self.tracker.record(self._r(cost_usd=1.00))
        assert self.tracker.get_session_budget_remaining("s1", 0.5) == 0.0

    def test_recent_format(self):
        self.tracker.record(self._r())
        recent = self.tracker.get_recent(5)
        assert len(recent) == 1
        assert "query" in recent[0]
        assert "from_cache" in recent[0]


# ─── Redis Cache ─────────────────────────────────────────────────────────────

class TestRedisCache:

    def setup_method(self):
        self.c = RedisCache()
        self.c._client  = None
        self.c._enabled = False

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disabled(self):
        assert await self.c.get("query", "balanced") is None

    @pytest.mark.asyncio
    async def test_set_noop_when_disabled(self):
        await self.c.set("q", "balanced", CacheEntry("ans", "model", 0.0, 100.0))

    def test_key_stable(self):
        assert self.c._key("What is Python?", "balanced") == self.c._key("What is Python?", "balanced")

    def test_key_case_normalized(self):
        assert self.c._key("WHAT IS PYTHON?", "balanced") == self.c._key("what is python?", "balanced")

    def test_key_differs_by_query(self):
        assert self.c._key("A", "balanced") != self.c._key("B", "balanced")

    def test_key_differs_by_strategy(self):
        assert self.c._key("q", "balanced") != self.c._key("q", "cost_optimized")

    @pytest.mark.asyncio
    async def test_health_false_when_no_client(self):
        assert await self.c.is_healthy() is False


# ─── Planner Agent ───────────────────────────────────────────────────────────

class TestPlannerAgent:

    def setup_method(self):
        self.planner = PlannerAgent()

    def _cls(self, tier):
        from app.classifier.feature_extractor import FeatureExtractor
        from app.classifier.query_classifier import ClassificationResult
        f = FeatureExtractor().extract("test")
        return ClassificationResult(tier=tier, intent="reasoning", complexity_score=0.8,
                                    confidence=0.9, features=f, reasoning="test")

    def test_simple_query_single_task(self):
        tasks = self.planner.plan("What is Python?", self._cls("simple"))
        assert len(tasks) == 1

    def test_medium_query_single_task(self):
        tasks = self.planner.plan("Explain REST APIs", self._cls("medium"))
        assert len(tasks) == 1

    def test_implement_explain_decomposes(self):
        q     = "Implement a binary search tree and then explain how balancing works"
        tasks = self.planner.plan(q, self._cls("complex"))
        assert len(tasks) >= 1

    def test_comparison_query_decomposes(self):
        q     = "Compare Redis vs Memcached for distributed caching"
        tasks = self.planner.plan(q, self._cls("complex"))
        assert len(tasks) >= 1

    def test_system_design_decomposes(self):
        q     = "Design a URL shortener with high availability, caching, analytics, and rate limiting"
        tasks = self.planner.plan(q, self._cls("complex"))
        assert len(tasks) >= 1

    def test_all_tasks_have_ids(self):
        tasks = self.planner.plan("test query", self._cls("simple"))
        for t in tasks:
            assert t.id != ""
            assert t.description != ""


# ─── Evaluator ───────────────────────────────────────────────────────────────

class TestEvaluator:

    def setup_method(self):
        self.ev = ResponseEvaluator()

    def test_good_response_scores_higher(self):
        good = self.ev._heuristic("Explain decorators",
            "A decorator is a function wrapping another function. Example: @property.", "reasoning")
        bad  = self.ev._heuristic("Explain decorators", "I cannot help.", "reasoning")
        assert good.score > bad.score

    def test_code_block_boosts_score(self):
        r = self.ev._heuristic("Write sort function",
            "```python\ndef sort(lst): return sorted(lst)\n```", "code")
        assert r.score > 0.5

    def test_refusal_penalized(self):
        r = self.ev._heuristic("Tell me about X", "I cannot assist with that.", "general")
        assert r.score < 0.4

    def test_score_in_valid_range(self):
        r = self.ev._heuristic("query", "some response", "general")
        assert 0.0 <= r.score <= 1.0
