import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    session_id: str
    query: str
    response: Optional[str]
    model_used: str
    tier: str
    intent: str
    complexity_score: float
    routing_strategy: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    retry_count: int
    from_cache: bool = False
    was_decomposed: bool = False
    agent_steps: int = 1
    quality_score: Optional[float] = None
    quality_reason: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class Tracker:
    """Dual-write: PostgreSQL (persistent) + in-memory (fast reads)."""

    def __init__(self):
        self._records: List[RequestRecord]      = []
        self._session_costs: Dict[str, float]   = defaultdict(float)
        self._model_stats: Dict[str, dict]      = defaultdict(lambda: {
            "requests": 0, "success": 0, "cost": 0.0,
            "latency": 0.0, "quality": [],
        })
        self._tier_counts:   Dict[str, int]     = defaultdict(int)
        self._intent_counts: Dict[str, int]     = defaultdict(int)
        self._db_ok = True

    async def record_async(self, r: RequestRecord):
        self._write_memory(r)
        await self._write_db(r)

    def record(self, r: RequestRecord):
        self._write_memory(r)

    def _write_memory(self, r: RequestRecord):
        self._records.append(r)
        self._session_costs[r.session_id] += r.cost_usd
        s = self._model_stats[r.model_used]
        s["requests"]  += 1
        s["cost"]      += r.cost_usd
        s["latency"]   += r.latency_ms
        if r.success:           s["success"]  += 1
        if r.quality_score:     s["quality"].append(r.quality_score)
        self._tier_counts[r.tier]     += 1
        self._intent_counts[r.intent] += 1

    async def _write_db(self, r: RequestRecord):
        if not self._db_ok:
            return
        try:
            from app.db.session import get_db
            from app.db.repository import RequestRepository, SessionRepository
            async with get_db() as db:
                await RequestRepository(db).save(
                    session_id=r.session_id, query=r.query, response=r.response,
                    tier=r.tier, intent=r.intent, complexity_score=r.complexity_score,
                    model_used=r.model_used, routing_strategy=r.routing_strategy,
                    was_decomposed=r.was_decomposed, agent_steps=r.agent_steps,
                    input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                    latency_ms=r.latency_ms, cost_usd=r.cost_usd,
                    quality_score=r.quality_score, quality_reason=r.quality_reason,
                    success=r.success, retry_count=r.retry_count,
                    from_cache=r.from_cache, error_message=r.error_message,
                )
                await SessionRepository(db).add_cost(r.session_id, r.cost_usd)
        except Exception as e:
            logger.warning(f"DB write failed (memory fallback): {e}")
            self._db_ok = False

    def get_session_budget_remaining(self, session_id: str, budget: float) -> float:
        return max(0.0, budget - self._session_costs.get(session_id, 0.0))

    async def get_session_budget_remaining_db(self, session_id: str, budget: float) -> float:
        try:
            from app.db.session import get_db
            from app.db.repository import SessionRepository
            async with get_db() as db:
                spent = await SessionRepository(db).get_spent(session_id)
                return max(0.0, budget - spent)
        except Exception:
            return self.get_session_budget_remaining(session_id, budget)

    def get_metrics(self) -> dict:
        records   = self._records
        total     = len(records)
        succ      = sum(1 for r in records if r.success)
        lats      = [r.latency_ms for r in records if r.success]
        qual      = [r.quality_score for r in records if r.quality_score]
        cached    = sum(1 for r in records if r.from_cache)
        decomposed = sum(1 for r in records if r.was_decomposed)

        by_model = {}
        for mid, s in self._model_stats.items():
            qs = s["quality"]
            by_model[mid] = {
                "requests":     s["requests"],
                "success_rate": round(s["success"] / max(1, s["requests"]), 3),
                "avg_latency_ms": round(s["latency"] / max(1, s["requests"]), 1),
                "total_cost_usd": round(s["cost"], 6),
                "avg_quality":  round(sum(qs)/len(qs), 3) if qs else None,
            }

        return {
            "source": "memory",
            "summary": {
                "total_requests":   total,
                "success_rate":     round(succ / max(1, total), 3),
                "total_cost_usd":   round(sum(r.cost_usd for r in records), 6),
                "avg_latency_ms":   round(sum(lats) / max(1, len(lats)), 1),
                "p95_latency_ms":   self._p(lats, 95),
                "avg_quality":      round(sum(qual)/len(qual), 3) if qual else None,
                "cache_hit_rate":   round(cached / max(1, total), 3),
                "agentic_rate":     round(decomposed / max(1, total), 3),
            },
            "by_model":  by_model,
            "by_tier":   dict(self._tier_counts),
            "by_intent": dict(self._intent_counts),
        }

    async def get_metrics_db(self) -> dict:
        try:
            from app.db.session import get_db
            from app.db.repository import RequestRepository
            async with get_db() as db:
                repo = RequestRepository(db)
                return {
                    "source":         "database",
                    "summary":        await repo.summary_stats(),
                    "by_model":       await repo.by_model(),
                    "by_tier":        await repo.by_tier(),
                    "by_intent":      await repo.by_intent(),
                    "cost_over_time": await repo.cost_over_time(24),
                }
        except Exception as e:
            logger.warning(f"DB metrics failed: {e}")
            return self.get_metrics()

    def get_recent(self, n: int = 20) -> List[dict]:
        return [
            {
                "session_id":    r.session_id,
                "tier":          r.tier,
                "intent":        r.intent,
                "model_used":    r.model_used,
                "latency_ms":    r.latency_ms,
                "cost_usd":      r.cost_usd,
                "success":       r.success,
                "from_cache":    r.from_cache,
                "was_decomposed": r.was_decomposed,
                "quality_score": r.quality_score,
                "query":         r.query[:80] + "..." if len(r.query) > 80 else r.query,
            }
            for r in self._records[-n:][::-1]
        ]

    def _p(self, data: List[float], p: int) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return round(s[min(int(len(s) * p / 100), len(s)-1)], 1)
