"""
Nexus LLM Router — Main Application
"""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.router.llm_router import LLMRouter
from app.models.base_model import Message
from app.router.routing_rules import RoutingStrategy
from app.cache.redis_cache import cache
from app.middleware.rate_limiter import RateLimitMiddleware
from config.settings import settings

logging.basicConfig(level=settings.log_level)
logger = structlog.get_logger()

router_instance: Optional[LLMRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_instance
    logger.info("=== Nexus LLM Router starting ===")

    try:
        from app.db.session import init_db
        await init_db()
        logger.info("PostgreSQL: tables ready")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable: {e} — memory fallback active")

    await cache.connect()

    router_instance = LLMRouter()
    logger.info("Router + Agents initialized")
    yield

    await cache.disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Nexus LLM Router",
    description="Intelligent multi-LLM orchestration with ML classification, Redis caching, PostgreSQL analytics, and Agentic AI pipeline.",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str                  = Field(..., min_length=1, max_length=8000)
    history: List[ChatMessage]  = Field(default_factory=list)
    system_prompt: Optional[str]= None
    session_id: Optional[str]   = None
    strategy: RoutingStrategy   = RoutingStrategy.BALANCED
    temperature: float          = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int             = Field(default=1024, ge=1, le=4096)
    use_agents: Optional[bool]  = None   # None = auto (complex → agents)


class ClassifyRequest(BaseModel):
    query: str
    context_turns: int = 0


class ChatResponse(BaseModel):
    content: str
    model_used: str
    session_id: str
    from_cache: bool
    was_decomposed: bool

    tier: str
    intent: str
    complexity_score: float
    classification_confidence: float

    routing_strategy: str
    routing_reason: str

    latency_ms: float
    cost_usd: float
    total_tokens: int
    attempts: int

    quality_score: Optional[float]
    quality_reason: Optional[str]

    success: bool
    error: Optional[str]

    agent_steps: Optional[list]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main endpoint. Routes query through full pipeline:
    classify → cache → route → (agents if complex) → evaluate → store.
    """
    session_id = req.session_id or str(uuid.uuid4())
    history    = [Message(role=m.role, content=m.content) for m in req.history]

    result = await router_instance.route(
        query=req.query, history=history, system_prompt=req.system_prompt,
        strategy=req.strategy, session_id=session_id,
        temperature=req.temperature, max_tokens=req.max_tokens,
        use_agents=req.use_agents,
    )

    agent_steps = None
    if result.agent_trace:
        agent_steps = result.agent_trace.steps

    return ChatResponse(
        content=result.content,
        model_used=result.model_used,
        session_id=session_id,
        from_cache=result.from_cache,
        was_decomposed=result.agent_trace.was_decomposed if result.agent_trace else False,
        tier=result.classification.tier,
        intent=result.classification.intent,
        complexity_score=result.classification.complexity_score,
        classification_confidence=result.classification.confidence,
        routing_strategy=result.decision.strategy.value,
        routing_reason=result.decision.reason,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        total_tokens=result.total_tokens,
        attempts=result.attempts,
        quality_score=result.evaluation.score if result.evaluation else None,
        quality_reason=result.evaluation.reason if result.evaluation else None,
        success=result.success,
        error=result.error,
        agent_steps=agent_steps,
    )


@app.post("/classify")
async def classify(req: ClassifyRequest):
    """Classify a query without calling any LLM — for debugging routing logic."""
    r = router_instance.classifier.classify(req.query, req.context_turns)
    return {
        "tier":             r.tier,
        "intent":           r.intent,
        "complexity_score": r.complexity_score,
        "confidence":       r.confidence,
        "reasoning":        r.reasoning,
        "will_use_agents":  r.tier == "complex" and settings.enable_agents,
        "features": {
            "token_count":             r.features.token_count,
            "has_code_block":          r.features.has_code_block,
            "has_math":                r.features.has_math,
            "clause_depth":            r.features.clause_depth,
            "code_keyword_score":      round(r.features.code_keyword_score, 3),
            "reasoning_keyword_score": round(r.features.reasoning_keyword_score, 3),
            "factual_keyword_score":   round(r.features.factual_keyword_score, 3),
        },
    }


@app.get("/metrics")
async def metrics(source: str = Query(default="auto", description="auto | db | memory")):
    """
    Analytics. source=db returns full PostgreSQL history.
    source=memory returns current-session in-memory stats (always fast).
    """
    if source == "db":
        data = await router_instance.tracker.get_metrics_db()
    elif source == "memory":
        data = router_instance.tracker.get_metrics()
    else:
        data = await router_instance.tracker.get_metrics_db()

    return {
        "metrics":  data,
        "cache":    await cache.get_stats(),
        "recent":   router_instance.tracker.get_recent(20),
    }


@app.get("/analytics/cost")
async def cost_analytics(hours: int = Query(default=24, ge=1, le=168)):
    """Hourly cost breakdown for the last N hours (from PostgreSQL)."""
    try:
        from app.db.session import get_db
        from app.db.repository import RequestRepository
        async with get_db() as db:
            data = await RequestRepository(db).cost_over_time(hours)
        return {"hours": hours, "data": data}
    except Exception as e:
        raise HTTPException(503, detail=f"DB unavailable: {e}")


@app.get("/analytics/requests")
async def request_log(limit: int = Query(default=20, ge=1, le=100)):
    """Recent request log from PostgreSQL."""
    try:
        from app.db.session import get_db
        from app.db.repository import RequestRepository
        async with get_db() as db:
            logs = await RequestRepository(db).recent(limit)
        return {"requests": [
            {
                "id":            log.id,
                "session_id":    log.session_id,
                "tier":          log.tier,
                "intent":        log.intent,
                "model_used":    log.model_used,
                "latency_ms":    log.latency_ms,
                "cost_usd":      log.cost_usd,
                "quality_score": log.quality_score,
                "success":       log.success,
                "from_cache":    log.from_cache,
                "was_decomposed": log.was_decomposed,
                "agent_steps":   log.agent_steps,
                "created_at":    str(log.created_at),
                "query":         log.query[:100],
            }
            for log in logs
        ]}
    except Exception as e:
        raise HTTPException(503, detail=f"DB unavailable: {e}")


@app.get("/health")
async def health():
    """Full system health check."""
    model_health = {}
    for mid, model in router_instance._models.items():
        try:
            ok = await model.health_check()
            model_health[mid] = "ok" if ok else "unreachable"
        except Exception as e:
            model_health[mid] = f"error: {str(e)[:50]}"

    db_ok = False
    try:
        from app.db.session import get_db
        from sqlalchemy import text
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    redis_ok = await cache.is_healthy()

    return {
        "status":     "healthy" if any(v == "ok" for v in model_health.values()) else "degraded",
        "models":     model_health,
        "postgresql": "ok" if db_ok else "unavailable (memory fallback active)",
        "redis":      "ok" if redis_ok else "unavailable (memory fallback active)",
        "agents":     "enabled" if settings.enable_agents else "disabled",
        "config": {
            "version":               settings.app_version,
            "budget_per_session":    f"${settings.default_budget_usd}",
            "cache_ttl":             f"{settings.cache_ttl_seconds}s",
            "rate_limit":            f"{settings.rate_limit_requests} req/{settings.rate_limit_window_seconds}s",
            "evaluation":            settings.enable_evaluation,
            "agent_max_steps":       settings.agent_max_steps,
        },
    }


@app.get("/models")
async def list_models():
    """List all registered LLM adapters and their current health."""
    result = []
    for mid, model in router_instance._models.items():
        ok = await model.health_check()
        result.append({
            "id":           mid,
            "display_name": model.display_name,
            "status":       "ok" if ok else "unreachable",
            "context_tokens": model.max_context_tokens,
        })
    return {"models": result}


@app.delete("/cache")
async def clear_cache():
    """Flush all Redis cached responses."""
    await cache.flush()
    return {"message": "Cache cleared"}
