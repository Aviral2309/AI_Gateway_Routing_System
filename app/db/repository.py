from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import RequestLog, SessionBudget


class RequestRepository:

    def __init__(self, session: AsyncSession):
        self.db = session

    async def save(self, **kwargs) -> RequestLog:
        log = RequestLog(**kwargs, created_at=datetime.utcnow())
        self.db.add(log)
        await self.db.flush()
        return log

    async def recent(self, limit: int = 20) -> List[RequestLog]:
        r = await self.db.execute(
            select(RequestLog).order_by(RequestLog.created_at.desc()).limit(limit)
        )
        return list(r.scalars().all())

    async def summary_stats(self) -> dict:
        total  = (await self.db.execute(select(func.count(RequestLog.id)))).scalar() or 0
        cost   = (await self.db.execute(select(func.sum(RequestLog.cost_usd)))).scalar() or 0.0
        lat    = (await self.db.execute(select(func.avg(RequestLog.latency_ms)))).scalar() or 0.0
        qual   = (await self.db.execute(select(func.avg(RequestLog.quality_score)))).scalar() or 0.0
        succ   = (await self.db.execute(
            select(func.count(RequestLog.id)).where(RequestLog.success == True)
        )).scalar() or 0
        cached = (await self.db.execute(
            select(func.count(RequestLog.id)).where(RequestLog.from_cache == True)
        )).scalar() or 0
        decomp = (await self.db.execute(
            select(func.count(RequestLog.id)).where(RequestLog.was_decomposed == True)
        )).scalar() or 0

        return {
            "total_requests":   total,
            "total_cost_usd":   round(cost, 6),
            "avg_latency_ms":   round(lat, 1),
            "avg_quality_score": round(qual, 3),
            "success_rate":     round(succ / max(1, total), 3),
            "cache_hit_rate":   round(cached / max(1, total), 3),
            "agentic_decomposition_rate": round(decomp / max(1, total), 3),
        }

    async def by_model(self) -> List[dict]:
        r = await self.db.execute(
            select(
                RequestLog.model_used,
                func.count(RequestLog.id).label("requests"),
                func.sum(RequestLog.cost_usd).label("total_cost"),
                func.avg(RequestLog.latency_ms).label("avg_latency"),
                func.avg(RequestLog.quality_score).label("avg_quality"),
                func.sum(RequestLog.input_tokens + RequestLog.output_tokens).label("total_tokens"),
            ).group_by(RequestLog.model_used)
        )
        return [
            {
                "model":       row.model_used,
                "requests":    row.requests,
                "total_cost":  round(row.total_cost or 0, 6),
                "avg_latency": round(row.avg_latency or 0, 1),
                "avg_quality": round(row.avg_quality or 0, 3) if row.avg_quality else None,
                "total_tokens": row.total_tokens or 0,
            }
            for row in r.all()
        ]

    async def by_tier(self) -> dict:
        r = await self.db.execute(
            select(RequestLog.tier, func.count(RequestLog.id)).group_by(RequestLog.tier)
        )
        return {row[0]: row[1] for row in r.all()}

    async def by_intent(self) -> dict:
        r = await self.db.execute(
            select(RequestLog.intent, func.count(RequestLog.id)).group_by(RequestLog.intent)
        )
        return {row[0]: row[1] for row in r.all()}

    async def cost_over_time(self, hours: int = 24) -> List[dict]:
        since = datetime.utcnow() - timedelta(hours=hours)
        r = await self.db.execute(
            select(
                func.date_trunc("hour", RequestLog.created_at).label("hour"),
                func.sum(RequestLog.cost_usd).label("cost"),
                func.count(RequestLog.id).label("requests"),
            )
            .where(RequestLog.created_at >= since)
            .group_by(text("hour"))
            .order_by(text("hour"))
        )
        return [{"hour": str(row.hour), "cost": round(row.cost or 0, 6), "requests": row.requests}
                for row in r.all()]


class SessionRepository:

    def __init__(self, session: AsyncSession):
        self.db = session

    async def get_spent(self, session_id: str) -> float:
        r = await self.db.execute(
            select(SessionBudget.total_spent).where(SessionBudget.session_id == session_id)
        )
        return r.scalar() or 0.0

    async def add_cost(self, session_id: str, cost: float):
        r = await self.db.execute(
            select(SessionBudget).where(SessionBudget.session_id == session_id)
        )
        sb = r.scalar_one_or_none()
        if sb:
            sb.total_spent   += cost
            sb.request_count += 1
            sb.updated_at     = datetime.utcnow()
        else:
            self.db.add(SessionBudget(
                session_id=session_id, total_spent=cost,
                request_count=1, created_at=datetime.utcnow(),
            ))
        await self.db.flush()
