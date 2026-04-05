import time
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from config.settings import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, max_requests: int = None, window_seconds: int = None):
        super().__init__(app)
        self.max  = max_requests   or settings.rate_limit_requests
        self.win  = window_seconds or settings.rate_limit_window_seconds
        self._mem: dict = {}

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in ("/chat", "/agent"):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        if not await self._allow(ip):
            raise HTTPException(status_code=429, detail={
                "error": "rate_limit_exceeded",
                "message": f"Max {self.max} requests per {self.win}s",
            })
        return await call_next(request)

    async def _allow(self, ip: str) -> bool:
        from app.cache.redis_cache import cache
        if cache._client:
            return await self._redis(ip)
        return self._memory(ip)

    async def _redis(self, ip: str) -> bool:
        from app.cache.redis_cache import cache
        try:
            key = f"rl:{ip}"
            now = time.time()
            pipe = cache._client.pipeline()
            pipe.zremrangebyscore(key, 0, now - self.win)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.win)
            results = await pipe.execute()
            return results[2] <= self.max
        except Exception:
            return True

    def _memory(self, ip: str) -> bool:
        now   = time.time()
        start = now - self.win
        ts    = [t for t in self._mem.get(ip, []) if t > start]
        ts.append(now)
        self._mem[ip] = ts
        return len(ts) <= self.max
