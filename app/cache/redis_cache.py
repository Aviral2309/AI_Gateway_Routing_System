import hashlib
import json
import logging
from typing import Optional
import redis.asyncio as aioredis
from config.settings import settings

logger = logging.getLogger(__name__)


class CacheEntry:
    def __init__(self, content: str, model_used: str, cost_usd: float, latency_ms: float):
        self.content    = content
        self.model_used = model_used
        self.cost_usd   = cost_usd
        self.latency_ms = latency_ms
        self.from_cache = True


class RedisCache:

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._enabled = settings.cache_enabled
        self._ttl     = settings.cache_ttl_seconds

    async def connect(self):
        if not self._enabled:
            return
        try:
            self._client = aioredis.from_url(
                settings.redis_url, encoding="utf-8",
                decode_responses=True, socket_connect_timeout=3,
            )
            await self._client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e} — cache disabled")
            self._client  = None
            self._enabled = False

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    async def get(self, query: str, strategy: str) -> Optional[CacheEntry]:
        if not self._client:
            return None
        try:
            data = await self._client.get(self._key(query, strategy))
            if data:
                obj = json.loads(data)
                logger.info("Cache HIT")
                return CacheEntry(**obj)
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
        return None

    async def set(self, query: str, strategy: str, entry: CacheEntry):
        if not self._client:
            return
        try:
            await self._client.setex(
                self._key(query, strategy), self._ttl,
                json.dumps({"content": entry.content, "model_used": entry.model_used,
                            "cost_usd": entry.cost_usd, "latency_ms": entry.latency_ms})
            )
        except Exception as e:
            logger.debug(f"Cache set error: {e}")

    async def flush(self):
        if self._client:
            await self._client.flushdb()

    async def get_stats(self) -> dict:
        if not self._client:
            return {"enabled": False}
        try:
            info = await self._client.info("stats")
            hits   = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            return {
                "enabled":   True,
                "hits":      hits,
                "misses":    misses,
                "hit_rate":  round(hits / max(1, hits + misses), 3),
                "ttl_seconds": self._ttl,
            }
        except Exception:
            return {"enabled": True, "error": "stats unavailable"}

    async def is_healthy(self) -> bool:
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    def _key(self, query: str, strategy: str) -> str:
        raw = f"{query.lower().strip()}::{strategy}"
        return "nexus:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


cache = RedisCache()
