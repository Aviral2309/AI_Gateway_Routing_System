import time
import logging
from typing import List, Optional
import httpx

from app.models.base_model import BaseLLM, LLMResponse, Message
from config.settings import settings, MODEL_COSTS

logger = logging.getLogger(__name__)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiModel(BaseLLM):

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        self.model_id     = model_id
        self.display_name = f"Gemini {model_id.split('-')[-1].title()}"
        self.max_context_tokens = 128_000
        self._costs = MODEL_COSTS.get(model_id, {"input": 0.0001, "output": 0.0003})

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not settings.gemini_api_key:
            return self._make_error_response("GEMINI_API_KEY not set")

        t0 = time.monotonic()
        try:
            contents = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

            payload: dict = {
                "contents": contents,
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            url = f"{GEMINI_BASE}/{self.model_id}:generateContent"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, params={"key": settings.gemini_api_key})
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.monotonic() - t0) * 1000
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            inp   = usage.get("promptTokenCount", 0)
            out   = usage.get("candidatesTokenCount", 0)

            return LLMResponse(
                content=text, model_id=self.model_id,
                input_tokens=inp, output_tokens=out,
                latency_ms=round(latency_ms, 1),
                cost_usd=round(self.estimate_cost(inp, out), 8),
                success=True, raw=data,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.error(f"Gemini error: {e}")
            return self._make_error_response(str(e)[:200], latency_ms)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000 * self._costs["input"] +
                output_tokens / 1000 * self._costs["output"])

    async def health_check(self) -> bool:
        if not settings.gemini_api_key:
            return False
        try:
            r = await self.complete([Message(role="user", content="hi")], max_tokens=5)
            return r.success
        except Exception:
            return False
