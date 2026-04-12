"""
Gemini adapter — supports gemini-2.5-flash and gemini-2.5-flash-lite.
Every error is caught and returned as a failed LLMResponse — never raises.
The router will automatically fallback to Ollama on any Gemini failure.
"""
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
        self.model_id = model_id
        self.display_name = self._fmt_name(model_id)
        self.max_context_tokens = 1_000_000
        self._costs = MODEL_COSTS.get(model_id, {"input": 0.00015, "output": 0.0006})

    def _fmt_name(self, mid: str) -> str:
        if "lite" in mid:
            return "Gemini 2.5 Flash Lite"
        if "flash" in mid:
            return "Gemini 2.5 Flash"
        if "pro" in mid:
            return "Gemini 2.5 Pro"
        return mid

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:

        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            return self._err("No Gemini API key — using Ollama instead")

        t0 = time.monotonic()
        try:
            contents = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

            payload: dict = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            # Disable thinking budget for latency (Gemini 2.5 specific)
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            url = f"{GEMINI_BASE}/{self.model_id}:generateContent"

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url, json=payload,
                    params={"key": settings.gemini_api_key}
                )

            latency_ms = (time.monotonic() - t0) * 1000

            # Handle every HTTP error — never let it propagate
            if resp.status_code == 400:
                detail = resp.json().get("error", {}).get("message", "Bad request")
                logger.warning(f"Gemini 400: {detail}")
                return self._err(f"Gemini 400: {detail}", latency_ms)
            if resp.status_code == 401 or resp.status_code == 403:
                return self._err("Gemini: invalid API key or quota exceeded", latency_ms)
            if resp.status_code == 404:
                return self._err(f"Gemini: model '{self.model_id}' not available", latency_ms)
            if resp.status_code == 429:
                return self._err("Gemini: rate limit hit — falling back", latency_ms)
            if resp.status_code >= 500:
                logger.error(f"Gemini server error {resp.status_code}")
                return self._err(f"Gemini server error {resp.status_code} — falling back", latency_ms)

            data = resp.json()

            candidates = data.get("candidates", [])
            if not candidates:
                reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                return self._err(f"Gemini blocked response: {reason}", latency_ms)

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return self._err("Gemini returned empty content", latency_ms)

            text  = "".join(p.get("text", "") for p in parts)
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

        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - t0) * 1000
            return self._err(f"Gemini timed out after {latency_ms:.0f}ms", latency_ms)
        except httpx.ConnectError:
            latency_ms = (time.monotonic() - t0) * 1000
            return self._err("Gemini unreachable — check internet connection", latency_ms)
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.error(f"Gemini unexpected: {e}")
            return self._err(str(e)[:200], latency_ms)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens  / 1000 * self._costs["input"] +
            output_tokens / 1000 * self._costs["output"]
        )

    async def health_check(self) -> bool:
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{GEMINI_BASE}/{self.model_id}:generateContent",
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                          "generationConfig": {"maxOutputTokens": 5, "thinkingConfig": {"thinkingBudget": 0}}},
                    params={"key": settings.gemini_api_key}
                )
                return resp.status_code == 200
        except Exception:
            return False
