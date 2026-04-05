import time
import logging
from typing import List, Optional
import httpx

from app.models.base_model import BaseLLM, LLMResponse, Message
from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaModel(BaseLLM):

    def __init__(self, model_name: Optional[str] = None):
        self.model_name   = model_name or settings.ollama_default_model
        self.model_id     = f"ollama-local"
        self.display_name = f"Ollama/{self.model_name}"
        self.max_context_tokens = 8000
        self._base_url    = settings.ollama_base_url

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.monotonic()
        try:
            formatted = []
            if system_prompt:
                formatted.append({"role": "system", "content": system_prompt})
            for m in messages:
                formatted.append({"role": m.role, "content": m.content})

            payload = {
                "model": self.model_name,
                "messages": formatted,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.monotonic() - t0) * 1000
            return LLMResponse(
                content=data["message"]["content"],
                model_id=self.model_id,
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency_ms=round(latency_ms, 1),
                cost_usd=0.0, success=True, raw=data,
            )
        except httpx.ConnectError:
            latency_ms = (time.monotonic() - t0) * 1000
            return self._make_error_response(f"Ollama not running at {self._base_url}", latency_ms)
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            return self._make_error_response(str(e)[:200], latency_ms)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
