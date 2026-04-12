"""
Ollama adapter — local models, completely free.

Your installed models (best → pick order):
  llama3.1:8b       — BEST overall (smart, 4.9GB)   ← default
  mistral:latest    — Great reasoning (4.4GB)
  qwen2.5-coder:3b  — Best for code (small, fast)
  llama3.2:latest   — Fast general (2GB)
  phi3:mini         — Lightweight (2.2GB)
  gemma3:1b         — Ultra-fast (815MB)
  tinyllama:latest  — Emergency fallback (637MB)
"""
import time
import logging
from typing import List, Optional
import httpx

from app.models.base_model import BaseLLM, LLMResponse, Message
from config.settings import settings

logger = logging.getLogger(__name__)

# Best model per intent from your installed list
INTENT_MODEL_MAP = {
    "code":      "qwen2.5-coder:3b",
    "reasoning": "llama3.1:8b",
    "factual":   "llama3.2:latest",
    "creative":  "mistral:latest",
    "general":   "llama3.1:8b",
}

# Ordered fallback chain when preferred model missing
FALLBACK_CHAIN = [
    "llama3.1:8b",
    "mistral:latest",
    "llama3.2:latest",
    "phi3:mini",
    "gemma3:1b",
    "tinyllama:latest",
]


class OllamaModel(BaseLLM):

    def __init__(self, model_name: Optional[str] = None, intent: Optional[str] = None):
        if intent and intent in INTENT_MODEL_MAP:
            self.model_name = INTENT_MODEL_MAP[intent]
        else:
            self.model_name = model_name or settings.ollama_default_model

        self.model_id     = "ollama-local"
        self.display_name = f"Ollama / {self.model_name}"
        self.max_context_tokens = 8192
        self._base_url    = settings.ollama_base_url

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.monotonic()

        # Try preferred model, then fall through the chain
        models_to_try = [self.model_name] + [
            m for m in FALLBACK_CHAIN if m != self.model_name
        ]

        for model in models_to_try:
            result = await self._try_model(model, messages, system_prompt, temperature, max_tokens, t0)
            if result is not None:
                return result

        latency_ms = (time.monotonic() - t0) * 1000
        return self._err(
            "Ollama has no working models. Run: ollama pull llama3.2:latest",
            latency_ms
        )

    async def _try_model(
        self, model_name: str,
        messages: List[Message],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        t0: float,
    ) -> Optional[LLMResponse]:
        try:
            formatted = []
            if system_prompt:
                formatted.append({"role": "system", "content": system_prompt})
            for m in messages:
                formatted.append({"role": m.role, "content": m.content})

            payload = {
                "model": model_name,
                "messages": formatted,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 4096,
                },
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)

            if resp.status_code == 404:
                logger.debug(f"Ollama: model '{model_name}' not found, trying next")
                return None  # try next in chain

            if resp.status_code != 200:
                logger.warning(f"Ollama {resp.status_code} for {model_name}")
                return None

            data = resp.json()
            latency_ms = (time.monotonic() - t0) * 1000

            content = data.get("message", {}).get("content", "")
            if not content:
                return None

            self.model_name   = model_name          # update to what actually worked
            self.display_name = f"Ollama / {model_name}"

            return LLMResponse(
                content=content,
                model_id=self.model_id,
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency_ms=round(latency_ms, 1),
                cost_usd=0.0, success=True, raw=data,
            )

        except httpx.ConnectError:
            latency_ms = (time.monotonic() - t0) * 1000
            # Only log once for connection errors
            return self._err(
                f"Ollama not running at {self._base_url}. Start with: ollama serve",
                latency_ms
            )
        except httpx.TimeoutException:
            logger.warning(f"Ollama timeout on {model_name}, trying next")
            return None
        except Exception as e:
            logger.error(f"Ollama error on {model_name}: {e}")
            return None

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                if r.status_code != 200:
                    return False
                # Check at least one model is available
                models = r.json().get("models", [])
                return len(models) > 0
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
