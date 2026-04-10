import time
import logging
from typing import List, Optional
import os

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

from app.models.base_model import BaseLLM, LLMResponse, Message
from config.settings import MODEL_COSTS

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class GeminiModel(BaseLLM):

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        self.model_id = model_id
        self.display_name = f"Gemini {model_id.split('-')[-1].title()}"
        self.max_context_tokens = 1_000_000  # updated for newer models
        self._costs = MODEL_COSTS.get(model_id, {"input": 0.0001, "output": 0.0003})

        try:
            self.model = genai.GenerativeModel(self.model_id)
        except Exception as e:
            logger.error(f"Model init failed: {e}")
            self.model = None

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:

        if not self.model:
            return self._make_error_response("Model not initialized")

        t0 = time.monotonic()

        try:
            # Convert messages to prompt
            prompt = ""

            if system_prompt:
                prompt += f"System: {system_prompt}\n"

            for msg in messages:
                role = "User" if msg.role != "assistant" else "Assistant"
                prompt += f"{role}: {msg.content}\n"

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )

            latency_ms = (time.monotonic() - t0) * 1000

            text = response.text if hasattr(response, "text") else ""

            # Token usage (approx — SDK doesn't always return exact)
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(text.split()) * 1.3

            return LLMResponse(
                content=text,
                model_id=self.model_id,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                latency_ms=round(latency_ms, 1),
                cost_usd=round(self.estimate_cost(input_tokens, output_tokens), 8),
                success=True,
                raw={"response": text},
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            latency_ms = (time.monotonic() - t0) * 1000
            logger.error(f"Gemini error: {e}")

            return self._make_error_response(str(e)[:200], latency_ms)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000 * self._costs["input"]) +
            (output_tokens / 1000 * self._costs["output"])
        )

    async def health_check(self) -> bool:
        try:
            r = await self.complete([Message(role="user", content="hi")], max_tokens=5)
            return r.success
        except Exception:
            return False