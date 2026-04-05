from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class Message:
    role: str       # user | assistant | system
    content: str


@dataclass
class LLMResponse:
    content: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    raw: Optional[dict] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLM(ABC):
    model_id: str
    display_name: str
    supports_system_prompt: bool = True
    max_context_tokens: int = 8000

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    def _make_error_response(self, error: str, latency_ms: float = 0.0) -> LLMResponse:
        return LLMResponse(
            content="", model_id=self.model_id,
            input_tokens=0, output_tokens=0,
            latency_ms=latency_ms, cost_usd=0.0,
            success=False, error=error,
        )
