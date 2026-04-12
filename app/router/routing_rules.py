from dataclasses import dataclass
from typing import List
from enum import Enum
from app.classifier.query_classifier import ClassificationResult


class RoutingStrategy(str, Enum):
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_FIRST  = "quality_first"
    BALANCED       = "balanced"
    LOCAL_FIRST    = "local_first"


@dataclass
class RoutingDecision:
    primary_model_id: str
    fallback_model_ids: List[str]
    strategy: RoutingStrategy
    reason: str
    estimated_cost_usd: float
    skip_evaluation: bool


# ── Model IDs ─────────────────────────────────────────────────────────────────
FLASH      = "gemini-2.5-flash"
FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
OLLAMA     = "ollama-local"

# ── Routing table ─────────────────────────────────────────────────────────────
# Simple → Gemini Flash Lite (cheapest, fastest)
# Medium → Gemini Flash (balanced)
# Complex → Ollama (powerful local, free) → Flash fallback
ROUTING_TABLE = {
    ("simple",  "factual"):   [FLASH_LITE, OLLAMA],
    ("simple",  "creative"):  [FLASH_LITE, OLLAMA],
    ("simple",  "general"):   [FLASH_LITE, OLLAMA],
    ("simple",  "code"):      [FLASH_LITE, OLLAMA],
    ("simple",  "reasoning"): [FLASH_LITE, OLLAMA],
    ("medium",  "factual"):   [FLASH, OLLAMA],
    ("medium",  "creative"):  [FLASH, OLLAMA],
    ("medium",  "general"):   [FLASH, OLLAMA],
    ("medium",  "code"):      [OLLAMA, FLASH],       # Ollama great for code
    ("medium",  "reasoning"): [FLASH, OLLAMA],
    ("complex", "factual"):   [OLLAMA, FLASH],
    ("complex", "creative"):  [FLASH, OLLAMA],
    ("complex", "general"):   [OLLAMA, FLASH],
    ("complex", "code"):      [OLLAMA, FLASH],       # llama3.1:8b / qwen for complex code
    ("complex", "reasoning"): [OLLAMA, FLASH],
}

# ── Cost estimates (per 1K tokens, USD) ───────────────────────────────────────
COST_TABLE = {
    FLASH:      {"input": 0.00015,  "output": 0.0006},
    FLASH_LITE: {"input": 0.000075, "output": 0.0003},
    OLLAMA:     {"input": 0.0,      "output": 0.0},
}

TOKEN_ESTIMATE = {
    "simple":  {"input": 50,  "output": 100},
    "medium":  {"input": 200, "output": 400},
    "complex": {"input": 500, "output": 800},
}


class RoutingRules:

    def decide(
        self,
        cls: ClassificationResult,
        budget_remaining_usd: float,
        available_models: List[str],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    ) -> RoutingDecision:

        preferred = list(ROUTING_TABLE.get(
            (cls.tier, cls.intent),
            ROUTING_TABLE.get((cls.tier, "general"), [FLASH, OLLAMA])
        ))

        # Apply strategy overrides
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            preferred.sort(key=lambda m: self._cost_rank(m))
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            preferred.sort(key=lambda m: -self._quality_rank(m))
        elif strategy == RoutingStrategy.LOCAL_FIRST:
            preferred.sort(key=lambda m: 0 if "ollama" in m else 1)

        # Filter to available only
        avail     = [m for m in preferred if m in available_models]
        in_budget = [m for m in avail if self._est_cost(m, cls.tier) <= budget_remaining_usd]

        if not in_budget:
            in_budget = [OLLAMA] if OLLAMA in available_models else avail

        primary   = in_budget[0] if in_budget else (available_models[0] if available_models else FLASH)
        fallbacks = [m for m in (in_budget + avail) if m != primary][:2]

        return RoutingDecision(
            primary_model_id=primary,
            fallback_model_ids=fallbacks,
            strategy=strategy,
            reason=f"tier={cls.tier} intent={cls.intent} → {primary} [{strategy.value}]",
            estimated_cost_usd=self._est_cost(primary, cls.tier),
            skip_evaluation=(cls.tier == "simple" and cls.intent == "factual"),
        )

    def _est_cost(self, model_id: str, tier: str) -> float:
        c = COST_TABLE.get(model_id, {"input": 0.00015, "output": 0.0006})
        t = TOKEN_ESTIMATE.get(tier, {"input": 200, "output": 400})
        return t["input"] / 1000 * c["input"] + t["output"] / 1000 * c["output"]

    def _cost_rank(self, m: str) -> float:
        if "ollama"     in m: return 0.0
        if "lite"       in m: return 1.0
        if "flash"      in m: return 2.0
        if "pro"        in m: return 3.0
        return 2.0

    def _quality_rank(self, m: str) -> float:
        if "pro"        in m: return 4.0
        if "ollama"     in m: return 3.0   # llama3.1:8b is very capable
        if "flash"      in m and "lite" not in m: return 2.5
        if "lite"       in m: return 1.5
        return 2.0
