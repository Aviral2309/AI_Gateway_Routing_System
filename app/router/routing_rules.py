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


ROUTING_TABLE = {
    ("simple",  "factual"):   ["gemini-1.5-flash", "ollama-local"],
    ("simple",  "creative"):  ["gemini-1.5-flash", "ollama-local"],
    ("simple",  "general"):   ["gemini-1.5-flash", "ollama-local"],
    ("simple",  "code"):      ["gemini-1.5-flash", "ollama-local"],
    ("simple",  "reasoning"): ["gemini-1.5-flash", "ollama-local"],
    ("medium",  "factual"):   ["gemini-1.5-flash", "ollama-local"],
    ("medium",  "creative"):  ["gemini-1.5-flash", "ollama-local"],
    ("medium",  "general"):   ["gemini-1.5-flash", "ollama-local"],
    ("medium",  "code"):      ["ollama-local", "gemini-1.5-flash"],
    ("medium",  "reasoning"): ["gemini-1.5-flash", "ollama-local"],
    ("complex", "factual"):   ["ollama-local", "gemini-1.5-flash"],
    ("complex", "creative"):  ["gemini-1.5-flash", "ollama-local"],
    ("complex", "general"):   ["ollama-local", "gemini-1.5-flash"],
    ("complex", "code"):      ["ollama-local", "gemini-1.5-flash"],
    ("complex", "reasoning"): ["ollama-local", "gemini-1.5-flash"],
}

COST_TABLE = {
    "gemini-1.5-flash": {"input": 0.000035, "output": 0.000105},
    "gemini-1.5-pro":   {"input": 0.00035,  "output": 0.00105},
    "ollama-local":     {"input": 0.0,       "output": 0.0},
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
            ROUTING_TABLE.get((cls.tier, "general"), ["gemini-1.5-flash", "ollama-local"])
        ))

        if strategy == RoutingStrategy.COST_OPTIMIZED:
            preferred.sort(key=lambda m: self._cost_rank(m))
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            preferred.sort(key=lambda m: -self._quality_rank(m))
        elif strategy == RoutingStrategy.LOCAL_FIRST:
            preferred.sort(key=lambda m: 0 if "ollama" in m else 1)

        avail = [m for m in preferred if m in available_models]
        in_budget = [m for m in avail if self._est_cost(m, cls.tier) <= budget_remaining_usd]

        if not in_budget:
            in_budget = ["ollama-local"] if "ollama-local" in available_models else avail

        primary   = in_budget[0] if in_budget else (available_models[0] if available_models else "gemini-1.5-flash")
        fallbacks = [m for m in (in_budget + avail) if m != primary][:2]

        return RoutingDecision(
            primary_model_id=primary,
            fallback_model_ids=fallbacks,
            strategy=strategy,
            reason=f"tier={cls.tier} intent={cls.intent} model={primary} strategy={strategy.value}",
            estimated_cost_usd=self._est_cost(primary, cls.tier),
            skip_evaluation=(cls.tier == "simple" and cls.intent == "factual"),
        )

    def _est_cost(self, model_id: str, tier: str) -> float:
        c = COST_TABLE.get(model_id, {"input": 0.0001, "output": 0.0003})
        t = TOKEN_ESTIMATE.get(tier, {"input": 200, "output": 400})
        return t["input"] / 1000 * c["input"] + t["output"] / 1000 * c["output"]

    def _cost_rank(self, m: str) -> float:
        if "ollama" in m: return 0.0
        if "flash"  in m: return 1.0
        if "pro"    in m: return 2.0
        return 1.5

    def _quality_rank(self, m: str) -> float:
        if "pro"    in m: return 3.0
        if "ollama" in m: return 2.5
        if "flash"  in m: return 1.5
        return 1.0
