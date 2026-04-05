import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert AI response quality evaluator.

Score the following AI response strictly from 0.0 to 1.0:
- 1.0 = Perfect: accurate, complete, well-structured, directly answers
- 0.8 = Good: accurate and helpful with minor gaps
- 0.6 = Acceptable: partially answers, some issues
- 0.4 = Poor: incomplete or partially wrong
- 0.2 = Bad: mostly wrong or off-topic
- 0.0 = Terrible: completely wrong or refused inappropriately

Query intent: {intent}
User query: {query}
AI response: {response}

Respond ONLY with valid JSON — no other text:
{{"score": 0.85, "reason": "one sentence explanation"}}"""


@dataclass
class EvaluationResult:
    score: float
    reason: str
    evaluator_model: str
    latency_ms: float
    skipped: bool = False


class ResponseEvaluator:

    async def evaluate(self, query: str, response: str, intent: str) -> EvaluationResult:
        if not settings.gemini_api_key:
            return self._heuristic(query, response, intent)
        t0 = time.monotonic()
        try:
            prompt = JUDGE_PROMPT.format(
                intent=intent, query=query[:500], response=response[:1000]
            )
            raw = await self._call_gemini(prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            parsed = self._parse(raw)
            return EvaluationResult(
                score=parsed["score"], reason=parsed["reason"],
                evaluator_model="gemini-1.5-flash", latency_ms=round(latency_ms, 1),
            )
        except Exception as e:
            logger.warning(f"LLM eval failed: {e} — using heuristic")
            return self._heuristic(query, response, intent)

    def _heuristic(self, query: str, response: str, intent: str) -> EvaluationResult:
        score = 0.5
        if len(response) < 10:
            score = 0.1
        elif len(response) > 50:
            score += 0.1
        if intent == "code" and "```" in response:
            score += 0.15
        q_words = set(query.lower().split())
        r_words = set(response.lower().split())
        overlap = len(q_words & r_words) / max(1, len(q_words))
        score  += overlap * 0.2
        if any(p in response.lower() for p in ["i cannot", "i can't", "i'm unable"]):
            score -= 0.3
        return EvaluationResult(
            score=round(min(1.0, max(0.0, score)), 3),
            reason="heuristic scoring",
            evaluator_model="heuristic", latency_ms=0.0,
        )

    async def _call_gemini(self, prompt: str) -> str:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 120},
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, json=payload, params={"key": settings.gemini_api_key})
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _parse(self, text: str) -> dict:
        try:
            return json.loads(text.strip())
        except Exception:
            pass
        m = re.search(r'\{[^}]+\}', text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        sm = re.search(r'"?score"?\s*:\s*([0-9.]+)', text)
        score = float(sm.group(1)) if sm else 0.5
        return {"score": min(1.0, max(0.0, score)), "reason": "partially parsed"}
