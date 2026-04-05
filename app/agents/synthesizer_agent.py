"""
SynthesizerAgent — merges results from multiple sub-tasks into a
single coherent final answer.

For single-task plans: returns the result directly.
For multi-task plans: calls an LLM to synthesize all partial results
into a unified, well-structured response.
"""
import logging
from typing import List, Optional, Dict
from app.agents.base_agent import BaseAgent, SubTask, AgentStatus
from app.models.base_model import BaseLLM, Message

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are synthesizing the results of a multi-part analysis into one clear, well-structured answer.

Original question: {original_query}

Sub-task results:
{task_results}

Write a single comprehensive response that:
1. Integrates all the above analysis coherently
2. Eliminates redundancy
3. Uses clear structure (headers if helpful)
4. Ends with a summary or key takeaways

Do not mention that this was broken into sub-tasks."""


class SynthesizerAgent(BaseAgent):

    def __init__(self, models: Dict[str, BaseLLM]):
        super().__init__("Synthesizer")
        self.models = models

    async def synthesize(
        self,
        original_query: str,
        tasks: List[SubTask],
        available_models: List[str],
    ) -> str:
        """Merge sub-task results into a final answer."""

        successful = [t for t in tasks if t.status == AgentStatus.SUCCESS and t.result]

        # Single task — no synthesis needed
        if len(successful) <= 1:
            return successful[0].result if successful else "No results available."

        # All tasks have same result (degenerate case)
        if len(set(t.result for t in successful)) == 1:
            return successful[0].result

        # Build synthesis prompt
        task_results = "\n\n".join([
            f"### Part {i+1}: {t.description[:80]}\n{t.result}"
            for i, t in enumerate(successful)
        ])

        prompt = SYNTHESIS_PROMPT.format(
            original_query=original_query,
            task_results=task_results[:6000],  # cap to avoid token overflow
        )

        # Use best available model for synthesis
        model = self._pick_model(available_models)
        if not model:
            # Fallback: simple concatenation
            return self._simple_concat(original_query, successful)

        try:
            response = await model.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=2048,
            )
            if response.success:
                self.log(f"Synthesized {len(successful)} tasks via {model.model_id}")
                return response.content
        except Exception as e:
            logger.warning(f"Synthesis LLM call failed: {e} — using concat fallback")

        return self._simple_concat(original_query, successful)

    def _pick_model(self, available_models: List[str]) -> Optional[BaseLLM]:
        """Pick best available model for synthesis."""
        # Prefer flash for synthesis (fast + cheap, synthesis is not ultra-complex)
        for preferred in ["gemini-1.5-flash", "ollama-local"]:
            if preferred in available_models and preferred in self.models:
                return self.models[preferred]
        for mid in available_models:
            if mid in self.models:
                return self.models[mid]
        return None

    def _simple_concat(self, query: str, tasks: List[SubTask]) -> str:
        """Fallback: structured concatenation without LLM."""
        parts = [f"**Analysis for:** {query}\n"]
        for i, task in enumerate(tasks, 1):
            parts.append(f"\n## Part {i}\n{task.result}")
        return "\n".join(parts)
