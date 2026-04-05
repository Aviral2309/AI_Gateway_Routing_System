"""
PlannerAgent — decides whether to decompose a query or handle it as-is.

For simple/medium queries: returns single sub-task (pass-through).
For complex queries: breaks into 2-4 focused sub-tasks that can be
answered independently and then synthesized.

Decomposition logic is rule-based (fast, no LLM call needed for planning).
"""
import uuid
import re
from typing import List
from app.agents.base_agent import BaseAgent, SubTask
from app.classifier.query_classifier import ClassificationResult


class PlannerAgent(BaseAgent):

    def __init__(self):
        super().__init__("Planner")

    def plan(self, query: str, classification: ClassificationResult) -> List[SubTask]:
        """
        Returns a list of sub-tasks to execute.
        Single task = no decomposition.
        Multiple tasks = decomposed query.
        """
        if classification.tier != "complex":
            return [self._single_task(query, classification)]

        decomposed = self._try_decompose(query, classification)
        if decomposed:
            self.log(f"Decomposed into {len(decomposed)} sub-tasks")
            return decomposed

        return [self._single_task(query, classification)]

    def _single_task(self, query: str, cls: ClassificationResult) -> SubTask:
        return SubTask(
            id=str(uuid.uuid4())[:8],
            description=query,
            intent=cls.intent,
            complexity=cls.tier,
            depends_on=[],
        )

    def _try_decompose(self, query: str, cls: ClassificationResult) -> List[SubTask]:
        """
        Pattern-based decomposition for compound queries.
        Returns empty list if decomposition is not applicable.
        """
        lower = query.lower()

        # Pattern 1: "implement X and explain Y"
        if re.search(r'\b(implement|write|create|build)\b.+\b(and|then)\b.+\b(explain|analyze|compare|test)\b', lower):
            return self._split_implement_explain(query, cls)

        # Pattern 2: "compare A vs B" or "difference between A and B"
        if re.search(r'\b(compare|vs|versus|difference between|contrast)\b', lower):
            return self._split_comparison(query, cls)

        # Pattern 3: Multiple requirements separated by commas or "including"
        if lower.count(',') >= 3 or re.search(r'\bincluding\b.+,', lower):
            return self._split_multi_requirement(query, cls)

        # Pattern 4: "design X with Y, Z, W" (system design)
        if re.search(r'\b(design|architect|build)\b.+\bwith\b', lower) and lower.count(',') >= 2:
            return self._split_system_design(query, cls)

        return []

    def _split_implement_explain(self, query: str, cls: ClassificationResult) -> List[SubTask]:
        t1 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Implementation part: {query}",
            intent="code",
            complexity="complex",
            depends_on=[],
        )
        t2 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Explanation and analysis part: {query}",
            intent="reasoning",
            complexity="medium",
            depends_on=[t1.id],
        )
        return [t1, t2]

    def _split_comparison(self, query: str, cls: ClassificationResult) -> List[SubTask]:
        t1 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"First option analysis for: {query}",
            intent="reasoning",
            complexity="medium",
            depends_on=[],
        )
        t2 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Second option analysis for: {query}",
            intent="reasoning",
            complexity="medium",
            depends_on=[],
        )
        t3 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Final comparison and recommendation for: {query}",
            intent="reasoning",
            complexity="medium",
            depends_on=[t1.id, t2.id],
        )
        return [t1, t2, t3]

    def _split_multi_requirement(self, query: str, cls: ClassificationResult) -> List[SubTask]:
        t1 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Core implementation for: {query}",
            intent=cls.intent,
            complexity="complex",
            depends_on=[],
        )
        t2 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Additional requirements and integration for: {query}",
            intent=cls.intent,
            complexity="medium",
            depends_on=[t1.id],
        )
        return [t1, t2]

    def _split_system_design(self, query: str, cls: ClassificationResult) -> List[SubTask]:
        t1 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"High-level architecture and design decisions for: {query}",
            intent="reasoning",
            complexity="complex",
            depends_on=[],
        )
        t2 = SubTask(
            id=str(uuid.uuid4())[:8],
            description=f"Implementation details, code examples, and trade-offs for: {query}",
            intent="code",
            complexity="complex",
            depends_on=[t1.id],
        )
        return [t1, t2]
