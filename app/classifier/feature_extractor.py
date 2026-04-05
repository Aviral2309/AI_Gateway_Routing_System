import re
import math
from dataclasses import dataclass
from typing import List

CODE_KEYWORDS = {
    "code", "function", "class", "implement", "debug", "error", "bug",
    "syntax", "algorithm", "script", "program", "compile", "runtime",
    "api", "library", "framework", "sql", "query", "regex", "bash",
    "python", "javascript", "typescript", "java", "c++", "rust", "go",
    "docker", "kubernetes", "git", "database", "async", "thread",
}

REASONING_KEYWORDS = {
    "explain", "why", "how", "analyze", "compare", "evaluate", "assess",
    "difference", "pros", "cons", "tradeoff", "should i", "recommend",
    "strategy", "approach", "design", "architecture", "best practice",
    "when to", "which is", "versus", "vs", "advantages", "disadvantages",
}

FACTUAL_KEYWORDS = {
    "what is", "who is", "when", "where", "define", "meaning", "capital",
    "how many", "list", "name", "tell me", "what are", "history of",
}

CREATIVE_KEYWORDS = {
    "write", "create", "generate", "draft", "story", "poem", "essay",
    "blog", "email", "letter", "summarize", "rewrite", "rephrase",
}


@dataclass
class QueryFeatures:
    token_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    has_code_block: bool
    has_math: bool
    question_count: int
    clause_depth: int
    code_keyword_score: float
    reasoning_keyword_score: float
    factual_keyword_score: float
    creative_keyword_score: float
    complexity_score: float
    intent: str

    def to_vector(self) -> List[float]:
        return [
            self.token_count / 500,
            self.word_count / 200,
            self.sentence_count / 10,
            self.avg_word_length / 10,
            float(self.has_code_block),
            float(self.has_math),
            self.question_count / 5,
            self.clause_depth / 10,
            self.code_keyword_score,
            self.reasoning_keyword_score,
            self.factual_keyword_score,
            self.creative_keyword_score,
        ]


class FeatureExtractor:

    def extract(self, query: str) -> QueryFeatures:
        text  = query.strip()
        lower = text.lower()
        words = text.split()

        token_count    = max(1, len(text) // 4)
        word_count     = len(words)
        sentence_count = max(1, len(re.split(r'[.!?]+', text)))
        avg_word_len   = sum(len(w) for w in words) / max(1, word_count)

        has_code_block = bool(re.search(r'```|`[^`]+`', text))
        has_math       = bool(re.search(r'[\+\-\*/=<>^√∑∫]|\d+\.\d+|\bx\b.*=', text))
        question_count = text.count('?')
        clause_depth   = text.count(',') + len(re.findall(
            r'\b(and|but|because|however|therefore|although|whereas|if|unless|including|with|plus)\b', lower
        ))

        code_score     = self._kw_score(lower, CODE_KEYWORDS)
        reason_score   = self._kw_score(lower, REASONING_KEYWORDS)
        factual_score  = self._kw_score(lower, FACTUAL_KEYWORDS)
        creative_score = self._kw_score(lower, CREATIVE_KEYWORDS)

        complexity = self._complexity(token_count, has_code_block, has_math, clause_depth, reason_score, code_score)
        intent     = self._intent(code_score, reason_score, factual_score, creative_score)

        return QueryFeatures(
            token_count=token_count, word_count=word_count,
            sentence_count=sentence_count, avg_word_length=avg_word_len,
            has_code_block=has_code_block, has_math=has_math,
            question_count=question_count, clause_depth=clause_depth,
            code_keyword_score=code_score, reasoning_keyword_score=reason_score,
            factual_keyword_score=factual_score, creative_keyword_score=creative_score,
            complexity_score=complexity, intent=intent,
        )

    def _kw_score(self, lower: str, kw: set) -> float:
        hits = sum(1 for k in kw if k in lower)
        return min(1.0, hits / max(1, math.sqrt(len(kw))))

    def _complexity(self, tokens, code_block, math, clauses, reason, code) -> float:
        s  = min(0.35, tokens / 300 * 0.35)
        if code_block: s += 0.20
        if math:       s += 0.10
        s += min(0.20, clauses / 10 * 0.20)
        s += reason * 0.15
        s += code   * 0.10
        return round(min(1.0, s), 4)

    def _intent(self, code, reasoning, factual, creative) -> str:
        scores = {"code": code, "reasoning": reasoning, "factual": factual, "creative": creative}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0.05 else "general"
