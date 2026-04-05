import logging
from dataclasses import dataclass
from typing import Literal
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.classifier.feature_extractor import FeatureExtractor, QueryFeatures

logger = logging.getLogger(__name__)
ComplexityTier = Literal["simple", "medium", "complex"]


@dataclass
class ClassificationResult:
    tier: ComplexityTier
    confidence: float
    complexity_score: float
    intent: str
    features: QueryFeatures
    reasoning: str


class QueryClassifier:

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.scaler    = StandardScaler()
        self.model     = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self._train()

    def classify(self, query: str, context_turns: int = 0) -> ClassificationResult:
        features = self.extractor.extract(query)
        adjusted = min(1.0, features.complexity_score + context_turns * 0.03)

        # Fast path for obvious cases
        if adjusted < 0.2 and features.intent == "factual":
            tier, conf = "simple", 0.95
        elif adjusted > 0.75 or (features.has_code_block and features.reasoning_keyword_score > 0.1):
            tier, conf = "complex", 0.90
        else:
            tier, conf = self._ml_predict(features)

        parts = [f"complexity={adjusted:.2f}", f"intent={features.intent}"]
        if features.has_code_block: parts.append("has_code")
        if features.has_math:       parts.append("has_math")
        reasoning = f"[{tier.upper()}] " + ", ".join(parts)

        return ClassificationResult(
            tier=tier, confidence=conf, complexity_score=adjusted,
            intent=features.intent, features=features, reasoning=reasoning,
        )

    def _ml_predict(self, features: QueryFeatures):
        vec   = np.array(features.to_vector()).reshape(1, -1)
        scaled = self.scaler.transform(vec)
        proba  = self.model.predict_proba(scaled)[0]
        idx    = int(np.argmax(proba))
        return {0: "simple", 1: "medium", 2: "complex"}[idx], round(float(proba[idx]), 3)

    def _train(self):
        examples = [
            # simple (0)
            ("What is the capital of France?", 0),
            ("Hi", 0), ("Tell me a joke", 0), ("What does API stand for?", 0),
            ("List 5 fruits", 0), ("What year was Python created?", 0),
            ("Translate hello to Spanish", 0), ("What is 2 + 2?", 0),
            ("Who wrote Harry Potter?", 0), ("What is HTTP?", 0),
            # medium (1)
            ("Explain how REST APIs work", 1),
            ("Write a short email to my manager about a delay", 1),
            ("Summarize the pros and cons of microservices", 1),
            ("Write a Python function to reverse a string", 1),
            ("What is the difference between SQL and NoSQL?", 1),
            ("Help me debug this: TypeError: NoneType", 1),
            ("Explain gradient descent in simple terms", 1),
            ("What are best practices for REST API design?", 1),
            ("Write a script to read a CSV file in Python", 1),
            ("Explain the CAP theorem", 1),
            # complex (2)
            ("Design a scalable event-driven microservices architecture for an e-commerce platform with millions of users, including service discovery, message queues, and CQRS pattern", 2),
            ("Implement a distributed rate limiter using Redis and explain the trade-offs of token bucket vs leaky bucket", 2),
            ("Analyze and refactor this class to use SOLID principles, add type hints, and write unit tests", 2),
            ("Compare transformer architecture vs LSTM for time series forecasting, implement both and benchmark", 2),
            ("Build a CI/CD pipeline using GitHub Actions with Docker, staging and production environments, and rollback", 2),
            ("Explain and implement consistent hashing with virtual nodes for a distributed cache", 2),
            ("Write a detailed system design for a real-time collaborative document editor like Google Docs", 2),
            ("Design a multi-tenant SaaS application with row-level security, audit logs, and per-tenant rate limiting", 2),
        ]
        texts, labels = zip(*examples)
        X = np.array([self.extractor.extract(t).to_vector() for t in texts])
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        logger.info(f"Classifier trained on {len(texts)} examples")
