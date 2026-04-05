from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, Index
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(String(64), nullable=False, index=True)
    query          = Column(Text, nullable=False)
    response       = Column(Text, nullable=True)

    tier             = Column(String(16), nullable=False)
    intent           = Column(String(32), nullable=False)
    complexity_score = Column(Float, nullable=False)

    model_used       = Column(String(64), nullable=False, index=True)
    routing_strategy = Column(String(32), nullable=False)
    was_decomposed   = Column(Boolean, default=False)
    agent_steps      = Column(Integer, default=1)

    input_tokens  = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms    = Column(Float, nullable=False)
    cost_usd      = Column(Float, default=0.0)

    quality_score  = Column(Float, nullable=True)
    quality_reason = Column(Text, nullable=True)

    success       = Column(Boolean, default=True)
    retry_count   = Column(Integer, default=0)
    from_cache    = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_logs_model_created", "model_used", "created_at"),
        Index("ix_logs_session_created", "session_id", "created_at"),
    )


class SessionBudget(Base):
    __tablename__ = "session_budgets"

    session_id    = Column(String(64), primary_key=True)
    total_spent   = Column(Float, default=0.0)
    request_count = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
