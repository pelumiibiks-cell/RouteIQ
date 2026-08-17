"""Routing telemetry: the record written after every /route call (and
optionally updated after execution/feedback). This is the substrate the
`adaptive` module reads to move the routing policy from pure heuristics
toward a learned policy over time, without changing the pipeline shape.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RoutingRecord(Base):
    __tablename__ = "routing_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    prompt: Mapped[str] = mapped_column(String)
    task_features: Mapped[dict] = mapped_column(JSON)
    estimated_difficulty: Mapped[float] = mapped_column(Float)

    selected_model: Mapped[str] = mapped_column(String)
    selected_effort: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)

    estimated_cost: Mapped[float] = mapped_column(Float)
    estimated_latency_ms: Mapped[float] = mapped_column(Float)

    # Filled in later via feedback endpoint; nullable until then.
    actual_result_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_feedback: Mapped[str | None] = mapped_column(String, nullable=True)
    success: Mapped[bool | None] = mapped_column(nullable=True)
