from __future__ import annotations

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    max_cost: float | None = None
    max_latency_ms: float | None = None
    minimum_quality: float | None = None


class RouteRequest(BaseModel):
    prompt: str
    context: str = ""
    attachments: list[str] = Field(default_factory=list)
    constraints: Constraints | None = None


class AlternativeModel(BaseModel):
    model: str
    effort: str
    utility: float
    quality_estimate: float
    overkill_risk: float
    underpowered_risk: float
    estimated_cost: float
    estimated_latency_ms: float
    rejected_reason: str


class RouteResponse(BaseModel):
    model: str
    effort: str
    confidence: float
    difficulty: float
    reasoning_score: float
    categories: list[str]
    dimension_scores: dict[str, float]
    estimated_cost: float
    estimated_latency_ms: float
    overkill_risk: float
    underpowered_risk: float
    quality_estimate: float
    two_pass_used: bool
    alternatives: list[AlternativeModel]
    explanation: str
    positive_reasons: list[str]
    negative_reasons: list[str]
    rejected_alternatives: list[str]
    record_id: int | None = None


class FeedbackRequest(BaseModel):
    record_id: int
    actual_result_quality: float | None = None
    actual_latency_ms: float | None = None
    actual_cost: float | None = None
    user_feedback: str | None = None
    success: bool | None = None
