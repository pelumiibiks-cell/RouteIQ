"""Deterministic cost/latency estimation used during ranking (separate from
the randomized MockProvider used for execution simulation in evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.analysis.task_analyzer import TaskAnalysis
from app.registry.model_profile import ModelProfile

EFFORT_OUTPUT_MULTIPLIER = {
    "low": 0.4,
    "medium": 0.8,
    "high": 1.2,
    "xhigh": 1.7,
    "max": 2.3,
}

EFFORT_LATENCY_MULTIPLIER = {
    "low": 0.5,
    "medium": 0.85,
    "high": 1.3,
    "xhigh": 1.8,
    "max": 2.5,
}


@dataclass
class CostLatencyEstimate:
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


def estimate(task: TaskAnalysis, model: ModelProfile, effort: str) -> CostLatencyEstimate:
    input_tokens = max(20, int(task.normalized.char_count / 4) + 200)
    base_output = max(30, int(input_tokens * 0.4 + task.requirements["output_complexity"] * 40))
    output_tokens = int(base_output * EFFORT_OUTPUT_MULTIPLIER.get(effort, 1.0))

    cost = (input_tokens / 1000) * model.cost_per_input_token + (output_tokens / 1000) * model.cost_per_output_token

    base_latency = 250 + input_tokens * 0.6 + output_tokens * 1.4
    latency = base_latency * EFFORT_LATENCY_MULTIPLIER.get(effort, 1.0) * (11 - model.latency_score) / 5

    return CostLatencyEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency, 1),
    )
