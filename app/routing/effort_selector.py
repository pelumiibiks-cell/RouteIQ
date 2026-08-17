"""Effort/reasoning level selection -- a decision independent of which model
is chosen, then clamped to what the chosen model actually supports.
"""
from __future__ import annotations

from app.analysis.complexity_scorer import ComplexityScore
from app.analysis.task_analyzer import TaskAnalysis
from app.registry.model_profile import EFFORT_LEVELS, ModelProfile

_THRESHOLDS = [
    (2.5, "low"),
    (4.5, "medium"),
    (6.5, "high"),
    (8.3, "xhigh"),
    (10.01, "max"),
]


def select_effort(task: TaskAnalysis, complexity: ComplexityScore) -> str:
    req = task.requirements

    # Effort driver score: reasoning complexity, error tolerance (inverse of
    # ambiguity/precision), number of steps, planning depth, consequence of failure.
    driver = (
        complexity.overall * 0.45
        + req["number_of_steps"] * 0.15
        + req["output_complexity"] * 0.15
        + req["reliability_requirement"] * 0.15
        + req["ambiguity"] * 0.10
    )
    driver = max(0.0, min(10.0, driver))

    for ceiling, level in _THRESHOLDS:
        if driver < ceiling:
            return level
    return "max"


def clamp_effort_to_model(effort: str, model: ModelProfile) -> str:
    if model.supports_effort(effort):
        return effort
    return model.max_reasoning_effort
