"""Difficulty/complexity scoring.

Produces a 0-10 dimension breakdown plus a single overall difficulty score.
The overall score is NOT a plain average: reasoning/coding/math/planning are
weighted more heavily than secondary dimensions, and the result is nudged by
ambiguity and reliability requirements since both make a task harder to get
right on the first try.

Hard constraints (vision required but model lacks vision, context too long,
etc.) are NOT applied here -- they belong to capability matching, since they
are about a *model*, not the task in isolation. This module only scores the
task itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.analysis.task_analyzer import TaskAnalysis

DIMENSION_WEIGHTS = {
    "reasoning_depth": 1.4,
    "coding_complexity": 1.2,
    "mathematical_complexity": 1.2,
    "output_complexity": 0.9,  # proxy for "planning complexity"
    "agentic_requirement": 1.1,  # proxy for "tool/agent complexity"
    "multimodal_requirement": 0.8,
    "precision_requirement": 0.8,
    "ambiguity": 0.6,
    "reliability_requirement": 0.7,
    "context_length": 0.5,
    "domain_specialization": 1.1,
    "research_requirement": 0.9,
}


@dataclass
class ComplexityScore:
    dimensions: dict[str, float]
    overall: float
    drivers: list[str]


def score(task_analysis: TaskAnalysis) -> ComplexityScore:
    req = task_analysis.requirements

    dims = {
        "reasoning": req["reasoning_depth"],
        "context": req["context_length"],
        "coding_complexity": req["coding_complexity"],
        "math_complexity": req["mathematical_complexity"],
        "planning_complexity": req["output_complexity"],
        "tool_agent_complexity": max(req["tool_usage_requirement"], req["agentic_requirement"]),
        "multimodal_complexity": req["multimodal_requirement"],
        "precision_requirement": req["precision_requirement"],
        "ambiguity": req["ambiguity"],
        "reliability_requirement": req["reliability_requirement"],
        "domain_specialization": req["domain_specialization"],
        "research_requirement": req["research_requirement"],
    }

    # Overall difficulty is dominated by whichever dimensions are actually
    # elevated for THIS task, not diluted by irrelevant dimensions sitting at
    # zero (e.g. a pure agentic-planning task has math_complexity == 0, which
    # a plain weighted average would use to drag difficulty down). We take a
    # weighted blend of the top-3 weighted dimension values instead of
    # averaging across all ten every time.
    weighted_dims = {key: req[key] * DIMENSION_WEIGHTS[key] / max(DIMENSION_WEIGHTS.values()) for key in DIMENSION_WEIGHTS}
    top_sorted = sorted(weighted_dims.values(), reverse=True)
    top3 = (top_sorted + [0.0, 0.0])[:3]
    overall = top3[0] * 0.55 + top3[1] * 0.30 + top3[2] * 0.15

    # Ambiguity and reliability act as modifiers rather than pure linear terms:
    # a highly ambiguous OR high-stakes task is harder than the raw average suggests.
    if req["ambiguity"] >= 7:
        overall += 0.4
    if req["reliability_requirement"] >= 8:
        overall += 0.5
    # A genuinely ambiguous, under-specified task benefits from a model with
    # stronger instruction-following/judgment even if no single dimension is
    # individually "hard" -- so ambiguity gets a floor effect, not just a bonus.
    if req["ambiguity"] >= 5.5:
        overall = max(overall, 3.2 + (req["ambiguity"] - 5.5) * 0.4)

    overall = max(0.0, min(10.0, overall))

    drivers = sorted(dims.keys(), key=lambda k: -dims[k])[:4]
    drivers = [d for d in drivers if dims[d] >= 4.0]

    return ComplexityScore(dimensions=dims, overall=round(overall, 2), drivers=drivers)
