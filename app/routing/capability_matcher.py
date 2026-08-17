"""Capability compatibility matching: hard constraints + soft quality/risk
estimates for a single (task, model) pair.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.analysis.complexity_scorer import ComplexityScore
from app.analysis.task_analyzer import TaskAnalysis
from app.registry.model_profile import ModelProfile

ESTIMATED_CHARS_PER_TOKEN = 4
MULTIMODAL_HARD_THRESHOLD = 4.0
TOOL_USE_HARD_THRESHOLD = 6.0


@dataclass
class MatchResult:
    model: ModelProfile
    eliminated: bool
    elimination_reason: str | None
    quality_estimate: float  # 0-1, expected probability of a satisfactory result
    underpowered_risk: float  # 0-1
    overkill_risk: float  # 0-1
    notes: list[str]


def _estimate_required_tokens(task: TaskAnalysis) -> int:
    return int(task.normalized.char_count / ESTIMATED_CHARS_PER_TOKEN * 1.3) + 500


def _ideal_tier(difficulty_overall: float) -> int:
    """Maps overall task difficulty (0-10) to the weakest model tier
    expected to handle it reliably. Kept as one place so tier boundaries
    stay consistent with the two-pass borderline bands in router.py.
    """
    if difficulty_overall < 3.0:
        return 1
    if difficulty_overall < 5.0:
        return 2
    if difficulty_overall < 7.0:
        return 3
    return 4


def match(task: TaskAnalysis, complexity: ComplexityScore, model: ModelProfile) -> MatchResult:
    notes: list[str] = []
    req = task.requirements

    # --- hard constraints (elimination) ---
    if req["multimodal_requirement"] >= MULTIMODAL_HARD_THRESHOLD and not model.vision:
        return MatchResult(model, True, "requires multimodal/vision capability the model does not have", 0.0, 1.0, 0.0, notes)

    required_tokens = _estimate_required_tokens(task)
    if required_tokens > model.context_window:
        return MatchResult(
            model, True,
            f"requires ~{required_tokens} tokens of context, exceeds model window of {model.context_window}",
            0.0, 1.0, 0.0, notes,
        )

    if req["tool_usage_requirement"] >= TOOL_USE_HARD_THRESHOLD and not model.tool_use:
        return MatchResult(model, True, "requires tool/function-calling the model does not support", 0.0, 1.0, 0.0, notes)

    # --- soft quality estimate ---
    reasoning_gap = req["reasoning_depth"] - model.reasoning_score
    coding_gap = req["coding_complexity"] - model.coding_score
    math_gap = req["mathematical_complexity"] - model.math_score
    instruction_gap = req["instruction_complexity"] - model.instruction_following_score

    weighted_gap = (
        reasoning_gap * 0.40
        + coding_gap * 0.25
        + math_gap * 0.20
        + instruction_gap * 0.15
    )

    # quality_estimate: 1.0 when model capability comfortably exceeds requirement,
    # degrading as the gap grows. Centered via a soft logistic-like curve.
    quality_estimate = 1.0 / (1.0 + max(0.0, weighted_gap) ** 1.6 / 8.0)
    quality_estimate *= 0.5 + 0.5 * (model.reliability_score / 10.0)
    quality_estimate = max(0.02, min(0.99, quality_estimate))

    # --- underpowered risk: task genuinely needs more than this model offers ---
    underpowered_risk = 0.0
    if reasoning_gap > 1.5:
        underpowered_risk += min(0.5, reasoning_gap * 0.08)
        notes.append("reasoning requirement exceeds model's reasoning capability")
    if coding_gap > 1.5:
        underpowered_risk += min(0.3, coding_gap * 0.06)
        notes.append("coding requirement exceeds model's coding capability")
    if math_gap > 1.5:
        underpowered_risk += min(0.3, math_gap * 0.06)
    if req["reliability_requirement"] >= 8 and model.reliability_score < 8.5:
        underpowered_risk += 0.15

    # --- ideal-tier gap: continuous "use the weakest model that can do the
    # job" signal, derived from overall task difficulty rather than keywords.
    ideal_tier = _ideal_tier(complexity.overall)
    tier_gap = model.tier - ideal_tier
    overkill_risk = 0.0
    if tier_gap > 0:
        overkill_risk += min(0.75, tier_gap * 0.22)
        notes.append(f"model tier ({model.tier}) exceeds the difficulty-implied tier ({ideal_tier})")
    if tier_gap < 0:
        underpowered_risk += min(0.6, -tier_gap * 0.20)
        notes.append(f"model tier ({model.tier}) is below the difficulty-implied tier ({ideal_tier})")

    underpowered_risk = max(0.0, min(1.0, underpowered_risk))
    overkill_risk = max(0.0, min(1.0, overkill_risk))

    return MatchResult(
        model=model,
        eliminated=False,
        elimination_reason=None,
        quality_estimate=round(quality_estimate, 4),
        underpowered_risk=round(underpowered_risk, 4),
        overkill_risk=round(overkill_risk, 4),
        notes=notes,
    )
