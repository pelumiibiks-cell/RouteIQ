from __future__ import annotations

from dataclasses import dataclass, field

from app.analysis import complexity_scorer
from app.analysis.normalizer import normalize
from app.analysis.task_analyzer import (
    TaskAnalysis,
    approximate_task_analysis,
    analyze,
    quick_analyze,
)
from app.registry.model_profile import ModelProfile
from app.registry.registry import ModelRegistry, get_default_registry
from app.routing import capability_matcher, cost_latency
from app.routing.candidate_generator import generate_candidates
from app.routing.effort_selector import clamp_effort_to_model, select_effort
from app.routing.explanation import build_explanation

# Borderline band around the tier-difficulty cut points; if the cheap Pass 1
# estimate falls within this margin of a cut point, run full Pass 2 analysis.
_TIER_CUTPOINTS = [3.0, 5.0, 7.0]
_BORDERLINE_MARGIN = 1.0

UTILITY_WEIGHTS = {
    "quality": 1.0,
    "cost": 0.25,
    "latency": 0.15,
    "overkill": 0.35,
    "underpowered": 0.45,
}


@dataclass
class RouteConstraints:
    max_cost: float | None = None
    max_latency_ms: float | None = None
    minimum_quality: float | None = None


@dataclass
class CandidateEvaluation:
    model: ModelProfile
    effort: str
    match: capability_matcher.MatchResult
    cost_latency: cost_latency.CostLatencyEstimate
    utility: float
    eliminated: bool
    elimination_reason: str | None


@dataclass
class RouteDecision:
    prompt: str
    task_analysis: TaskAnalysis
    complexity: complexity_scorer.ComplexityScore
    two_pass_used: bool
    selected: CandidateEvaluation
    ranked: list[CandidateEvaluation]
    confidence: float
    explanation_text: str
    positive_reasons: list[str]
    negative_reasons: list[str]
    rejected_alternatives: list[str]


def _is_borderline(rough_difficulty: float) -> bool:
    return any(abs(rough_difficulty - cp) <= _BORDERLINE_MARGIN for cp in _TIER_CUTPOINTS)


def _normalize_penalty(value: float, values: list[float]) -> float:
    m = max(values) if values else 0.0
    if m <= 0:
        return 0.0
    return value / m


def route(
    prompt: str,
    context: str = "",
    attachments: list[str] | None = None,
    constraints: RouteConstraints | None = None,
    registry: ModelRegistry | None = None,
) -> RouteDecision:
    registry = registry or get_default_registry()
    constraints = constraints or RouteConstraints()

    normalized = normalize(prompt, context, attachments)

    # --- Pass 1: cheap analysis ---
    quick = quick_analyze(normalized)
    two_pass_used = _is_borderline(quick.rough_difficulty)

    # --- Pass 2: deeper analysis, only when borderline ---
    if two_pass_used:
        task_analysis = analyze(normalized)
    else:
        task_analysis = approximate_task_analysis(quick)

    complexity = complexity_scorer.score(task_analysis)
    desired_effort = select_effort(task_analysis, complexity)

    candidates = generate_candidates(registry)
    evaluations: list[CandidateEvaluation] = []

    for model in candidates:
        match_result = capability_matcher.match(task_analysis, complexity, model)
        effort = clamp_effort_to_model(desired_effort, model)
        cl_estimate = cost_latency.estimate(task_analysis, model, effort)

        eliminated = match_result.eliminated
        reason = match_result.elimination_reason

        if not eliminated and constraints.max_cost is not None and cl_estimate.cost_usd > constraints.max_cost:
            eliminated, reason = True, f"estimated cost {cl_estimate.cost_usd} exceeds max_cost constraint {constraints.max_cost}"
        if not eliminated and constraints.max_latency_ms is not None and cl_estimate.latency_ms > constraints.max_latency_ms:
            eliminated, reason = True, f"estimated latency {cl_estimate.latency_ms}ms exceeds max_latency_ms constraint {constraints.max_latency_ms}"
        if not eliminated and constraints.minimum_quality is not None and match_result.quality_estimate < constraints.minimum_quality:
            eliminated, reason = True, f"estimated quality {match_result.quality_estimate} below minimum_quality constraint {constraints.minimum_quality}"

        evaluations.append(
            CandidateEvaluation(
                model=model, effort=effort, match=match_result, cost_latency=cl_estimate,
                utility=0.0, eliminated=eliminated, elimination_reason=reason,
            )
        )

    survivors = [e for e in evaluations if not e.eliminated]
    if not survivors:
        # Relax constraints as a last resort: keep hard-capability survivors, ignore soft constraints.
        survivors = [e for e in evaluations if not e.match.eliminated]
        for e in survivors:
            e.eliminated = False
            e.elimination_reason = None
    if not survivors:
        raise ValueError("No candidate model satisfies the task's hard capability requirements.")

    costs = [e.cost_latency.cost_usd for e in survivors]
    latencies = [e.cost_latency.latency_ms for e in survivors]

    for e in survivors:
        cost_penalty = _normalize_penalty(e.cost_latency.cost_usd, costs)
        latency_penalty = _normalize_penalty(e.cost_latency.latency_ms, latencies)
        e.utility = (
            e.match.quality_estimate * UTILITY_WEIGHTS["quality"]
            - cost_penalty * UTILITY_WEIGHTS["cost"]
            - latency_penalty * UTILITY_WEIGHTS["latency"]
            - e.match.overkill_risk * UTILITY_WEIGHTS["overkill"]
            - e.match.underpowered_risk * UTILITY_WEIGHTS["underpowered"]
        )

    ranked = sorted(survivors, key=lambda e: -e.utility)
    selected = ranked[0]

    # confidence: margin between best and runner-up utility, tempered by ambiguity
    if len(ranked) > 1:
        gap = selected.utility - ranked[1].utility
    else:
        gap = 0.4
    ambiguity_penalty = task_analysis.requirements["ambiguity"] / 10.0 * 0.15
    confidence = 0.55 + min(0.4, max(0.0, gap) * 1.2) - ambiguity_penalty
    confidence = round(max(0.05, min(0.99, confidence)), 2)

    rejected_pairs = []
    for e in evaluations:
        if e is selected:
            continue
        if e.eliminated:
            rejected_pairs.append((e.model.name, e.elimination_reason or "eliminated"))
        else:
            if e.match.overkill_risk >= 0.3:
                rejected_pairs.append((e.model.name, "unnecessary cost/latency for expected quality gain"))
            elif e.match.underpowered_risk >= 0.3:
                rejected_pairs.append((e.model.name, "acceptable but higher failure risk than the selected model"))
            else:
                rejected_pairs.append((e.model.name, "lower overall utility than the selected model"))

    explanation = build_explanation(
        selected_model_name=selected.model.name,
        effort=selected.effort,
        complexity_overall=complexity.overall,
        drivers=complexity.drivers,
        quality_estimate=selected.match.quality_estimate,
        overkill_risk=selected.match.overkill_risk,
        underpowered_risk=selected.match.underpowered_risk,
        rejected=rejected_pairs,
        two_pass_used=two_pass_used,
    )

    return RouteDecision(
        prompt=prompt,
        task_analysis=task_analysis,
        complexity=complexity,
        two_pass_used=two_pass_used,
        selected=selected,
        ranked=ranked,
        confidence=confidence,
        explanation_text=explanation.summary,
        positive_reasons=explanation.positive_reasons,
        negative_reasons=explanation.negative_reasons,
        rejected_alternatives=explanation.rejected_alternatives,
    )
