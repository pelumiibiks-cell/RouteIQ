from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.dataset import EvalCase
from app.routing.router import RouteDecision


@dataclass
class CaseResult:
    case: EvalCase
    decision: RouteDecision
    selected_tier: int
    correct: bool          # exact match against expected_tier
    in_band: bool          # within the case's defensible acceptable_tiers
    underpowered: bool     # below the whole band
    overkill: bool         # above the whole band
    tier_error: int        # signed distance to the nearest acceptable tier


@dataclass
class BenchmarkReport:
    total: int
    routing_accuracy: float       # exact-tier, the strict number
    banded_accuracy: float        # within acceptable_tiers, the primary number
    underpowered_rate: float
    overkill_rate: float
    mean_abs_tier_error: float
    average_cost: float
    average_latency_ms: float
    average_confidence: float
    tier_confusion: dict[str, dict[str, int]]
    results: list[CaseResult]


def evaluate_case(case: EvalCase, decision: RouteDecision) -> CaseResult:
    """Scored two ways. Exact-tier equality is kept as the strict number, but
    under/overkill are judged against the whole acceptable band -- a pick one
    tier off a point label is only a real miss if it also falls outside the
    range a reasonable engineer would accept."""
    selected_tier = decision.selected.model.tier
    band = case.acceptable_tiers or [case.expected_tier]

    correct = selected_tier == case.expected_tier
    in_band = selected_tier in band
    underpowered = selected_tier < min(band)
    overkill = selected_tier > max(band)

    if in_band:
        tier_error = 0
    elif underpowered:
        tier_error = selected_tier - min(band)
    else:
        tier_error = selected_tier - max(band)

    return CaseResult(case, decision, selected_tier, correct, in_band, underpowered, overkill, tier_error)


def build_report(results: list[CaseResult]) -> BenchmarkReport:
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    in_band = sum(1 for r in results if r.in_band)
    underpowered = sum(1 for r in results if r.underpowered)
    overkill = sum(1 for r in results if r.overkill)
    mean_err = sum(abs(r.tier_error) for r in results) / n
    avg_cost = sum(r.decision.selected.cost_latency.cost_usd for r in results) / n
    avg_latency = sum(r.decision.selected.cost_latency.latency_ms for r in results) / n
    avg_conf = sum(r.decision.confidence for r in results) / n

    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        exp = f"tier{r.case.expected_tier}"
        got = f"tier{r.selected_tier}"
        confusion.setdefault(exp, {}).setdefault(got, 0)
        confusion[exp][got] += 1

    return BenchmarkReport(
        total=n,
        routing_accuracy=round(correct / n, 4),
        banded_accuracy=round(in_band / n, 4),
        underpowered_rate=round(underpowered / n, 4),
        overkill_rate=round(overkill / n, 4),
        mean_abs_tier_error=round(mean_err, 4),
        average_cost=round(avg_cost, 6),
        average_latency_ms=round(avg_latency, 1),
        average_confidence=round(avg_conf, 4),
        tier_confusion=confusion,
        results=results,
    )
