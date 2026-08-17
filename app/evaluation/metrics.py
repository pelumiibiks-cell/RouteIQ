from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.dataset import EvalCase
from app.routing.router import RouteDecision


@dataclass
class CaseResult:
    case: EvalCase
    decision: RouteDecision
    selected_tier: int
    correct: bool
    underpowered: bool
    overkill: bool


@dataclass
class BenchmarkReport:
    total: int
    routing_accuracy: float
    underpowered_rate: float
    overkill_rate: float
    average_cost: float
    average_latency_ms: float
    average_confidence: float
    tier_confusion: dict[str, dict[str, int]]
    results: list[CaseResult]


def evaluate_case(case: EvalCase, decision: RouteDecision) -> CaseResult:
    selected_tier = decision.selected.model.tier
    correct = selected_tier == case.expected_tier
    underpowered = selected_tier < case.expected_tier
    overkill = selected_tier > case.expected_tier
    return CaseResult(case, decision, selected_tier, correct, underpowered, overkill)


def build_report(results: list[CaseResult]) -> BenchmarkReport:
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    underpowered = sum(1 for r in results if r.underpowered)
    overkill = sum(1 for r in results if r.overkill)
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
        underpowered_rate=round(underpowered / n, 4),
        overkill_rate=round(overkill / n, 4),
        average_cost=round(avg_cost, 6),
        average_latency_ms=round(avg_latency, 1),
        average_confidence=round(avg_conf, 4),
        tier_confusion=confusion,
        results=results,
    )
