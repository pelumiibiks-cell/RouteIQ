from __future__ import annotations

from app.evaluation.dataset import EvalCase, get_dataset
from app.evaluation.metrics import BenchmarkReport, build_report, evaluate_case
from app.registry.registry import ModelRegistry, get_default_registry
from app.routing.router import route


def run_benchmark(registry: ModelRegistry | None = None, cases: list[EvalCase] | None = None) -> BenchmarkReport:
    registry = registry or get_default_registry()
    cases = cases if cases is not None else get_dataset()

    results = []
    for case in cases:
        decision = route(case.prompt, case.context, case.attachments or [], registry=registry)
        results.append(evaluate_case(case, decision))

    return build_report(results)


def compare_to_always_strongest(registry: ModelRegistry | None = None, cases: list[EvalCase] | None = None) -> dict:
    registry = registry or get_default_registry()
    cases = cases if cases is not None else get_dataset()
    strongest = max(registry.all(), key=lambda m: m.tier)

    router_cost = 0.0
    strongest_cost = 0.0
    from app.routing import cost_latency
    from app.analysis.normalizer import normalize
    from app.analysis.task_analyzer import analyze

    for case in cases:
        decision = route(case.prompt, case.context, case.attachments or [], registry=registry)
        router_cost += decision.selected.cost_latency.cost_usd

        normalized = normalize(case.prompt, case.context, case.attachments or [])
        task_analysis = analyze(normalized)
        est = cost_latency.estimate(task_analysis, strongest, strongest.max_reasoning_effort)
        strongest_cost += est.cost_usd

    savings_pct = 0.0 if strongest_cost == 0 else round((1 - router_cost / strongest_cost) * 100, 1)

    return {
        "router_total_cost": round(router_cost, 6),
        "always_strongest_total_cost": round(strongest_cost, 6),
        "cost_savings_pct": savings_pct,
    }
