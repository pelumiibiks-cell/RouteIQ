from __future__ import annotations

from app.evaluation.dataset import EvalCase, get_all_cases, get_dataset, get_test_dataset
from app.evaluation.metrics import BenchmarkReport, build_report, evaluate_case
from app.registry.registry import ModelRegistry, get_default_registry
from app.routing.router import route


def resolve_cases(split: str = "dev") -> list[EvalCase]:
    """`dev` is tunable, `test` is held out, `all` is both. Anything that picks
    a weight or threshold must use `dev` only."""
    if split == "dev":
        return get_dataset()
    if split == "test":
        return get_test_dataset()
    if split == "all":
        return get_all_cases()
    raise ValueError(f"unknown split {split!r}; expected 'dev', 'test' or 'all'")


def run_benchmark(
    registry: ModelRegistry | None = None,
    cases: list[EvalCase] | None = None,
    split: str = "dev",
) -> BenchmarkReport:
    registry = registry or get_default_registry()
    cases = cases if cases is not None else resolve_cases(split)

    results = []
    for case in cases:
        decision = route(case.prompt, case.context, case.attachments or [], registry=registry)
        results.append(evaluate_case(case, decision))

    return build_report(results)


def compare_to_always_strongest(
    registry: ModelRegistry | None = None,
    cases: list[EvalCase] | None = None,
    split: str = "dev",
    report: BenchmarkReport | None = None,
) -> dict:
    """Pass `report` from a benchmark over the same cases to reuse its routing
    decisions. Without it this re-runs the whole pipeline twice per case (once
    to route, once to analyze for the baseline estimate) purely to add up costs."""
    registry = registry or get_default_registry()
    cases = cases if cases is not None else resolve_cases(split)
    strongest = max(registry.all(), key=lambda m: m.tier)

    router_cost = 0.0
    strongest_cost = 0.0
    from app.routing import cost_latency

    reused = {r.case.id: r.decision for r in report.results} if report else {}

    for case in cases:
        decision = reused.get(case.id)
        if decision is None:
            decision = route(case.prompt, case.context, case.attachments or [], registry=registry)
        router_cost += decision.selected.cost_latency.cost_usd

        # The baseline pays the same task analysis the router already did.
        est = cost_latency.estimate(decision.task_analysis, strongest, strongest.max_reasoning_effort)
        strongest_cost += est.cost_usd

    savings_pct = 0.0 if strongest_cost == 0 else round((1 - router_cost / strongest_cost) * 100, 1)

    return {
        "router_total_cost": round(router_cost, 6),
        "always_strongest_total_cost": round(strongest_cost, 6),
        "cost_savings_pct": savings_pct,
    }
