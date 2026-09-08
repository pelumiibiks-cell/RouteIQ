from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from functools import lru_cache

from app.evaluation.runner import compare_to_always_strongest, resolve_cases, run_benchmark
from app.evaluation.tournament import run_tournament

router = APIRouter()


@lru_cache(maxsize=4)
def _benchmark_payload(split: str) -> dict:
    """Cached: the benchmark is a pure function of the dataset and the router's
    constants, neither of which change at runtime, but the endpoint recomputed
    the whole thing on every request. It also called compare_to_always_strongest
    with no report, which re-ran the full pipeline for every case a second time
    and re-analyzed each prompt a third -- roughly 48 route() calls plus 24
    analyses per HTTP GET, on a publicly reachable endpoint."""
    cases = resolve_cases(split)
    report = run_benchmark(cases=cases)
    return {
        "split": split,
        "total": report.total,
        "routing_accuracy": report.routing_accuracy,
        "banded_accuracy": report.banded_accuracy,
        "mean_abs_tier_error": report.mean_abs_tier_error,
        "underpowered_rate": report.underpowered_rate,
        "overkill_rate": report.overkill_rate,
        "average_cost": report.average_cost,
        "average_latency_ms": report.average_latency_ms,
        "average_confidence": report.average_confidence,
        "tier_confusion": report.tier_confusion,
        "cost_vs_always_strongest": compare_to_always_strongest(cases=cases, report=report),
        "cases": [
            {
                "id": r.case.id,
                "expected_tier": r.case.expected_tier,
                "acceptable_tiers": r.case.acceptable_tiers,
                "selected_tier": r.selected_tier,
                "selected_model": r.decision.selected.model.name,
                "correct": r.correct,
                "in_band": r.in_band,
                "difficulty": r.decision.complexity.overall,
            }
            for r in report.results
        ],
    }


@router.get("/evaluate/benchmark")
def benchmark(split: str = "dev"):
    if split not in ("dev", "test", "all"):
        raise HTTPException(status_code=422, detail="split must be dev, test or all")
    return _benchmark_payload(split)


class TournamentRequest(BaseModel):
    prompt: str
    context: str = ""


@router.post("/evaluate/tournament")
def tournament(req: TournamentRequest):
    result = run_tournament(req.prompt, req.context)
    return {
        "prompt": result.prompt,
        "entries": [e.__dict__ for e in result.entries],
    }
