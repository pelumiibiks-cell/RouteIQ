from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.evaluation.runner import compare_to_always_strongest, run_benchmark
from app.evaluation.tournament import run_tournament

router = APIRouter()


@router.get("/evaluate/benchmark")
def benchmark():
    report = run_benchmark()
    return {
        "total": report.total,
        "routing_accuracy": report.routing_accuracy,
        "underpowered_rate": report.underpowered_rate,
        "overkill_rate": report.overkill_rate,
        "average_cost": report.average_cost,
        "average_latency_ms": report.average_latency_ms,
        "average_confidence": report.average_confidence,
        "tier_confusion": report.tier_confusion,
        "cost_vs_always_strongest": compare_to_always_strongest(),
        "cases": [
            {
                "id": r.case.id,
                "expected_tier": r.case.expected_tier,
                "selected_tier": r.selected_tier,
                "selected_model": r.decision.selected.model.name,
                "correct": r.correct,
                "difficulty": r.decision.complexity.overall,
            }
            for r in report.results
        ],
    }


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
