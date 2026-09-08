from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.telemetry import RoutingRecord
from app.routing.router import RouteConstraints, route
from app.schemas import AlternativeModel, FeedbackRequest, RouteRequest, RouteResponse

router = APIRouter()

PROMPT_PREVIEW_CHARS = 500


@router.post("/route", response_model=RouteResponse)
def route_prompt(req: RouteRequest, db: Session = Depends(get_db)):
    constraints = RouteConstraints(
        max_cost=req.constraints.max_cost if req.constraints else None,
        max_latency_ms=req.constraints.max_latency_ms if req.constraints else None,
        minimum_quality=req.constraints.minimum_quality if req.constraints else None,
    )
    try:
        decision = route(req.prompt, req.context, req.attachments, constraints)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    alternatives = [
        AlternativeModel(
            model=e.model.name,
            effort=e.effort,
            utility=round(e.utility, 4),
            quality_estimate=e.match.quality_estimate,
            overkill_risk=e.match.overkill_risk,
            underpowered_risk=e.match.underpowered_risk,
            estimated_cost=e.cost_latency.cost_usd,
            estimated_latency_ms=e.cost_latency.latency_ms,
            rejected_reason=e.elimination_reason or "not selected",
        )
        for e in decision.ranked
        if e is not decision.selected
    ]

    record = RoutingRecord(
        # Truncated: this column was unbounded and every request wrote its full
        # prompt forever, including the keep-warm ping every 12 minutes.
        prompt=req.prompt[:PROMPT_PREVIEW_CHARS],
        task_features={
            "categories": decision.task_analysis.categories,
            "requirements": decision.task_analysis.requirements,
        },
        estimated_difficulty=decision.complexity.overall,
        selected_model=decision.selected.model.name,
        selected_effort=decision.selected.effort,
        confidence=decision.confidence,
        estimated_cost=decision.selected.cost_latency.cost_usd,
        estimated_latency_ms=decision.selected.cost_latency.latency_ms,
    )
    db.add(record)
    # The commit has to stay on the request path because the response carries
    # record_id and /feedback is keyed on it. db.refresh() did not: it issued a
    # second round-trip purely to read back an autoincrement primary key that
    # SQLAlchemy already populates on commit.
    #
    # Measured: db.add is 0.065ms, db.commit is 13.5ms, and route() itself is
    # 0.85ms -- so 73% of a /route request is one fsync, and the routing logic
    # is under 5%. Moving the write to a BackgroundTask does not work as-is: a
    # yield-dependency session is closed before background tasks run, and the
    # client needs the id before the response is sent either way. Removing this
    # cost means changing the contract -- return an application-generated
    # correlation id (UUID) immediately and persist in the background, with
    # /feedback keyed on that instead of the autoincrement PK. That is a public
    # API change, so it is written down here rather than made silently.
    db.commit()

    return RouteResponse(
        model=decision.selected.model.name,
        effort=decision.selected.effort,
        confidence=decision.confidence,
        difficulty=decision.complexity.overall,
        reasoning_score=decision.task_analysis.requirements["reasoning_depth"],
        categories=decision.task_analysis.categories,
        dimension_scores=decision.complexity.dimensions,
        estimated_cost=decision.selected.cost_latency.cost_usd,
        estimated_latency_ms=decision.selected.cost_latency.latency_ms,
        overkill_risk=decision.selected.match.overkill_risk,
        underpowered_risk=decision.selected.match.underpowered_risk,
        quality_estimate=decision.selected.match.quality_estimate,
        two_pass_used=decision.two_pass_used,
        alternatives=alternatives,
        explanation=decision.explanation_text,
        positive_reasons=decision.positive_reasons,
        negative_reasons=decision.negative_reasons,
        rejected_alternatives=decision.rejected_alternatives,
        record_id=record.id,
    )


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    record = db.get(RoutingRecord, req.record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record not found")
    if req.actual_result_quality is not None:
        record.actual_result_quality = req.actual_result_quality
    if req.actual_latency_ms is not None:
        record.actual_latency_ms = req.actual_latency_ms
    if req.actual_cost is not None:
        record.actual_cost = req.actual_cost
    if req.user_feedback is not None:
        record.user_feedback = req.user_feedback
    if req.success is not None:
        record.success = req.success
    db.commit()
    return {"status": "ok"}


@router.get("/models")
def list_models():
    from app.registry.registry import get_default_registry

    return [m.__dict__ for m in get_default_registry().all()]
