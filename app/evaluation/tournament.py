"""Model tournament: run the same prompt against every registered model
(via the mock provider) and compare quality/cost/latency, independent of
what the router would have picked. Useful for validating/improving the
router's quality_estimate function against something resembling real
outcomes once a real provider is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.analysis.complexity_scorer import score as score_complexity
from app.analysis.normalizer import normalize
from app.analysis.task_analyzer import analyze
from app.providers.mock_provider import MockProvider
from app.registry.model_profile import ModelProfile
from app.registry.registry import ModelRegistry, get_default_registry
from app.routing.capability_matcher import match
from app.routing.effort_selector import clamp_effort_to_model, select_effort


@dataclass
class TournamentEntry:
    model: str
    tier: int
    effort: str
    eliminated: bool
    elimination_reason: str | None
    quality_estimate: float
    simulated_output_tokens: int
    simulated_latency_ms: float
    simulated_cost_usd: float


@dataclass
class TournamentResult:
    prompt: str
    entries: list[TournamentEntry]


def run_tournament(prompt: str, context: str = "", registry: ModelRegistry | None = None, seed: int = 42) -> TournamentResult:
    registry = registry or get_default_registry()
    provider = MockProvider(seed=seed)

    normalized = normalize(prompt, context)
    task_analysis = analyze(normalized)
    complexity = score_complexity(task_analysis)
    desired_effort = select_effort(task_analysis, complexity)

    entries = []
    for model in registry.all():
        m = match(task_analysis, complexity, model)
        effort = clamp_effort_to_model(desired_effort, model)
        if m.eliminated:
            entries.append(TournamentEntry(model.name, model.tier, effort, True, m.elimination_reason, 0.0, 0, 0.0, 0.0))
            continue
        gen = provider.generate(model, normalized.text, effort)
        cost = provider.estimate_cost(model, gen.input_tokens, gen.output_tokens)
        entries.append(
            TournamentEntry(
                model=model.name, tier=model.tier, effort=effort, eliminated=False, elimination_reason=None,
                quality_estimate=m.quality_estimate, simulated_output_tokens=gen.output_tokens,
                simulated_latency_ms=gen.latency_ms, simulated_cost_usd=cost,
            )
        )

    entries.sort(key=lambda e: (-e.quality_estimate, e.simulated_cost_usd))
    return TournamentResult(prompt=prompt, entries=entries)
