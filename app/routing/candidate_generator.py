from __future__ import annotations

from app.registry.model_profile import ModelProfile
from app.registry.registry import ModelRegistry


def generate_candidates(registry: ModelRegistry) -> list[ModelProfile]:
    """Every registered model is a candidate; elimination happens later in
    capability matching. Kept as its own stage so future logic (e.g. only
    consider models allowed by an org policy) can filter here without
    touching matching/scoring.
    """
    return registry.all()
