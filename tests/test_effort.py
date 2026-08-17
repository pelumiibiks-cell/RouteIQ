from app.registry.registry import get_default_registry
from app.routing.effort_selector import clamp_effort_to_model


def test_effort_clamped_to_model_ceiling():
    registry = get_default_registry()
    haiku = registry.get("claude-haiku-4-5")
    assert clamp_effort_to_model("max", haiku) == haiku.max_reasoning_effort == "low"


def test_effort_not_clamped_when_within_ceiling():
    registry = get_default_registry()
    fable = registry.get("claude-fable-5")
    assert clamp_effort_to_model("high", fable) == "high"
