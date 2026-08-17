from app.registry.registry import ModelRegistry


def test_loads_all_tiers():
    registry = ModelRegistry()
    tiers = {m.tier for m in registry.all()}
    assert tiers == {1, 2, 3, 4}


def test_get_by_name():
    registry = ModelRegistry()
    model = registry.get("claude-haiku-4-5")
    assert model.tier == 1
    assert model.vision is True


def test_supports_effort():
    registry = ModelRegistry()
    haiku = registry.get("claude-haiku-4-5")
    assert haiku.supports_effort("low")
    assert not haiku.supports_effort("medium")

    fable = registry.get("claude-fable-5")
    assert fable.supports_effort("max")
