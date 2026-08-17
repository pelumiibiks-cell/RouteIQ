from __future__ import annotations

from pathlib import Path

import yaml

from app.registry.model_profile import ModelProfile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class ModelRegistry:
    """Loads model capability/cost metadata from configuration (YAML/JSON).

    Adding a new model = adding an entry to config/models.yaml, no code
    changes required. Providers registered separately (see app/providers)
    supply the actual `generate()` implementation per model.
    """

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH):
        self._config_path = Path(config_path)
        self._models: dict[str, ModelProfile] = {}
        self.reload()

    def reload(self) -> None:
        with open(self._config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        models = {}
        for entry in data.get("models", []):
            profile = ModelProfile(**entry)
            models[profile.name] = profile
        self._models = models

    def get(self, name: str) -> ModelProfile:
        return self._models[name]

    def all(self) -> list[ModelProfile]:
        return list(self._models.values())

    def by_tier(self, tier: int) -> list[ModelProfile]:
        return [m for m in self._models.values() if m.tier == tier]

    def names(self) -> list[str]:
        return list(self._models.keys())


_default_registry: ModelRegistry | None = None


def get_default_registry() -> ModelRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry
