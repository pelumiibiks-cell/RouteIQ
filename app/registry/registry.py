from __future__ import annotations

from pathlib import Path

import yaml

from app.registry.model_profile import ModelProfile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class RegistryError(ValueError):
    """Raised when the model registry file is missing, malformed or inconsistent."""


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
        """Validates as it loads. A malformed registry used to surface as a
        TypeError from a dataclass constructor, or worse, as a duplicate name
        silently overwriting an earlier model."""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise RegistryError(f"model registry not found at {self._config_path}") from exc
        except yaml.YAMLError as exc:
            raise RegistryError(f"model registry at {self._config_path} is not valid YAML: {exc}") from exc

        if not isinstance(data, dict) or not data.get("models"):
            raise RegistryError(f"model registry at {self._config_path} defines no models")

        models: dict[str, ModelProfile] = {}
        for i, entry in enumerate(data["models"]):
            try:
                profile = ModelProfile(**entry)
            except TypeError as exc:
                raise RegistryError(f"model #{i} in {self._config_path} has invalid fields: {exc}") from exc
            if profile.name in models:
                raise RegistryError(f"duplicate model name {profile.name!r} in {self._config_path}")
            if profile.context_window <= 0:
                raise RegistryError(f"{profile.name}: context_window must be positive")
            if profile.provider != "local" and profile.cost_per_input_token <= 0:
                raise RegistryError(f"{profile.name}: non-local model needs a positive cost_per_input_token")
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
