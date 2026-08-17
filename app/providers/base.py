from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.registry.model_profile import ModelProfile


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ModelProvider(Protocol):
    """Common interface every provider adapter implements. The router never
    calls a provider SDK directly -- it only depends on this interface, so
    swapping/adding providers (OpenAI-like, Anthropic-like, local, self
    -hosted) never touches routing logic.
    """

    def generate(self, model: ModelProfile, prompt: str, effort: str) -> GenerationResult: ...

    def estimate_cost(self, model: ModelProfile, input_tokens: int, output_tokens: int) -> float: ...

    def get_capabilities(self, model: ModelProfile) -> dict: ...
