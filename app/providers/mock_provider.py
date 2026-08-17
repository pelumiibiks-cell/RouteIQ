"""Mock provider: simulates generation for demo/eval purposes without any
network calls or API keys. Used by the evaluation/tournament framework to
produce reproducible cost/latency/quality figures. Swap for a real provider
adapter (implementing the same ModelProvider interface) to go live.
"""
from __future__ import annotations

import random

from app.providers.base import GenerationResult
from app.registry.model_profile import ModelProfile

EFFORT_LATENCY_MULTIPLIER = {
    "low": 0.5,
    "medium": 0.85,
    "high": 1.3,
    "xhigh": 1.8,
    "max": 2.5,
}

EFFORT_OUTPUT_MULTIPLIER = {
    "low": 0.4,
    "medium": 0.8,
    "high": 1.2,
    "xhigh": 1.7,
    "max": 2.3,
}


class MockProvider:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self, model: ModelProfile, prompt: str, effort: str) -> GenerationResult:
        input_tokens = max(1, len(prompt) // 4)
        base_output = max(20, input_tokens // 3)
        output_tokens = int(base_output * EFFORT_OUTPUT_MULTIPLIER.get(effort, 1.0))

        base_latency = 300 + input_tokens * 0.8 + output_tokens * 1.5
        latency = base_latency * EFFORT_LATENCY_MULTIPLIER.get(effort, 1.0) * (11 - model.latency_score) / 5
        latency *= self._rng.uniform(0.9, 1.1)

        text = f"[mock output from {model.name} @ {effort} effort]"
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency, 1),
        )

    def estimate_cost(self, model: ModelProfile, input_tokens: int, output_tokens: int) -> float:
        return round(
            (input_tokens / 1000) * model.cost_per_input_token
            + (output_tokens / 1000) * model.cost_per_output_token,
            6,
        )

    def get_capabilities(self, model: ModelProfile) -> dict:
        return {
            "vision": model.vision,
            "audio": model.audio,
            "tool_use": model.tool_use,
            "structured_output": model.structured_output,
            "context_window": model.context_window,
            "max_reasoning_effort": model.max_reasoning_effort,
        }
