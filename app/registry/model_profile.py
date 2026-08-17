from __future__ import annotations

from dataclasses import dataclass

EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    tier: int
    reasoning_score: float
    coding_score: float
    math_score: float
    instruction_following_score: float
    creativity_score: float
    vision: bool
    audio: bool
    tool_use: bool
    structured_output: bool
    context_window: int
    latency_score: float  # 0-10, higher = faster
    cost_per_input_token: float  # USD per 1K input tokens
    cost_per_output_token: float  # USD per 1K output tokens
    reliability_score: float
    max_reasoning_effort: str

    def supports_effort(self, effort: str) -> bool:
        return EFFORT_LEVELS.index(effort) <= EFFORT_LEVELS.index(self.max_reasoning_effort)

    def capability_score(self, dimension: str) -> float:
        mapping = {
            "reasoning": self.reasoning_score,
            "coding": self.coding_score,
            "math": self.math_score,
            "instruction_following": self.instruction_following_score,
            "creativity": self.creativity_score,
        }
        return mapping.get(dimension, self.reasoning_score)
