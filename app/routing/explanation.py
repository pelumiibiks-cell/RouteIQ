from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Explanation:
    summary: str
    positive_reasons: list[str] = field(default_factory=list)
    negative_reasons: list[str] = field(default_factory=list)
    rejected_alternatives: list[str] = field(default_factory=list)


def build_explanation(
    selected_model_name: str,
    effort: str,
    complexity_overall: float,
    drivers: list[str],
    quality_estimate: float,
    overkill_risk: float,
    underpowered_risk: float,
    rejected: list[tuple[str, str]],
    two_pass_used: bool,
) -> Explanation:
    positives = []
    driver_labels = {
        "reasoning": "requires multi-step reasoning",
        "context": "requires substantial context",
        "coding_complexity": "significant code analysis/generation",
        "math_complexity": "nontrivial mathematical reasoning",
        "planning_complexity": "multi-part planning/output structure",
        "tool_agent_complexity": "tool use / agentic behavior",
        "multimodal_complexity": "multimodal (vision/audio) understanding",
        "precision_requirement": "high correctness requirement",
        "ambiguity": "ambiguous task needing judgment",
        "reliability_requirement": "high-stakes / reliability-sensitive",
    }
    for d in drivers:
        if d in driver_labels:
            positives.append(driver_labels[d])
    if not positives:
        positives.append("low-to-moderate complexity across all dimensions")

    negatives = []
    if overkill_risk >= 0.3:
        negatives.append("a stronger model would provide negligible additional value for this task")
    if underpowered_risk >= 0.3:
        negatives.append("weaker models carry meaningful risk of an unsatisfactory result")

    rejected_lines = [f"{name} → {reason}" for name, reason in rejected]

    pass_note = " (deeper Pass 2 analysis was run because the task was borderline between tiers)" if two_pass_used else ""
    summary = (
        f"Selected {selected_model_name} at {effort} effort for a task with overall "
        f"difficulty {complexity_overall}/10{pass_note}. Estimated quality {round(quality_estimate * 100)}%."
    )

    return Explanation(
        summary=summary,
        positive_reasons=positives,
        negative_reasons=negatives,
        rejected_alternatives=rejected_lines,
    )
