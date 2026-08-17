"""Task analysis: category classification + requirement extraction.

Design note
------------
This is a *rule-based semantic analyzer*: it combines many weighted signals
(domain vocabulary density, structural features, discourse markers,
conjunction/step counting, negation/ambiguity markers, code/data presence)
rather than a single keyword lookup. It is intentionally built behind the
`TaskAnalyzer` interface so it can be swapped for an LLM-backed analyzer
(see `LLMBackedAnalyzer` stub below) without touching the rest of the
pipeline -- that swap is the natural evolution path described in the README.

It deliberately does NOT do `if "bug" in prompt: category = coding`. Every
signal is a weighted vote across multiple categories/dimensions, and no
single keyword can push a score past a moderate ceiling on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.analysis.normalizer import NormalizedPrompt

CATEGORIES = [
    "qa",
    "summarization",
    "translation",
    "classification",
    "extraction",
    "coding",
    "debugging",
    "software_architecture",
    "math_reasoning",
    "data_analysis",
    "research",
    "long_context_analysis",
    "agentic",
    "creative_generation",
    "planning",
    "multimodal_reasoning",
    "image_understanding",
    "document_understanding",
    "tool_usage",
    "multi_step_reasoning",
]

REQUIREMENT_FIELDS = [
    "reasoning_depth",
    "context_length",
    "coding_complexity",
    "mathematical_complexity",
    "research_requirement",
    "multimodal_requirement",
    "instruction_complexity",
    "number_of_steps",
    "ambiguity",
    "precision_requirement",
    "factuality_requirement",
    "creativity_requirement",
    "tool_usage_requirement",
    "agentic_requirement",
    "output_complexity",
    "domain_specialization",
    "latency_sensitivity",
    "cost_sensitivity",
    "reliability_requirement",
]


@dataclass
class CategorySignal:
    category: str
    weight: float
    reason: str


@dataclass
class TaskAnalysis:
    normalized: NormalizedPrompt
    categories: list[str]
    category_scores: dict[str, float]
    requirements: dict[str, float]
    signals: list[CategorySignal] = field(default_factory=list)


# Vocabulary bundles used as *signals*, each with a modest weight. Multiple
# independent signals must agree before a category/dimension score gets high.
_VOCAB = {
    "coding": [
        "function", "class", "variable", "compile", "runtime", "stack trace",
        "api", "endpoint", "rest api", "repository", "repo", "codebase", "refactor",
        "implement", "python", "javascript", "typescript", "java ", "c++",
        "sql", "algorithm", "data structure", "unit test", "pytest", "npm",
        "package", "dependency", "syntax error", "null pointer", "json",
        "validates", "validate", "fastapi", "flask", "django", "http request",
        "post request", "status code", "request body", "middleware",
        "```",
    ],
    "debugging": [
        "bug", "error", "exception", "crash", "fails", "failing", "traceback",
        "stack trace", "race condition", "deadlock", "memory leak",
        "doesn't work", "not working", "unexpected behavior", "reproduce",
        "root cause", "fix this", "why is this", "concurrency", "concurrent",
        "thread-safe", "synchroniz", "subtle bug", "flaky", "intermittent",
    ],
    "software_architecture": [
        "architecture", "architectural", "modular structure", "microservice", "distributed system", "distributed python",
        "distributed payments", "distributed caching", "scalab", "design pattern",
        "migration plan", "system design", "load balanc", "caching layer",
        "cache invalidation", "consistency", "message queue",
        "event-driven", "monolith", "high availability", "fault toleran",
        "throughput", "latency budget", "redesign", "cascading failure",
        "outage", "warehouse", "logistics network", "high-traffic", "failure modes",
        "multiple regions", "across regions",
    ],
    "math_reasoning": [
        "prove", "theorem", "equation", "integral", "derivative", "matrix",
        "probability", "combinatorics", "optimi", "solve for", "algebra",
        "calculus", "geometry", "number theory", "p != np", "p vs np",
        "np-hard", "np-complete", "millennium", "formulate the problem",
        "mathematically", "stochastic", "conjecture", "primality",
        "show your reasoning", "prime factor", "modular arithmetic",
    ],
    "data_analysis": [
        "csv", "dataset", "dataframe", "correlation", "regression",
        "statistics", "outlier", "distribution", "pivot table", "analyze this data",
        "trend", "aggregat", "time series", "sales data", "class imbalance",
        "training pipeline", "retrain", "ml pipeline", "fraud-detection",
        "fraud detection", "online evaluation", "streaming transaction",
    ],
    "research": [
        "research", "literature", "compare approaches", "state of the art",
        "survey of", "cite", "evidence", "study shows", "investigate",
        "causal chain", "contributing", "root-cause analysis",
    ],
    "agentic": [
        "autonomous", "agent", "multi-agent", "orchestrat", "workflow",
        "tool call", "execute steps", "plan and execute", "self-correct",
    ],
    "tool_usage": [
        "call the api", "use a tool", "invoke", "function call", "search the web",
        "query the database", "run a script",
    ],
    "planning": [
        "plan", "roadmap", "strategy", "timeline", "milestones", "step by step plan",
        "propose a plan",
    ],
    "creative_generation": [
        "write a story", "poem", "creative", "brainstorm", "slogan", "narrative",
        "fictional", "compose a", "write a song",
    ],
    "translation": [
        "translate", "in french", "in spanish", "into german", "into japanese",
        "in english", "translation of",
    ],
    "summarization": [
        "summarize", "summary", "tl;dr", "condense", "key points", "brief overview",
    ],
    "classification": [
        "classify", "categorize", "label this", "is this spam", "sentiment",
        "which category",
    ],
    "extraction": [
        "extract", "pull out", "list all", "find all", "identify all instances",
    ],
    "document_understanding": [
        "this document", "this pdf", "this contract", "this report", "attached document",
    ],
    "image_understanding": [
        "this image", "this photo", "this screenshot", "in the picture", "diagram shows",
    ],
    "multimodal_reasoning": [
        "audio", "video", "image and text", "chart shown", "visual",
    ],
    "long_context_analysis": [
        "entire repository", "entire codebase", "the whole document", "full transcript",
        "2,000-line", "2000-line", "large codebase", "whole log file",
    ],
    "multi_step_reasoning": [
        "step by step", "first,", "then,", "after that", "finally,", "explain why",
        "walk through", "reason through",
    ],
}

_AMBIGUITY_MARKERS = [
    "maybe", "not sure", "something like", "or whatever", "i think", "kind of",
    "roughly", "vague", "open-ended", "up to you", "your best judgment",
]

_PRECISION_MARKERS = [
    "exact", "precisely", "must match", "strict", "no errors", "production-grade",
    "correctness", "guarantee", "verify", "rigorous",
]

_HIGH_STAKES_MARKERS = [
    "production", "critical", "safety", "financial", "medical", "legal",
    "cannot fail", "mission-critical", "compliance",
]

_STEP_CONJUNCTIONS = ["and then", "after that", "once done", "next,", "finally,", "also,"]

_TRIVIAL_TASK_MARKERS = [
    "convert", "translate this sentence", "what is the capital", "define ",
    "spell check", "capitalize", "how many words",
]


def _count_hits(text_lower: str, phrases: list[str]) -> int:
    return sum(1 for p in phrases if p in text_lower)


def analyze(normalized: NormalizedPrompt) -> TaskAnalysis:
    text = f"{normalized.text}\n{normalized.context}".lower()
    signals: list[CategorySignal] = []
    category_scores: dict[str, float] = {c: 0.0 for c in CATEGORIES}

    # --- vocabulary-driven category signals (capped contribution per bundle) ---
    for category, vocab in _VOCAB.items():
        hits = _count_hits(text, vocab)
        if hits:
            contribution = min(2.2 * hits ** 0.7, 9.0)
            category_scores[category] += contribution
            signals.append(CategorySignal(category, contribution, f"{hits} domain-term hits"))

    # --- structural signals ---
    if normalized.has_code_block:
        category_scores["coding"] += 3.0
        signals.append(CategorySignal("coding", 3.0, "contains fenced code block"))
        if normalized.line_count_in_code > 80:
            category_scores["long_context_analysis"] += 2.5
            category_scores["software_architecture"] += 1.0
            signals.append(CategorySignal("long_context_analysis", 2.5, "large code block (>80 lines)"))

    if normalized.has_data_block:
        category_scores["data_analysis"] += 2.0
        category_scores["extraction"] += 1.0

    if normalized.numbered_steps >= 2:
        category_scores["planning"] += 1.5
        category_scores["multi_step_reasoning"] += 2.0

    if normalized.char_count > 4000:
        category_scores["long_context_analysis"] += 3.0
        signals.append(CategorySignal("long_context_analysis", 3.0, "very long input (>4000 chars)"))
    elif normalized.char_count > 1500:
        category_scores["long_context_analysis"] += 1.2

    if any(att for att in normalized.attachments):
        category_scores["document_understanding"] += 1.5
        category_scores["multimodal_reasoning"] += 1.0

    step_conj_hits = _count_hits(text, _STEP_CONJUNCTIONS)
    if step_conj_hits:
        category_scores["multi_step_reasoning"] += min(step_conj_hits * 1.2, 4.0)

    if normalized.question_marks >= 3:
        category_scores["qa"] += 1.5

    # default fallback: plain QA if nothing else fired strongly
    if max(category_scores.values()) < 1.5:
        category_scores["qa"] += 3.0

    # select categories: anything within a reasonable band of the max, capped
    top = max(category_scores.values())
    threshold = max(1.5, top * 0.45)
    selected = sorted(
        [c for c, s in category_scores.items() if s >= threshold],
        key=lambda c: -category_scores[c],
    )[:5]
    if not selected:
        selected = ["qa"]

    requirements = _extract_requirements(normalized, text, category_scores, selected)

    return TaskAnalysis(
        normalized=normalized,
        categories=selected,
        category_scores=category_scores,
        requirements=requirements,
        signals=signals,
    )


def _clip(v: float) -> float:
    return max(0.0, min(10.0, v))


def _extract_requirements(
    normalized: NormalizedPrompt,
    text: str,
    category_scores: dict[str, float],
    selected: list[str],
) -> dict[str, float]:
    words = max(1, normalized.word_count)

    # reasoning_depth: driven by multi-step signals, debugging/architecture/math presence, step conjunctions
    reasoning_depth = 1.5
    reasoning_depth += min(category_scores["multi_step_reasoning"], 9) * 0.55
    reasoning_depth += min(category_scores["debugging"], 9) * 0.4
    reasoning_depth += min(category_scores["software_architecture"], 9) * 0.65
    reasoning_depth += min(category_scores["math_reasoning"], 9) * 0.65
    reasoning_depth += min(category_scores["agentic"], 9) * 0.65
    reasoning_depth += min(category_scores["research"], 9) * 0.4
    reasoning_depth += min(normalized.numbered_steps, 6) * 0.3

    # context_length: from actual char/word volume + long-context signal, not just "long prompt = hard task"
    if normalized.char_count > 20000:
        context_length = 9.5
    elif normalized.char_count > 6000:
        context_length = 7.5
    elif normalized.char_count > 2500:
        context_length = 5.0
    elif normalized.char_count > 800:
        context_length = 2.5
    else:
        context_length = 1.0
    context_length = _clip(context_length + min(category_scores["long_context_analysis"], 6) * 0.3)

    coding_complexity = _clip(
        min(category_scores["coding"], 9) * 0.8
        + min(category_scores["debugging"], 9) * 0.35
        + min(category_scores["software_architecture"], 9) * 0.6
        + (2.0 if normalized.line_count_in_code > 50 else 0.0)
    )

    mathematical_complexity = _clip(min(category_scores["math_reasoning"], 9) * 1.1)

    research_requirement = _clip(min(category_scores["research"], 9) * 1.1)

    multimodal_requirement = _clip(
        min(category_scores["multimodal_reasoning"], 9) * 0.8
        + min(category_scores["image_understanding"], 9) * 0.9
        + (4.5 if normalized.attachments else 0.0)
    )

    instruction_complexity = _clip(1.0 + min(normalized.numbered_steps, 8) * 0.5 + min(normalized.sentence_count / 4, 5))

    number_of_steps = _clip(1.0 + normalized.numbered_steps * 0.9 + category_scores["multi_step_reasoning"] * 0.4)

    ambiguity_hits = _count_hits(text, _AMBIGUITY_MARKERS)
    precision_hits = _count_hits(text, _PRECISION_MARKERS)
    is_trivial_task = any(m in text for m in _TRIVIAL_TASK_MARKERS)
    ambiguity = _clip(
        1.5 + ambiguity_hits * 1.8 - precision_hits * 0.8
        + (3.5 if words < 6 and not is_trivial_task else 0.0)
        - (1.0 if words > 40 else 0.0)
    )
    # very short prompts are often ambiguous unless they are clearly deterministic
    if words <= 8 and not is_trivial_task:
        ambiguity = _clip(ambiguity + 1.5)

    precision_requirement = _clip(2.0 + precision_hits * 2.0 + min(category_scores["coding"], 6) * 0.3 + min(category_scores["math_reasoning"], 6) * 0.3)

    factuality_requirement = _clip(2.0 + min(category_scores["research"], 6) * 0.6 + min(category_scores["qa"], 6) * 0.3 + _count_hits(text, _HIGH_STAKES_MARKERS) * 1.5)

    creativity_requirement = _clip(min(category_scores["creative_generation"], 8) * 1.2)

    tool_usage_requirement = _clip(min(category_scores["tool_usage"], 8) * 1.2 + min(category_scores["agentic"], 8) * 0.4)

    agentic_requirement = _clip(min(category_scores["agentic"], 8) * 1.3)

    output_complexity = _clip(
        1.0
        + min(category_scores["planning"], 6) * 0.5
        + min(category_scores["software_architecture"], 6) * 0.5
        + min(normalized.numbered_steps, 6) * 0.3
    )

    domain_specialization = _clip(
        max(
            min(category_scores["software_architecture"], 8) * 0.9,
            min(category_scores["math_reasoning"], 8) * 0.8,
            min(category_scores["research"], 8) * 0.8,
            min(category_scores["data_analysis"], 8) * 0.7,
        )
    )

    latency_sensitivity = 3.0  # neutral default; real system would read explicit constraints
    cost_sensitivity = 3.0

    reliability_requirement = _clip(3.0 + _count_hits(text, _HIGH_STAKES_MARKERS) * 2.0 + precision_hits * 1.0)

    return {
        "reasoning_depth": _clip(reasoning_depth),
        "context_length": context_length,
        "coding_complexity": coding_complexity,
        "mathematical_complexity": mathematical_complexity,
        "research_requirement": research_requirement,
        "multimodal_requirement": multimodal_requirement,
        "instruction_complexity": instruction_complexity,
        "number_of_steps": number_of_steps,
        "ambiguity": ambiguity,
        "precision_requirement": precision_requirement,
        "factuality_requirement": factuality_requirement,
        "creativity_requirement": creativity_requirement,
        "tool_usage_requirement": tool_usage_requirement,
        "agentic_requirement": agentic_requirement,
        "output_complexity": output_complexity,
        "domain_specialization": domain_specialization,
        "latency_sensitivity": latency_sensitivity,
        "cost_sensitivity": cost_sensitivity,
        "reliability_requirement": reliability_requirement,
    }


@dataclass
class QuickAnalysis:
    normalized: NormalizedPrompt
    categories: list[str]
    category_scores: dict[str, float]
    rough_difficulty: float


def quick_analyze(normalized: NormalizedPrompt) -> QuickAnalysis:
    """Pass 1: cheap analysis. Computes category signals and a rough
    difficulty estimate WITHOUT the full 19-dimension requirement
    extraction, so it stays fast. Used to decide whether the more
    expensive Pass 2 (`analyze`) is actually necessary.
    """
    text = f"{normalized.text}\n{normalized.context}".lower()
    category_scores: dict[str, float] = {c: 0.0 for c in CATEGORIES}

    for category, vocab in _VOCAB.items():
        hits = _count_hits(text, vocab)
        if hits:
            # Sub-linear in hit count: a few mentions of the same idea
            # (synonyms for "this is buggy") shouldn't stack as fast as
            # genuinely distinct signals spread across the prompt.
            category_scores[category] += min(2.2 * hits ** 0.7, 9.0)

    if normalized.has_code_block:
        category_scores["coding"] += 3.0
    if normalized.char_count > 4000:
        category_scores["long_context_analysis"] += 3.0
    if normalized.numbered_steps >= 2:
        category_scores["multi_step_reasoning"] += 2.0

    if max(category_scores.values()) < 1.5:
        category_scores["qa"] += 3.0

    top = max(category_scores.values())
    threshold = max(1.5, top * 0.45)
    selected = sorted(
        [c for c, s in category_scores.items() if s >= threshold],
        key=lambda c: -category_scores[c],
    )[:5] or ["qa"]

    # Rough difficulty: coarse proxy from top category signal strength +
    # structural volume, deliberately cheaper than the full requirement scorer.
    hard_categories = {"software_architecture", "debugging", "math_reasoning", "agentic", "multi_step_reasoning", "long_context_analysis"}
    hard_signal = sum(category_scores[c] for c in selected if c in hard_categories)
    size_signal = min(4.0, normalized.char_count / 2000)
    rough = 1.5 + min(hard_signal, 12) * 0.4 + size_signal * 0.5
    rough = max(0.0, min(10.0, rough))

    return QuickAnalysis(normalized=normalized, categories=selected, category_scores=category_scores, rough_difficulty=round(rough, 2))


def approximate_task_analysis(quick: QuickAnalysis) -> TaskAnalysis:
    """Builds a coarse TaskAnalysis straight from the cheap Pass 1 signals,
    skipping the finer-grained marker scanning (ambiguity/precision phrase
    detection etc.) that `_extract_requirements` does. Used when the router
    decides Pass 2 isn't warranted because the task isn't borderline.
    """
    normalized = quick.normalized
    scores = quick.category_scores
    words = max(1, normalized.word_count)
    text = f"{normalized.text}\n{normalized.context}".lower()

    reasoning_depth = _clip(1.5 + scores["multi_step_reasoning"] * 0.5 + scores["debugging"] * 0.5 + scores["software_architecture"] * 0.6 + scores["math_reasoning"] * 0.6 + scores["agentic"] * 0.6 + scores["research"] * 0.3)
    context_length = _clip(quick.rough_difficulty * 0.6 + min(scores["long_context_analysis"], 6) * 0.4)
    coding_complexity = _clip(scores["coding"] * 0.6 + scores["debugging"] * 0.4 + scores["software_architecture"] * 0.4)
    mathematical_complexity = _clip(scores["math_reasoning"] * 1.0)
    research_requirement = _clip(scores["research"] * 1.0)
    multimodal_requirement = _clip(scores["multimodal_reasoning"] * 0.8 + scores["image_understanding"] * 0.8 + (4.5 if normalized.attachments else 0.0))
    instruction_complexity = _clip(1.0 + min(normalized.numbered_steps, 8) * 0.5 + min(normalized.sentence_count / 4, 5))
    number_of_steps = _clip(1.0 + normalized.numbered_steps * 0.9 + scores["multi_step_reasoning"] * 0.4)
    _is_trivial = any(m in text for m in _TRIVIAL_TASK_MARKERS)
    ambiguity = _clip(2.0 + (3.5 if words < 6 and not _is_trivial else 0.0) + (1.5 if words <= 8 and not _is_trivial else 0.0))
    precision_requirement = _clip(2.0 + scores["coding"] * 0.2 + scores["math_reasoning"] * 0.2)
    factuality_requirement = _clip(2.0 + scores["research"] * 0.5 + scores["qa"] * 0.3)
    creativity_requirement = _clip(scores["creative_generation"] * 1.0)
    tool_usage_requirement = _clip(scores["tool_usage"] * 1.0 + scores["agentic"] * 0.3)
    agentic_requirement = _clip(scores["agentic"] * 1.1)
    output_complexity = _clip(1.0 + scores["planning"] * 0.4 + scores["software_architecture"] * 0.4)
    domain_specialization = _clip(max(scores["software_architecture"], scores["math_reasoning"], scores["research"], scores["data_analysis"]) * 0.7)
    reliability_requirement = _clip(3.0)

    requirements = {
        "reasoning_depth": reasoning_depth,
        "context_length": context_length,
        "coding_complexity": coding_complexity,
        "mathematical_complexity": mathematical_complexity,
        "research_requirement": research_requirement,
        "multimodal_requirement": multimodal_requirement,
        "instruction_complexity": instruction_complexity,
        "number_of_steps": number_of_steps,
        "ambiguity": ambiguity,
        "precision_requirement": precision_requirement,
        "factuality_requirement": factuality_requirement,
        "creativity_requirement": creativity_requirement,
        "tool_usage_requirement": tool_usage_requirement,
        "agentic_requirement": agentic_requirement,
        "output_complexity": output_complexity,
        "domain_specialization": domain_specialization,
        "latency_sensitivity": 3.0,
        "cost_sensitivity": 3.0,
        "reliability_requirement": reliability_requirement,
    }

    return TaskAnalysis(
        normalized=normalized,
        categories=quick.categories,
        category_scores=scores,
        requirements=requirements,
        signals=[],
    )


class TaskAnalyzer:
    """Pluggable interface. `HeuristicTaskAnalyzer` is the default (Pass 1/2)."""

    def analyze(self, normalized: NormalizedPrompt) -> TaskAnalysis:
        raise NotImplementedError


class HeuristicTaskAnalyzer(TaskAnalyzer):
    def analyze(self, normalized: NormalizedPrompt) -> TaskAnalysis:
        return analyze(normalized)


class LLMBackedAnalyzer(TaskAnalyzer):
    """Stub for a future analyzer that calls a cheap model for Pass 1/2
    classification instead of heuristics. Not implemented: no live provider
    credentials in this environment. Swapping this in requires no change to
    downstream routing code, since both return a `TaskAnalysis`.
    """

    def analyze(self, normalized: NormalizedPrompt) -> TaskAnalysis:
        raise NotImplementedError("LLM-backed analysis requires a configured provider.")
