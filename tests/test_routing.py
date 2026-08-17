from app.routing.router import RouteConstraints, route


def test_trivial_task_does_not_pick_frontier_model():
    d = route("Convert 25 USD to EUR.")
    assert d.selected.model.tier <= 2
    assert d.selected.match.overkill_risk < 0.3


def test_hard_concurrency_bug_escalates_beyond_lightweight():
    d = route(
        "Find the subtle concurrency bugs in this distributed Python system. "
        "Explain the race conditions and propose a safe redesign:\n"
        "```python\nclass Counter:\n    def __init__(self):\n        self.value = 0\n"
        "    def increment(self):\n        current = self.value\n        time.sleep(0.001)\n"
        "        self.value = current + 1\n```\n"
        "This is called concurrently from 50 worker threads and the final count is wrong."
    )
    assert d.selected.model.tier >= 3


def test_vision_requirement_eliminates_non_vision_models():
    d = route("What color is the car in this photo?", attachments=["car.jpg"])
    assert d.selected.model.vision is True
    for e in d.ranked:
        if not e.model.vision:
            assert False, "a non-vision model should never survive a vision-required task"


def test_context_window_hard_constraint():
    huge_text = "x " * 400000  # ~800k chars, exceeds even the biggest configured window comfortably estimated
    d = route(f"Summarize this document:\n{huge_text}")
    # Only models whose context window can plausibly fit this should remain.
    assert d.selected.model.context_window >= 100000


def test_effort_is_not_maximum_by_default_for_easy_tasks():
    d = route("Translate this sentence into French: The weather is nice today.")
    assert d.selected.effort == "low"


def test_explanation_and_alternatives_present():
    d = route("Design a distributed caching architecture for a high-traffic platform.")
    assert d.explanation_text
    assert isinstance(d.rejected_alternatives, list)


def test_max_cost_constraint_prefers_cheaper_models():
    unconstrained = route("Design a production-grade agentic system with autonomous multi-step planning.")
    constrained = route(
        "Design a production-grade agentic system with autonomous multi-step planning.",
        constraints=RouteConstraints(max_cost=0.0005),
    )
    assert constrained.selected.cost_latency.cost_usd <= unconstrained.selected.cost_latency.cost_usd


def test_confidence_within_bounds():
    d = route("Summarize this paragraph.")
    assert 0.0 <= d.confidence <= 1.0
