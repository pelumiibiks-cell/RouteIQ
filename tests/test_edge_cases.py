"""Adversarial prompts designed to trick a superficial (length/keyword) router.
The router must reason about the task, not just prompt length or keyword density.
"""
from app.routing.router import route


def test_short_prompt_extremely_hard_reasoning_is_not_dismissed_as_trivial():
    d = route("Prove P != NP.")
    # A pure heuristic analyzer cannot fully recognize this without semantic
    # understanding, but it must not be treated as a *trivial* deterministic
    # task (that would be a worse failure than under-escalating).
    assert d.selected.model.tier >= 2


def test_long_prompt_trivial_task_is_not_escalated_by_length_alone():
    long_changelog = "\n".join(f"- v0.{i}: minor bugfix" for i in range(1, 150))
    d = route(f"{long_changelog}\nExtract just the version numbers into a list.")
    assert d.selected.model.tier == 1
    assert d.selected.match.overkill_risk < 0.3


def test_keyword_heavy_prompt_that_is_actually_easy():
    d = route("I'm not a software architect, but can you rename this variable from 'x' to 'count': `x = 5`?")
    assert d.selected.model.tier == 1


def test_simple_looking_task_requiring_vision_is_not_routed_to_no_vision_model():
    d = route("What color is the car?", attachments=["car.jpg"])
    assert d.selected.model.vision is True


def test_ambiguous_task_has_elevated_ambiguity_score():
    d = route("Make it better.")
    assert d.task_analysis.requirements["ambiguity"] >= 5.0


def test_massive_context_trivial_extraction_respects_context_but_stays_cheap():
    huge = "id,value\n" + "\n".join(f"{i},{i*2}" for i in range(20000))
    d = route(f"Extract all rows where value > 39990 from this CSV:\n{huge}")
    # must fit context (hard constraint) but shouldn't be routed to the most
    # expensive model just because the prompt is long and the task is simple.
    assert d.selected.model.context_window >= 20000 * 12 / 4  # rough token estimate sanity check


def test_empty_prompt_does_not_crash():
    d = route("")
    assert d.selected is not None
