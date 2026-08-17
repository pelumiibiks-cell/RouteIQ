from app.analysis.complexity_scorer import score
from app.analysis.normalizer import normalize
from app.analysis.task_analyzer import analyze


def _analyze(prompt: str, context: str = "", attachments=None):
    normalized = normalize(prompt, context, attachments or [])
    return analyze(normalized)


def test_easy_task_has_low_complexity():
    a = _analyze("Convert 25 USD to EUR.")
    c = score(a)
    assert c.overall < 4.0


def test_coding_prompt_detected():
    a = _analyze("Fix this bug:\n```python\ndef f(): return 1/0\n```")
    assert "coding" in a.categories or "debugging" in a.categories
    assert a.requirements["coding_complexity"] > 0


def test_multimodal_requirement_from_attachment():
    a = _analyze("What color is the car?", attachments=["car.jpg"])
    assert a.requirements["multimodal_requirement"] >= 4.0


def test_long_prompt_increases_context_requirement():
    long_text = "word " * 3000
    a = _analyze(f"Summarize this: {long_text}")
    assert a.requirements["context_length"] >= 5.0


def test_ambiguous_short_prompt_scores_high_ambiguity():
    a = _analyze("Make it better.")
    assert a.requirements["ambiguity"] >= 5.0


def test_trivial_short_prompt_does_not_score_high_ambiguity():
    a = _analyze("Convert 25 USD to EUR.")
    assert a.requirements["ambiguity"] < 4.0


def test_normalizer_detects_code_block():
    normalized = normalize("Fix this:\n```python\nprint(1)\n```")
    assert normalized.has_code_block
    assert "python" in normalized.code_block_langs
