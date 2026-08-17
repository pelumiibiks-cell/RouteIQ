"""Prompt normalization: the first pipeline stage.

Cleans and structures the raw prompt/context/attachments into a single
NormalizedPrompt the rest of the pipeline can reason about consistently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class NormalizedPrompt:
    raw_text: str
    text: str
    context: str
    attachments: list[str] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    has_code_block: bool = False
    code_block_langs: list[str] = field(default_factory=list)
    line_count_in_code: int = 0
    has_data_block: bool = False  # csv/json/table-like content pasted in
    numbered_steps: int = 0
    question_marks: int = 0


_CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_NUMBERED_STEP_RE = re.compile(r"(?m)^\s*(\d{1,3}[.)]|- )\s+\S")
_DATA_BLOCK_RE = re.compile(r"[{\[].{20,}[}\]]", re.DOTALL)


def normalize(prompt: str, context: str = "", attachments: list[str] | None = None) -> NormalizedPrompt:
    attachments = attachments or []
    raw = prompt or ""
    text = raw.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    combined = f"{text}\n{context}".strip()

    code_blocks = _CODE_FENCE_RE.findall(combined)
    has_code = bool(code_blocks)
    langs = [lang or "unknown" for lang, _ in code_blocks]
    code_lines = sum(body.count("\n") + 1 for _, body in code_blocks)

    has_data = bool(_DATA_BLOCK_RE.search(combined)) and not has_code

    numbered = len(_NUMBERED_STEP_RE.findall(combined))

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]

    return NormalizedPrompt(
        raw_text=raw,
        text=text,
        context=context or "",
        attachments=attachments,
        word_count=len(text.split()),
        char_count=len(combined),
        sentence_count=max(1, len(sentences)),
        has_code_block=has_code,
        code_block_langs=langs,
        line_count_in_code=code_lines,
        has_data_block=has_data,
        numbered_steps=numbered,
        question_marks=text.count("?"),
    )
