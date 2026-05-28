from __future__ import annotations

import re


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*$", flags=re.IGNORECASE | re.DOTALL)
_STRAY_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", flags=re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """Remove hidden chain-of-thought wrappers emitted by reasoning models."""

    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    cleaned = _STRAY_THINK_TAG_RE.sub("", cleaned)
    return cleaned.strip()
