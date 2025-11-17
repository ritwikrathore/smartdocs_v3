"""Lightweight text helpers used by both chunking and verification modules."""

from typing import Optional
import re


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, strip, and squash whitespace for consistent comparisons."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", str(text).lower())
    return normalized.strip()
