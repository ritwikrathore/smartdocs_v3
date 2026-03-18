"""
Response parsing utilities for RLM.

Extracts code blocks from LLM responses and detects FINAL answer calls.
"""

import re
from typing import List, Optional

from ..config import RLM_STDOUT_TRUNCATE

# Matches ```repl, ```python, or bare ``` code blocks
_CODE_BLOCK_RE = re.compile(
    r"```(?:repl|python)?\s*\n(.*?)```",
    re.DOTALL,
)


def find_code_blocks(response: str) -> List[str]:
    """Extract fenced code blocks (```repl or ```python) from an LLM response."""
    return _CODE_BLOCK_RE.findall(response)


def find_final_answer(response: str) -> Optional[str]:
    """
    Detect if the LLM called FINAL(...) directly in its prose (outside a code block).

    Returns the answer string if found, None otherwise.
    This is a fallback — normally FINAL() is called inside executed code.
    """
    # Remove code blocks first so we only look at prose
    prose = _CODE_BLOCK_RE.sub("", response)

    # Match FINAL("...") or FINAL('...')
    match = re.search(r'FINAL\(\s*["\'](.+?)["\']\s*\)', prose, re.DOTALL)
    if match:
        return match.group(1)

    # Match FINAL_VAR("...")
    match = re.search(r'FINAL_VAR\(\s*["\'](\w+)["\']\s*\)', prose)
    if match:
        return match.group(1)  # Return var name; caller resolves

    return None


def truncate_output(text: str, max_chars: int = RLM_STDOUT_TRUNCATE) -> str:
    """Truncate text to max_chars, appending a notice if truncated."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated — {len(text)} total chars]"
