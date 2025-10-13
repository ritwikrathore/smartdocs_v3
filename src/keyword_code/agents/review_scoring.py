from __future__ import annotations

from typing import List
from .review_types import ToolFinding


def pre_score(findings: List[ToolFinding]) -> List[ToolFinding]:
    """Attach simple deterministic scores by tool kind as a prior.
    This is a light heuristic and should be tuned or replaced per deployment.
    """
    base = {
        "regex": 0.8,      # precise patterns score higher by default
        "semantic": 0.6,   # LLM judgments get moderated until corroborated
        "calc": 0.7,
        "rag": 0.5,
    }
    for f in findings:
        if f.score_raw is None:
            f.score_raw = base.get(f.kind, 0.5)
    return findings

