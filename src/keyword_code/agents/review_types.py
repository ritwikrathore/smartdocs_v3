from __future__ import annotations

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolFinding(BaseModel):
    """Normalized finding produced by any review tool.
    - rule_description ties back to the user-facing rule text
    - snippet is a short context or the violation text depending on tool kind
    - details can carry tool-specific fields like matched text, rag scores, etc.
    """
    id: str = Field(..., description="Unique identifier for this finding")
    page_num: int = Field(..., ge=1)
    rule_description: str
    kind: Literal["regex", "semantic", "calc", "rag"]
    snippet: str
    details: Dict[str, Any] = Field(default_factory=dict)
    score_raw: Optional[float] = None


class RankedFinding(BaseModel):
    """Finding after evaluation/ranking suitable for presentation."""
    id: str
    page_num: int
    rule_description: str
    finding: str
    context: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high"] = "medium"

