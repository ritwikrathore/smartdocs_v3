from __future__ import annotations

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolFinding(BaseModel):
    """Normalized finding produced by any review tool.
    - rule_description ties back to the user-facing rule text
    - snippet is a short context or the violation text depending on tool kind
    - details can carry tool-specific fields like matched text, rag scores, etc.

    Note: page_num should be >= 1, but constraint is not enforced in the schema
    to maintain compatibility with Databricks models that don't support 'minimum'.
    """
    id: str = Field(..., description="Unique identifier for this finding")
    page_num: int = Field(..., description="Page number (should be >= 1)")
    rule_description: str
    kind: Literal["regex", "semantic", "calc"]
    snippet: str
    details: Dict[str, Any] = Field(default_factory=dict)
    score_raw: Optional[float] = None


class RankedFinding(BaseModel):
    """Finding after evaluation/ranking suitable for presentation.
    Includes UI-oriented fields for card rendering.

    Note: confidence should be between 0.0 and 1.0, but constraints are not
    enforced in the schema to maintain compatibility with Databricks models
    that don't support 'minimum'/'maximum' in JSON schemas.
    """
    id: str
    page_num: int
    rule_description: str
    violation_type: Literal["regex", "semantic", "calc"]
    finding: str
    analysis: str
    context: str
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    severity: Literal["low", "medium", "high"] = "medium"

