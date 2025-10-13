from __future__ import annotations

import json
import os
from typing import List

from .review_types import ToolFinding, RankedFinding

# Optional: use Pydantic-AI if available for LLM-based evaluation
try:
    from pydantic_ai import Agent, models
    from pydantic_ai.providers.openai import OpenAIProvider  # OpenAI-compatible provider
    _HAS_PYDANTIC_AI = True
except Exception:  # pragma: no cover - graceful fallback if not installed
    Agent = None
    models = None
    OpenAIProvider = None
    _HAS_PYDANTIC_AI = False

# Import SmartReview to reuse Databricks config constants if available
try:
    import SmartReview as SR  # type: ignore
except Exception:  # pragma: no cover
    SR = None  # allow import without failing


# Create the agent lazily to avoid import-time side effects in environments
# where models/keys may not be configured yet.
_agent = None  # type: ignore

def _get_agent():  # -> Agent | None (kept untyped for broader compatibility)
    global _agent
    if not _HAS_PYDANTIC_AI:
        return None
    if _agent is not None:
        return _agent

    # Resolve Databricks OpenAI-compatible settings (no OpenAI key usage)
    dbx_api_key = os.getenv("DATABRICKS_API_KEY")
    if not dbx_api_key:
        # No key: do not construct an agent; fallback path will be used
        return None

    dbx_base_url = os.getenv("DATABRICKS_BASE_URL")
    if not dbx_base_url and SR is not None:
        dbx_base_url = getattr(SR, "DATABRICKS_BASE_URL", None)

    model_name = os.getenv("DATABRICKS_LLM_MODEL")
    if not model_name and SR is not None:
        model_name = getattr(SR, "DATABRICKS_LLM_MODEL", None)
    if not model_name:
        model_name = "databricks-llama-4-maverick"

    try:
        provider = OpenAIProvider(base_url=dbx_base_url, api_key=dbx_api_key)
        # Build a model explicitly bound to the provider so pydantic-ai won't look for OPENAI_* envs
        model = models.OpenAIChatModel(model_name, provider=provider)
        _agent = Agent(
            model=model,
            output_type=list[RankedFinding],
            system_prompt=(
                "You evaluate document review findings.\n"
                "Return only high-confidence issues. Prefer concise, quoted evidence.\n"
                "Penalize hedged language. Boost corroborated (regex+semantic) findings.\n"
            ),
        )
        return _agent
    except Exception:
        # If anything goes wrong, do not raise; let the caller use fallback logic
        return None


async def evaluate_findings(findings: List[ToolFinding]) -> List[RankedFinding]:
    """Evaluate and rank tool findings. Uses Databricks (OpenAI-compatible) via
    pydantic-ai if configured; otherwise falls back to deterministic scoring.
    """
    agent = _get_agent()
    if agent is not None:
        try:
            # Embed findings as JSON in the prompt for now. For larger batches, consider
            # chunking or passing via deps and custom tools.
            payload = json.dumps([f.model_dump() for f in findings])
            result = await agent.run(
                f"Rank and filter these findings (JSON array follows):\n\n{payload}\n",
            )
            return result.output
        except Exception:
            # Fall back below on any runtime issue
            pass

    # Fallback: trivial mapping with a confidence from the prior score_raw
    ranked: List[RankedFinding] = []
    for f in findings:
        conf = float(max(0.0, min(1.0, (f.score_raw or 0.5))))
        # Heuristic context/finding rendering by kind
        if f.kind == "regex":
            finding_text = f"Found violation: '{f.details.get('matched', '').strip() or f.snippet[:120]}'"
            context = f.snippet
        elif f.kind == "semantic":
            finding_text = f.snippet
            context = f.details.get("context", "Semantic check")
        else:
            finding_text = f.snippet
            context = f.details.get("context", "")
        ranked.append(
            RankedFinding(
                id=f.id,
                page_num=f.page_num,
                rule_description=f.rule_description,
                finding=finding_text,
                context=context,
                confidence=conf,
                severity="medium" if conf < 0.85 else "high",
            )
        )
    return ranked

