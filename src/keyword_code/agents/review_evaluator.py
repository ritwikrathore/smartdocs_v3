from __future__ import annotations

import json
import os
import logging
from typing import List

from .review_types import ToolFinding, RankedFinding

logger = logging.getLogger(__name__)

# Optional: use Pydantic-AI if available for LLM-based evaluation
try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    _HAS_PYDANTIC_AI = True
except Exception:  # pragma: no cover - graceful fallback if not installed
    Agent = None
    OpenAIChatModel = None
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
        logger.warning("pydantic-ai is not installed; evaluator will use deterministic fallback.")
        return None
    if _agent is not None:
        return _agent

    # Resolve Databricks OpenAI-compatible settings (no OpenAI key usage)
    dbx_api_key = os.getenv("DATABRICKS_API_KEY")
    if not dbx_api_key:
        logger.warning("DATABRICKS_API_KEY not found in environment; evaluator will use fallback.")
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
        # Use OpenAI-compatible provider bound to Databricks endpoint
        # This works because Databricks serving endpoints are OpenAI-compatible
        provider = OpenAIProvider(base_url=dbx_base_url, api_key=dbx_api_key)
        model = OpenAIChatModel(model_name, provider=provider)

        _agent = Agent(
            model=model,
            output_type=list[RankedFinding],
            system_prompt=(
                "You evaluate document review findings to identify TRUE VIOLATIONS ONLY.\n"
                "- CRITICAL: Only return findings that are ACTUAL VIOLATIONS of the rule. Do NOT return compliant/correct values.\n"
                "- If a finding shows text that COMPLIES with the rule (e.g., '5.5 billion' when rule requires decimal precision), REJECT it entirely (do not include in output).\n"
                "- If a finding shows text that VIOLATES the rule (e.g., '5 billion' when rule requires decimal precision), include it with high confidence.\n"
                "- Use rule_description plus tool details to decide if the matched text is a violation or compliant.\n"
                "- Penalize hedged language and partial/embedded matches.\n"
                "- If a regex match is a substring of a larger token (e.g., '5' within '5.5'), reject it unless the rule explicitly allows it.\n"
                "- Consider numeric and word boundaries, and whether the match changes meaning in context.\n"
                "Your output MUST be a JSON array of RankedFinding objects (VIOLATIONS ONLY) with fields:\n"
                "id, page_num, rule_description, violation_type, finding, analysis, context, confidence, severity.\n"
                "- violation_type should reflect the underlying tool signal (regex|semantic|calc|rag).\n"
                "- finding is the violation text to display; analysis is a short (<=2 sentences) explanation of WHY it violates the rule.\n"
                "- Return an EMPTY ARRAY [] if no true violations are found.\n"
            ),
        )
        return _agent
    except Exception as e:
        logger.exception("Failed to initialize evaluator agent with Databricks provider: %s", e)
        # If anything goes wrong, do not raise; let the caller use fallback logic
        return None


async def evaluate_findings(findings: List[ToolFinding]) -> List[RankedFinding]:
    """Evaluate and rank tool findings. Uses Databricks (OpenAI-compatible) via
    pydantic-ai if configured; otherwise falls back to deterministic scoring.
    """
    agent = _get_agent()
    if agent is None:
        logger.warning("Evaluator agent unavailable; using deterministic fallback. Set DATABRICKS_API_KEY/DATABRICKS_BASE_URL/DATABRICKS_LLM_MODEL to enable AI evaluation.")
    if agent is not None:
        try:
            # Embed findings as JSON in the prompt for now. For larger batches, consider
            # chunking or passing via deps and custom tools.
            payload = json.dumps([f.model_dump() for f in findings])
            result = await agent.run(
                (
                    "Evaluate these findings and return ONLY the ones that are TRUE VIOLATIONS of their rules.\n"
                    "REJECT any findings where the matched text actually COMPLIES with the rule.\n"
                    "For example, if the rule requires decimal precision for billion values:\n"
                    "- REJECT findings like '5.5 billion' or '1.0 billion' (these are compliant, not violations)\n"
                    "- INCLUDE findings like '5 billion' or '1 billion' (these are violations)\n"
                    "Return a JSON array of RankedFinding with the exact fields: "
                    "id, page_num, rule_description, violation_type, finding, analysis, context, confidence, severity.\n"
                    "Be strict about rejecting partial/embedded regex matches unless the rule explicitly allows them.\n"
                    "Return an EMPTY ARRAY [] if no true violations are found.\n"
                    f"INPUT (JSON array of ToolFinding):\n\n{payload}\n"
                ),
            )
            return result.output
        except Exception as e:
            logger.warning("AI evaluation failed; using deterministic fallback: %s", e)
            # Fall back below on any runtime issue
            pass

    # Fallback: trivial mapping with a confidence from the prior score_raw
    ranked: List[RankedFinding] = []
    for f in findings:
        base_conf = float(max(0.0, min(1.0, (f.score_raw or 0.5))))
        conf = base_conf
        # Generic boundary penalty for regex partial/embedded matches (fallback only)
        if f.kind == "regex" and isinstance(f.details, dict):
            left_issue = bool(f.details.get("left_is_alnum") or f.details.get("left_is_dot"))
            right_issue = bool(f.details.get("right_is_alnum") or f.details.get("right_is_dot"))
            if left_issue or right_issue:
                conf = max(0.0, conf - 0.3)
        # Heuristic rendering by kind (fallback only). Tailor to rule and match when possible.
        if f.kind == "regex":
            matched = (f.details.get("matched", "") if isinstance(f.details, dict) else "")
            finding_text = matched or f.snippet[:120]
            analysis = (
                f"Matched '{matched}' which may violate the rule: {f.rule_description}."
                if matched else f"Potential violation for rule: {f.rule_description}."
            )
            context = f.snippet
        elif f.kind == "semantic":
            finding_text = f.snippet
            analysis = f"Potential violation for rule: {f.rule_description}."
            context = f.details.get("context", "Semantic check") if isinstance(f.details, dict) else "Semantic check"
        else:
            finding_text = f.snippet
            analysis = f"Potential violation flagged by {f.kind} for rule: {f.rule_description}."
            context = f.details.get("context", "") if isinstance(f.details, dict) else ""
        ranked.append(
            RankedFinding(
                id=f.id,
                page_num=f.page_num,
                rule_description=f.rule_description,
                violation_type=f.kind,
                finding=finding_text,
                analysis=analysis,
                context=context,
                confidence=conf,
                severity="medium" if conf < 0.85 else "high",
            )
        )
    return ranked

