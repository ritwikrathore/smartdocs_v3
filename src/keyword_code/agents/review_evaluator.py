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
    import src.keyword_code.smartreview.smartreview as SR  # type: ignore
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
                "You are a VIOLATION FILTER for document review findings. Your role is to EXCLUDE false positives and return ONLY true violations.\n\n"

                "CRITICAL FILTERING RULES:\n"
                "1. EXCLUDE findings where the matched text is COMPLIANT with the rule (no violation exists)\n"
                "2. EXCLUDE findings where there is NO CONFUSION or error (e.g., 'The matched text does not contain any word confusion errors')\n"
                "3. EXCLUDE findings where the text MEETS the requirements (e.g., 'U.S. dollars' correctly capitalizes country name)\n"
                "4. EXCLUDE findings where the match is NOT RELEVANT to the rule (e.g., 'U.S. GAAP' flagged by a capitalization rule when it's a proper acronym)\n"
                "5. EXCLUDE partial/embedded matches (e.g., '5' within '5.5 billion' when rule targets integers like '5 billion')\n"
                "6. EXCLUDE findings with hedged language indicating uncertainty (e.g., 'may be', 'could be', 'possibly')\n\n"

                "WHAT TO INCLUDE (TRUE VIOLATIONS ONLY):\n"
                "- Text that ACTUALLY VIOLATES the rule (e.g., '5 billion' when rule requires decimal precision like '5.0 billion')\n"
                "- Text with ACTUAL ERRORS (e.g., 'deceased' instead of 'decreased', 'their' instead of 'there')\n"
                "- Text that FAILS to meet the rule requirements with clear evidence\n\n"

                "CONFIDENCE SCORING:\n"
                "- If you determine the finding is compliant/correct: DO NOT include it in output (filter it out completely)\n"
                "- If you determine it's a clear violation: assign confidence 0.8-1.0\n"
                "- If you're uncertain but lean toward violation: assign confidence 0.6-0.7\n"
                "- If you're uncertain and lean toward compliance: DO NOT include it (filter it out)\n\n"

                "OUTPUT FORMAT:\n"
                "Your output MUST be a JSON array of RankedFinding objects (TRUE VIOLATIONS ONLY) with fields:\n"
                "id, page_num, rule_description, violation_type, finding, analysis, context, confidence, severity.\n"
                "- violation_type: reflects the underlying tool signal (regex|semantic|calc|rag)\n"
                "- finding: the violation text to display\n"
                "- analysis: a short (<=2 sentences) explanation of WHY it violates the rule (NOT why it's compliant)\n"
                "- context: the actual document text snippet from 'details.context' field in the input ToolFinding\n"
                "  (NEVER use generic text like 'Batch semantic check across pages' or 'Semantic check' as context)\n"
                "- confidence: 0.6-1.0 for violations only (lower confidence violations may be filtered by downstream threshold)\n"
                "- severity: 'low', 'medium', or 'high' based on impact\n\n"

                "REMEMBER: Your job is to FILTER OUT false positives, not to report compliance. "
                "Return an EMPTY ARRAY [] if no true violations are found.\n"
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

            # Log breakdown of findings by rule and kind
            from collections import defaultdict
            breakdown = defaultdict(lambda: {"regex": 0, "semantic": 0})
            for f in findings:
                breakdown[f.rule_description][f.kind] += 1
            logger.info(f"Running AI evaluation on {len(findings)} findings...")
            for rule_desc, counts in breakdown.items():
                logger.info(f"  Rule: '{rule_desc[:80]}...' - regex: {counts['regex']}, semantic: {counts['semantic']}")

            result = await agent.run(
                (
                    "Evaluate these findings and return ONLY the ones that are TRUE VIOLATIONS of their rules.\n"
                    "REJECT any findings where the matched text actually COMPLIES with the rule.\n"
                    "For example, if the rule requires decimal precision for billion values:\n"
                    "- REJECT findings like '5.5 billion' or '1.0 billion' (these are compliant, not violations)\n"
                    "- INCLUDE findings like '5 billion' or '1 billion' (these are violations)\n"
                    "\n"
                    "For semantic rules (case sensitivity, word confusion, etc.), INCLUDE findings that identify violations.\n"
                    "Do NOT reject semantic findings just because they don't match the regex example above.\n"
                    "\n"
                    "Return a JSON array of RankedFinding with the exact fields: "
                    "id, page_num, rule_description, violation_type, finding, analysis, context, confidence, severity.\n"
                    "Be strict about rejecting partial/embedded regex matches unless the rule explicitly allows them.\n"
                    "Return an EMPTY ARRAY [] if no true violations are found.\n"
                    f"INPUT (JSON array of ToolFinding):\n\n{payload}\n"
                ),
            )

            # Log breakdown of results by rule
            result_breakdown = defaultdict(lambda: {"regex": 0, "semantic": 0})
            if result.output:
                for r in result.output:
                    vtype = getattr(r, "violation_type", "unknown")
                    result_breakdown[r.rule_description][vtype] += 1

            logger.info(f"AI evaluation completed. Returned {len(result.output) if result.output else 0} ranked findings.")
            for rule_desc, counts in result_breakdown.items():
                logger.info(f"  Rule: '{rule_desc[:80]}...' - regex: {counts['regex']}, semantic: {counts['semantic']}")

            return result.output
        except Exception as e:
            logger.error(
                f"AI evaluation failed with error: {e}. Falling back to deterministic scoring. "
                f"This may result in lower quality filtering of false positives.",
                exc_info=True
            )
            # Fall back below on any runtime issue
            pass

    # Fallback: trivial mapping with a confidence from the prior score_raw
    ranked: List[RankedFinding] = []
    for f in findings:
        base_conf = float(max(0.0, min(1.0, (f.score_raw or 0.5))))
        conf = base_conf
        # Generic boundary penalty for regex partial/embedded matches (fallback only)
        # Reduced penalty to avoid filtering out valid regex findings
        if f.kind == "regex" and isinstance(f.details, dict):
            left_issue = bool(f.details.get("left_is_alnum") or f.details.get("left_is_dot"))
            right_issue = bool(f.details.get("right_is_alnum") or f.details.get("right_is_dot"))
            if left_issue and right_issue:
                # Both sides have issues - more likely a false positive
                conf = max(0.0, conf - 0.15)
            elif left_issue or right_issue:
                # Only one side has issues - minor penalty
                conf = max(0.0, conf - 0.05)
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
            # Extract context from details, ensuring we use actual document text
            if isinstance(f.details, dict):
                context = f.details.get("context", "")
                # If context is generic placeholder text, try to use snippet as fallback
                if context in ["Semantic check", "Batch semantic check across pages", ""]:
                    context = f.snippet[:200] if len(f.snippet) > 200 else f.snippet
            else:
                context = f.snippet[:200] if len(f.snippet) > 200 else f.snippet
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

