from __future__ import annotations

import asyncio
from typing import List

from .review_types import ToolFinding, RankedFinding
from .review_scoring import pre_score
from .review_evaluator import evaluate_findings
from . import review_tools as tools


async def _run_for_rule_chunk(rule, chunk) -> List[ToolFinding]:
    """Run independent tools in parallel; sequence only when necessary."""
    tasks = []
    # Run regex only for regex rules
    if getattr(rule, "validation_type", None) == "regex":
        tasks.append(asyncio.create_task(tools.run_regex(rule, chunk)))
    results: List[List[ToolFinding]] = await asyncio.gather(*tasks)
    return [f for sub in results for f in sub]


async def orchestrate_review(template, doc_chunks) -> List[RankedFinding]:
    """Plan → Execute → Evaluate → Filter/Present (minimal viable orchestration).

    - Executes regex per (rule, chunk) with concurrency.
    - Executes semantic per rule (batch mode across all chunks).
    - Applies a simple prior score per tool kind.
    - Calls the evaluator to rank/filter findings and returns RankedFinding list.
    """
    per_tasks = []
    for rule in template.rules:
        validation_type = getattr(rule, "validation_type", None)

        if validation_type == "regex":
            # Regex validation: run per chunk
            for chunk in doc_chunks:
                per_tasks.append(asyncio.create_task(_run_for_rule_chunk(rule, chunk)))
        elif validation_type == "semantic":
            # Semantic validation: run batch mode (single call per rule across all chunks)
            per_tasks.append(asyncio.create_task(tools.run_semantic_batch(rule, doc_chunks)))
        else:
            # Unknown validation type: log warning and skip
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown validation_type '{validation_type}' for rule '{getattr(rule, 'description', 'unknown')}'. "
                f"Skipping this rule."
            )

    findings_nested: List[List[ToolFinding]] = await asyncio.gather(*per_tasks)
    findings: List[ToolFinding] = [f for sub in findings_nested for f in sub]

    findings = pre_score(findings)
    ranked: List[RankedFinding] = await evaluate_findings(findings)

    # Optional: thresholding here to reduce noise
    threshold = 0.6
    ranked = [r for r in ranked if r.confidence >= threshold]

    # Optional: cap per rule/page to avoid overload
    MAX_PER_PAGE_PER_RULE = 5
    key = lambda r: (r.rule_description, r.page_num)
    buckets = {}
    for r in ranked:
        k = key(r)
        buckets.setdefault(k, []).append(r)
    trimmed: List[RankedFinding] = []
    for _, items in buckets.items():
        trimmed.extend(sorted(items, key=lambda x: x.confidence, reverse=True)[:MAX_PER_PAGE_PER_RULE])

    return trimmed

