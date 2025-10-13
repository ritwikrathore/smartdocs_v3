from __future__ import annotations

import re
import uuid
from typing import List

# We deliberately import SmartReview lazily to reuse its LLM helper without
# creating a hard module-level circular import with orchestrator wiring.
import SmartReview as SR  # type: ignore

from .review_types import ToolFinding


def _context_snippet(text: str, start: int, end: int, pad: int = 50) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    return f"...{text[s:e]}..."


async def run_regex(rule, chunk) -> List[ToolFinding]:
    findings: List[ToolFinding] = []
    try:
        for m in re.finditer(rule.validator, chunk.content):
            findings.append(
                ToolFinding(
                    id=str(uuid.uuid4()),
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    kind="regex",
                    snippet=_context_snippet(chunk.content, m.start(), m.end()),
                    details={"matched": m.group(0)},
                )
            )
    except re.error as e:
        # Surface via Streamlit if desired: SR.st.warning(...)
        pass
    return findings


async def run_semantic(rule, chunk) -> List[ToolFinding]:
    system_prompt = (
        "You are an AI document validation assistant. You will be given a chunk of text and a rule.\n"
        "Your task is to check if the text violates the rule.\n"
        "- If you find a violation, respond with \"Violation: [Explain the violation and quote the specific text]\".\n"
        "- If there are no violations, respond only with \"No violation found.\".\n"
        "Do not be conversational. Provide only the violation report or \"No violation found.\"."
    )
    prompt = f"""
        --- RULE ---
        {rule.validator}

        --- TEXT TO VALIDATE ---
        {chunk.content}
    """
    try:
        response = await SR._chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=getattr(SR.client, "_default_model", "databricks-llama-4-maverick"),
            temperature=0.1,
        )
        message_content = response.choices[0].message.content
        if message_content and message_content.lower().strip() != "no violation found.":
            return [
                ToolFinding(
                    id=str(uuid.uuid4()),
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    kind="semantic",
                    snippet=message_content,
                    details={"context": f"Semantic check on page {chunk.page_num}."},
                )
            ]
    except Exception:
        pass
    return []

