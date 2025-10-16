from __future__ import annotations

import re
import uuid
import json
from typing import List, Dict

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
            start, end = m.start(), m.end()
            matched = m.group(0)
            pre_char = chunk.content[start-1] if start > 0 else ""
            post_char = chunk.content[end] if end < len(chunk.content) else ""
            details = {
                "matched": matched,
                "pattern": getattr(rule, "validator", ""),
                "start": start,
                "end": end,
                "pre_char": pre_char,
                "post_char": post_char,
                "left_is_alnum": pre_char.isalnum() if pre_char else False,
                "right_is_alnum": post_char.isalnum() if post_char else False,
                "left_is_digit": pre_char.isdigit() if pre_char else False,
                "right_is_digit": post_char.isdigit() if post_char else False,
                "left_is_dot": pre_char == "." if pre_char else False,
                "right_is_dot": post_char == "." if post_char else False,
            }
            findings.append(
                ToolFinding(
                    id=str(uuid.uuid4()),
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    kind="regex",
                    snippet=_context_snippet(chunk.content, start, end),
                    details=details,
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
        "- Do NOT flag numeric expressions that already satisfy the rule; for example, if decimal precision like \"1.0 billion\" is required, values like \"67.3 billion\" are compliant and must not be flagged; only integer forms like \"67 billion\" should be flagged.\n"
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


async def run_semantic_batch(rule, doc_chunks: List) -> List[ToolFinding]:
    """Batch semantic review per rule across the whole document.
    - Splits pages into simple ~30k-token batches (naive word-count estimator).
    - For each batch, asks the LLM to return ONLY a JSON array of {page_num, finding}.
    - Aggregates into ToolFinding objects with kind="semantic".
    """
    TARGET_TOKENS = 30000  # fixed; simple starting point as requested
    OVERLAP_TOKENS = 300  # fixed overlap between batches (words)

    def _tail_words(s: str, n: int) -> str:
        try:
            words = s.split()
            if len(words) <= n:
                return s
            return " ".join(words[-n:])
        except Exception:
            # crude fallback using characters
            return s[-n*4:] if s else s


    def _est_tokens(s: str) -> int:
        # naive proxy: tokens ~= words
        try:
            return max(1, len(s.split()))
        except Exception:
            return len(s) // 4 + 1

    # 1) Create batches of pages by naive token budget
    batches: List[List] = []
    cur: List = []
    cur_tokens = 0
    for ch in doc_chunks:
        t = _est_tokens(getattr(ch, "content", ""))
        if cur and (cur_tokens + t > TARGET_TOKENS):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(ch)
        cur_tokens += t
    if cur:
        batches.append(cur)

    findings: List[ToolFinding] = []

    # 2) For each batch, ask the LLM to enumerate violations with page numbers
    system_prompt = (
        "You are an AI document validation assistant. You will be given a rule and multiple pages of text,"
        " each labeled as 'Page N:'.\n"
        "Identify ALL violations of the rule anywhere in the provided pages.\n"
        "Return ONLY a JSON array. Each item must be an object with exactly these keys: \n"
        "- page_num (integer)\n- finding (string).\n"
        "Include a short explanation and the quoted offending text in 'finding'. No prose outside JSON.\n"
        "Do NOT flag numeric expressions that already satisfy the rule; for example, if decimal precision like \"1.0 billion\" is required, values like \"67.3 billion\" are compliant and must not be flagged; only integer forms like \"67 billion\" should be flagged."
    )

    prev_last_page = None
    for batch in batches:
        pages_text = []
        page_nums_in_batch = set()
        if prev_last_page is not None:
            page_nums_in_batch.add(getattr(prev_last_page, "page_num", -1))
            pages_text.append(
                f"Page {getattr(prev_last_page, 'page_num', '?')} (overlap excerpt):\n"
                f"{_tail_words(getattr(prev_last_page, 'content', ''), OVERLAP_TOKENS)}"
            )
        for ch in batch:
            page_nums_in_batch.add(getattr(ch, "page_num", -1))
            pages_text.append(f"Page {getattr(ch, 'page_num', '?')}:\n{getattr(ch, 'content', '')}")
        body = "\n\n".join(pages_text)
        prev_last_page = batch[-1] if batch else prev_last_page

        user_prompt = f"""
        --- RULE ---
        {getattr(rule, 'validator', '')}

        --- PAGES ---
        {body}
        --- END PAGES ---
        """

        try:
            resp = await SR._chat_completion_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=getattr(SR.client, "_default_model", "databricks-llama-4-maverick"),
                temperature=0.1,
            )
            raw = resp.choices[0].message.content
            # Parse strictly as JSON (array expected)
            parsed = SR._parse_model_json(raw)
            if isinstance(parsed, dict):
                items = parsed.get("items") or parsed.get("violations") or []
            else:
                items = parsed
            if not isinstance(items, list):
                continue
            for it in items:
                try:
                    page_num = int(it.get("page_num"))
                    finding_text = str(it.get("finding", "")).strip()
                    if not finding_text or page_num not in page_nums_in_batch:
                        continue
                    findings.append(
                        ToolFinding(
                            id=str(uuid.uuid4()),
                            page_num=page_num,
                            rule_description=getattr(rule, "description", ""),
                            kind="semantic",
                            snippet=finding_text,
                            details={"context": "Batch semantic check across pages"},
                        )
                    )
                except Exception:
                    continue
        except Exception:
            # Skip batch on any LLM error; continue with others
            continue

    return findings

