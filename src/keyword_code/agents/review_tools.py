from __future__ import annotations

import re
import uuid
import json
from typing import List, Dict

# We deliberately import SmartReview lazily to reuse its LLM helper without
# creating a hard module-level circular import with orchestrator wiring.
import src.keyword_code.smartreview.smartreview as SR  # type: ignore

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
    # Extract examples from rule if available
    extracted_examples = getattr(rule, "extracted_examples", [])
    examples_warning = ""
    if extracted_examples:
        examples_list = ", ".join([f"'{ex}'" for ex in extracted_examples])
        examples_warning = (
            f"\n\nIMPORTANT - EXAMPLES TO IGNORE:\n"
            f"The following text strings are EXAMPLES ONLY from the rule definition. "
            f"Do NOT flag these as violations unless they actually appear in the document being validated:\n"
            f"{examples_list}\n"
            f"These are for illustration purposes only. Only flag actual violations found in the document text below."
        )

    system_prompt = (
        "You are an AI document validation assistant. You will be given a chunk of text and a rule.\n"
        "Your task is to check if the text violates the rule.\n\n"

        "CRITICAL - ONLY FLAG TRUE VIOLATIONS:\n"
        "- Do NOT flag text that is COMPLIANT with the rule\n"
        "- Do NOT flag text where there is NO CONFUSION or error\n"
        "- Do NOT flag text that MEETS the requirements\n"
        "- Do NOT flag text that is NOT RELEVANT to the rule (e.g., proper acronyms like 'U.S. GAAP' when checking capitalization)\n"
        "- Do NOT flag numeric expressions that already satisfy the rule (e.g., '67.3 billion' is compliant if rule requires decimal precision)\n\n"

        "RESPONSE FORMAT:\n"
        "- If you find a TRUE VIOLATION, respond with ONLY the exact string of text from the document that violates the rule. "
        "Do NOT include explanations, commentary, or corrections. Extract and return ONLY the verbatim erroneous text.\n"
        "- If there are NO VIOLATIONS (including compliant cases, correct usage, or irrelevant matches), respond only with \"No violation found.\".\n\n"

        "Do not be conversational. Provide only the exact erroneous text or \"No violation found.\"."
        f"{examples_warning}"
    )

    # Use clarified rule if available, otherwise use validator
    rule_text = getattr(rule, 'clarified_rule', None) or getattr(rule, 'validator', '')

    prompt = f"""
        --- RULE ---
        {rule_text}

        --- TEXT TO VALIDATE ---
        {chunk.content}
    """
    try:
        response = await SR._chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=getattr(SR, "DATABRICKS_LLM_MODEL", "databricks-llama-4-maverick"),
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
    except Exception as e:
        # Log the error but don't fail the entire validation
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"Semantic validation failed for rule '{getattr(rule, 'description', 'unknown')}' "
            f"on page {getattr(chunk, 'page_num', '?')}: {e}",
            exc_info=True
        )
    return []


async def run_semantic_batch(rule, doc_chunks: List) -> List[ToolFinding]:
    """Batch semantic review per rule across the whole document.
    - Splits pages into simple ~30k-token batches (naive word-count estimator).
    - For each batch, asks the LLM to return ONLY a JSON array of {page_num, finding}.
    - Aggregates into ToolFinding objects with kind="semantic".
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch semantic validation for rule: '{getattr(rule, 'description', 'unknown')[:80]}...' across {len(doc_chunks)} chunks")

    # Create a mapping of page_num to chunk content for context extraction
    page_content_map = {getattr(ch, "page_num", -1): getattr(ch, "content", "") for ch in doc_chunks}

    TARGET_TOKENS = 15000  # Optimized for better processing efficiency
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
    # Extract examples from rule if available
    extracted_examples = getattr(rule, "extracted_examples", [])
    examples_warning = ""
    if extracted_examples:
        examples_list = ", ".join([f"'{ex}'" for ex in extracted_examples])
        examples_warning = (
            f"\n\nIMPORTANT - EXAMPLES TO IGNORE:\n"
            f"The following text strings are EXAMPLES ONLY from the rule definition. "
            f"Do NOT flag these as violations unless they actually appear in the document being validated:\n"
            f"{examples_list}\n"
            f"These are for illustration purposes only. Only flag actual violations found in the document pages below."
        )

    system_prompt = (
        "You are an AI document validation assistant. You will be given a rule and multiple pages of text,"
        " each labeled as 'Page N:'.\n\n"

        "CRITICAL - ONLY FLAG TRUE VIOLATIONS:\n"
        "- Do NOT flag text that is COMPLIANT with the rule\n"
        "- Do NOT flag text where there is NO CONFUSION or error\n"
        "- Do NOT flag text that MEETS the requirements\n"
        "- Do NOT flag text that is NOT RELEVANT to the rule (e.g., proper acronyms like 'U.S. GAAP' when checking capitalization)\n"
        "- Do NOT flag numeric expressions that already satisfy the rule (e.g., '67.3 billion' is compliant if rule requires decimal precision)\n\n"

        "TASK:\n"
        "Identify ALL TRUE VIOLATIONS of the rule anywhere in the provided pages.\n"
        "Return ONLY a JSON array. Each item must be an object with exactly these keys: \n"
        "- page_num (integer)\n- finding (string).\n\n"

        "CRITICAL: The 'finding' field MUST contain ONLY the exact string of text from the document that violates the rule. "
        "Do NOT include explanations, commentary, or corrections in the 'finding' field. "
        "Extract and return ONLY the verbatim erroneous text as it appears in the document.\n"
        "Do NOT return: 'The sentence is \"...\" The correct word is \"decreased\"...'\n\n"

        "No prose outside JSON. Return empty array [] if no violations found."
        f"{examples_warning}"
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

        # Use clarified rule if available, otherwise use validator
        rule_text = getattr(rule, 'clarified_rule', None) or getattr(rule, 'validator', '')

        user_prompt = f"""
        --- RULE ---
        {rule_text}

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
                model=getattr(SR, "DATABRICKS_LLM_MODEL", "databricks-llama-4-maverick"),
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
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Batch semantic validation for rule '{getattr(rule, 'description', 'unknown')}' "
                    f"returned non-list response: {type(items).__name__}. Skipping batch."
                )
                continue

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Batch semantic validation found {len(items)} items for rule '{getattr(rule, 'description', 'unknown')[:80]}...'")

            for it in items:
                try:
                    page_num = int(it.get("page_num"))
                    finding_text = str(it.get("finding", "")).strip()
                    if not finding_text or page_num not in page_nums_in_batch:
                        continue

                    # Extract actual context from the document chunk
                    context_snippet = "Semantic check"
                    if page_num in page_content_map:
                        page_content = page_content_map[page_num]
                        # Try to find the finding text in the page content
                        finding_pos = page_content.find(finding_text)
                        if finding_pos != -1:
                            # Extract context around the finding (50 chars before and after)
                            start = max(0, finding_pos - 50)
                            end = min(len(page_content), finding_pos + len(finding_text) + 50)
                            context_snippet = f"...{page_content[start:end]}..."
                        else:
                            # If exact match not found, try case-insensitive search
                            finding_lower = finding_text.lower()
                            content_lower = page_content.lower()
                            finding_pos = content_lower.find(finding_lower)
                            if finding_pos != -1:
                                start = max(0, finding_pos - 50)
                                end = min(len(page_content), finding_pos + len(finding_text) + 50)
                                context_snippet = f"...{page_content[start:end]}..."
                            else:
                                # Fallback: use first 150 chars of page as context
                                context_snippet = f"{page_content[:150]}..." if len(page_content) > 150 else page_content

                    findings.append(
                        ToolFinding(
                            id=str(uuid.uuid4()),
                            page_num=page_num,
                            rule_description=getattr(rule, "description", ""),
                            kind="semantic",
                            snippet=finding_text,
                            details={"context": context_snippet},
                        )
                    )
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to parse finding item in batch semantic validation: {e}. "
                        f"Item: {it if 'it' in locals() else 'N/A'}"
                    )
                    continue
        except Exception as e:
            # Log the error and skip this batch, but continue with others
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Batch semantic validation failed for rule '{getattr(rule, 'description', 'unknown')}': {e}",
                exc_info=True
            )
            continue

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Batch semantic validation completed for rule '{getattr(rule, 'description', 'unknown')[:80]}...'. Total findings: {len(findings)}")
    return findings

