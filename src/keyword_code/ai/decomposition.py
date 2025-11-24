"""
Prompt decomposition functionality.
"""

import json
import re
from typing import Any, Dict, List, Optional, Literal, Set

from pydantic import BaseModel, Field

from ..config import logger, DECOMPOSITION_MODEL_NAME, USE_DATABRICKS_LLM
from ..utils.langfuse_tracing import (
    optional_context,
    record_generation_error,
    set_generation_output,
    start_generation,
)


async def decompose_ask_mode_prompt(
    analyzer,
    user_prompt: str,
    *,
    first_page_preview: Optional[str] = None,
    document_index_preview: Optional[str] = None,
) -> Dict[str, Any]:
    """Decompose the prompt and decide whether keyword mode should run."""

    logger.info("[ASK MODE] Decomposing prompt with RAG optimization: '%s'", user_prompt[:100])

    system_prompt = """You are a query decomposition assistant for a document analysis RAG system. You receive a user query along with brief previews of the document's first page and table-of-contents index. Your job is to break down the query into focused sub-questions and provide retrieval guidance for each.

IMPORTANT: You are NOT answering the questions. You are preparing queries and hints for a retrieval system that will find relevant document chunks. Another LLM will analyze those chunks to answer the user's questions.

Return a single JSON object with the keys:
- "user_request_context": string capturing explicit output format instructions from the user (e.g., "return as bullet points", "provide counts"). Empty string if none.
- "decomposition": list of sub-prompt objects (see below)

Each sub-prompt object must include:
1. "title": concise descriptor (max 5-6 words)
2. "sub_prompt": the exact question/analysis task text
3. "rag_params": object with retrieval guidance

## Retrieval Guidance (rag_params)
- Set "retrieval_mode" to one of ["hybrid", "semantic", "bm25_dominant"]
- Provide numeric "bm25_weight" and "semantic_weight" that sum to 1.0
- Emit "bm25_terms" as an array of literal phrases for lexical matching (e.g., ["investment number", "investment #"])
- Explain choices in "reasoning"
- Use "hybrid" (0.5/0.5) for balanced questions, "semantic" (0.3/0.7) for interpretive analysis, "bm25_dominant" (0.65/0.35) for precise term lookups

Do not output any text outside the JSON object.

Example Input Prompt + Context:
Document First Page Preview: "...Loan Agreement dated..."
Document Index Preview: "Article I - Definitions (p.5) ..."
User Prompt: "What is the defined / lawful loan currency? What is the duration of the availability period?"

Example JSON Output:
{
    "user_request_context": "",
    "decomposition": [
        {
            "title": "Lawful Loan Currency",
            "sub_prompt": "What is the defined / lawful loan currency?",
            "rag_params": {
                "retrieval_mode": "semantic",
                "bm25_weight": 0.4,
                "semantic_weight": 0.6,
                "bm25_terms": ["lawful loan currency", "legal loan currency", "currency of the loan"],
                "reasoning": "Definition clauses use varied terminology; semantic search helps find conceptually related terms"
            }
        },
        {
            "title": "Availability Period Duration",
            "sub_prompt": "What is the duration of the availability period?",
            "rag_params": {
                "retrieval_mode": "hybrid",
                "bm25_weight": 0.5,
                "semantic_weight": 0.5,
                "bm25_terms": ["availability period", "commitment period", "drawdown period"],
                "reasoning": "Standard loan timing term with consistent phrasing benefits from balanced retrieval"
            }
        }
    ]
}

Example Input Prompt + Context:
Document First Page Preview: "...Guarantee Agreement..."
Document Index Preview: ""
User Prompt: "What happens if the borrower defaults?"

Example JSON Output:
{
    "user_request_context": "",
    "decomposition": [
        {
            "title": "Borrower Default Consequences",
            "sub_prompt": "What happens if the borrower defaults?",
            "rag_params": {
                "retrieval_mode": "semantic",
                "bm25_weight": 0.3,
                "semantic_weight": 0.7,
                "bm25_terms": ["event of default", "default remedies", "acceleration"],
                "reasoning": "Remedy language varies significantly; prioritize semantic matching to capture concept variations"
            }
        }
    ]
}

Example Input Prompt + Context:
Document First Page Preview: "...Fee Letter for Revolving Credit Facility..."
Document Index Preview: "Schedule 1 - Applicable Margin (p.37)"
User Prompt: "What is the drawn margin and the undrawn commitment fee?"

Example JSON Output:
{
    "user_request_context": "",
    "decomposition": [
        {
            "title": "Drawn Margin Rate",
            "sub_prompt": "What is the drawn applicable margin?",
            "rag_params": {
                "retrieval_mode": "bm25_dominant",
                "bm25_weight": 0.65,
                "semantic_weight": 0.35,
                "bm25_terms": ["applicable margin", "drawn margin"],
                "reasoning": "Margin percentages appear verbatim in fee schedules; prioritize exact keyword hits",
            },
            "keywords": ["applicable margin", "drawn margin"]
        },
        {
            "title": "Undrawn Commitment Fee",
            "sub_prompt": "What is the commitment fee on undrawn amounts?",
            "rag_params": {
                "retrieval_mode": "bm25_dominant",
                "bm25_weight": 0.65,
                "semantic_weight": 0.35,
                "bm25_terms": ["applicable margin", "drawn margin", "margin on advances"],
                "reasoning": "Margin percentages appear verbatim in fee schedules; prioritize exact keyword hits",
            }
        },
        {
            "title": "Undrawn Commitment Fee",
            "sub_prompt": "What is the commitment fee on undrawn amounts?",
            "rag_params": {
                "retrieval_mode": "bm25_dominant",
                "bm25_weight": 0.7,
                "semantic_weight": 0.3,
                "bm25_terms": ["commitment fee", "undrawn commitment", "unutilized commitment"],
                "reasoning": "Fee tables list exact phrasing; BM25 should lead with minimal semantic support",
            }
        }
    ]
}

Example Input Prompt + Context:
Document First Page Preview: "...Investment Agreement Number 51200..."
Document Index Preview: ""
User Prompt: "What is the investment number?"

Example JSON Output:
{
    "user_request_context": "",
    "decomposition": [
        {
            "title": "Investment Number",
            "sub_prompt": "What is the investment number?",
            "rag_params": {
                "retrieval_mode": "bm25_dominant",
                "bm25_weight": 0.7,
                "semantic_weight": 0.3,
                "bm25_terms": ["investment number", "investment no", "investment #"],
                "reasoning": "Specific identifier requires precise lexical matching",
            }
        }
    ]
}
"""

    first_page_preview = (first_page_preview or "").strip()
    document_index_preview = (document_index_preview or "").strip()

    context_sections: List[str] = []
    if first_page_preview:
        context_sections.append(
            "Document First Page Preview:\n<<<FIRST_PAGE_START>>>\n"
            + first_page_preview
            + "\n<<<FIRST_PAGE_END>>>"
        )
    if document_index_preview:
        context_sections.append(
            "Document Index Preview:\n<<<DOCUMENT_INDEX_START>>>\n"
            + document_index_preview
            + "\n<<<DOCUMENT_INDEX_END>>>"
        )

    context_block = "\n\n".join(context_sections) if context_sections else "Document preview unavailable."

    human_prompt = (
        "Use the document preview and user prompt below to follow the system instructions and emit the JSON object.\n\n"
        f"{context_block}\n\n"
        f"User Prompt:\n{user_prompt}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    fallback_result = {
        "user_request_context": "",
        "decomposition": [{
            "title": "Overall Analysis",
            "sub_prompt": user_prompt,
            "rag_params": {
                "retrieval_mode": "hybrid",
                "bm25_weight": 0.5,
                "semantic_weight": 0.5,
                "bm25_terms": [],
                "reasoning": "Default balanced weights due to decomposition failure",
                
            },
            "bm25_terms": []
        }]
    }

    def _clean_keyword_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value.strip())

    def _split_synonyms_if_needed(value: Any) -> List[str]:
        cleaned = _clean_keyword_text(value)
        if not cleaned:
            return []
        if re.search(r"\s*/\s*", cleaned):
            parts = [_clean_keyword_text(part) for part in re.split(r"\s*/\s*", cleaned)]
            parts = [part for part in parts if part]
            if len(parts) > 1:
                return parts
        return [cleaned]

    def _extract_terms_from_dict(payload: Dict[str, Any]) -> List[str]:
        if not isinstance(payload, dict):
            return []
        collected: List[str] = []
        for key in ("keywords", "terms", "phrases", "variants", "values"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    collected.extend(_split_synonyms_if_needed(item))
            elif isinstance(value, str):
                collected.extend(_split_synonyms_if_needed(value))
        if not collected:
            fallback_value = payload.get("keyword") or payload.get("term") or payload.get("phrase")
            collected.extend(_split_synonyms_if_needed(fallback_value))
        unique_terms: List[str] = []
        seen_terms: Set[str] = set()
        for candidate in collected:
            lowered = candidate.lower()
            if lowered in seen_terms:
                continue
            seen_terms.add(lowered)
            unique_terms.append(candidate)
        return unique_terms

    def _add_keyword_group(
        group_terms: List[str],
        keyword_groups: List[List[str]],
        normalized_keywords: List[str],
        seen_keywords: Set[str],
    ) -> None:
        deduped_group: List[str] = []
        seen_group_terms: Set[str] = set()
        for term in group_terms:
            cleaned_term = _clean_keyword_text(term)
            if not cleaned_term:
                continue
            lowered_term = cleaned_term.lower()
            if lowered_term in seen_group_terms:
                continue
            seen_group_terms.add(lowered_term)
            deduped_group.append(cleaned_term)
        if not deduped_group:
            return
        keyword_groups.append(deduped_group)
        for term in deduped_group:
            lowered_term = term.lower()
            if lowered_term in seen_keywords:
                continue
            seen_keywords.add(lowered_term)
            normalized_keywords.append(term)

    def _normalize_retrieval_mode(value: Any) -> str:
        if isinstance(value, str):
            mode = value.strip().lower()
        else:
            mode = "hybrid"

        if mode in {"keyword", "keyword_only", "keyword-mode", "exact", "bm25_only"}:
            return "keyword"
        if mode in {"semantic", "semantic_only", "semantic_dominant", "semantic-heavy"}:
            return "semantic"
        if mode in {"bm25", "bm25_dominant", "lexical", "bm25-heavy"}:
            return "bm25_dominant"
        return "hybrid"

    def _collect_keywords_from_value(value: Any) -> List[str]:
        collected: List[str] = []

        def _handle(candidate: Any) -> None:
            if isinstance(candidate, str):
                collected.extend(_split_synonyms_if_needed(candidate))
            elif isinstance(candidate, (list, tuple, set)):
                for item in candidate:
                    _handle(item)
            elif isinstance(candidate, dict):
                collected.extend(_extract_terms_from_dict(candidate))

        _handle(value)

        unique_terms: List[str] = []
        seen_terms: Set[str] = set()
        for term in collected:
            cleaned = _clean_keyword_text(term)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen_terms:
                continue
            seen_terms.add(lowered)
            unique_terms.append(cleaned)
        return unique_terms

    def _normalize_bm25_terms_field(
        value: Any,
        fallback_terms: Optional[List[str]] = None,
        fallback_prompt: str = "",
    ) -> List[str]:
        collected = _collect_keywords_from_value(value)

        if not collected and isinstance(value, str):
            # Split on OR and commas while keeping quoted phrases
            segments = re.split(r"\bOR\b|,", value, flags=re.IGNORECASE)
            for segment in segments:
                segment_clean = segment.strip().strip('"').strip("'")
                if segment_clean:
                    collected.append(segment_clean)

        if not collected and fallback_terms:
            collected = list(fallback_terms)

        if not collected and isinstance(fallback_prompt, str):
            fallback_clean = _clean_keyword_text(fallback_prompt)
            if fallback_clean:
                collected = [fallback_clean]

        normalized_terms: List[str] = []
        seen_terms: Set[str] = set()
        for term in collected:
            cleaned_term = _clean_keyword_text(term)
            if not cleaned_term:
                continue
            lowered_term = cleaned_term.lower()
            if lowered_term in seen_terms:
                continue
            seen_terms.add(lowered_term)
            normalized_terms.append(cleaned_term)

        return normalized_terms


    try:
        parsed_json: Any = None
        raw_response: Optional[str] = None
        generation_metadata = {"operation": "ask_mode.decomposition"}

        with optional_context(
            start_generation(
                name="ask_mode.decomposition",
                input_data={"messages": messages},
                metadata=generation_metadata,
                model=DECOMPOSITION_MODEL_NAME,
            )
        ) as generation:
            try:
                result = await analyzer._get_json_with_retries(
                    messages=messages,
                    model_name=DECOMPOSITION_MODEL_NAME,
                    context="ask mode prompt decomposition",
                    return_raw=True,
                )
            except Exception as err:
                record_generation_error(
                    generation,
                    err,
                    metadata={"stage": "ask_mode.prompt_decomposition"},
                )
                raise

            if isinstance(result, tuple) and len(result) == 2:
                parsed_json, raw_response = result
            else:
                parsed_json = result
                raw_response = None

            response_preview = (
                raw_response[:500] if isinstance(raw_response, str) else None
            )
            set_generation_output(
                generation,
                output=parsed_json,
                metadata={"raw_response_preview": response_preview},
            )

        if parsed_json is None:
            logger.error("Decomposition LLM returned no data; using fallback result")
            return fallback_result
    except json.JSONDecodeError as json_err:
        logger.error("Failed to parse decomposition response as JSON after retries: %s", json_err)
        logger.warning("Falling back to using the original prompt due to JSON parsing error.")
        return fallback_result
    except TimeoutError:
        logger.error("Prompt decomposition request timed out. Falling back to original prompt.")
        return fallback_result
    except Exception as err:
        logger.error("Error during prompt decomposition LLM call: %s", err, exc_info=True)
        logger.warning("Falling back to using the original prompt.")
        return fallback_result

    # Parse successful response
    try:
        if isinstance(parsed_json, dict) and "decomposition" in parsed_json:
            decomposition_list = parsed_json["decomposition"]
            if isinstance(decomposition_list, list):
                valid_items: List[Dict[str, Any]] = []

                for item in decomposition_list:
                    if not isinstance(item, dict) or "title" not in item or "sub_prompt" not in item:
                        logger.warning("Skipping invalid decomposition item (missing title or sub_prompt): %s", item)
                        continue

                    if not isinstance(item["title"], str) or not isinstance(item["sub_prompt"], str):
                        logger.warning("Skipping invalid decomposition item (non-string title or sub_prompt): %s", item)
                        continue

                    if not item["title"].strip() or not item["sub_prompt"].strip():
                        logger.warning("Skipping decomposition item with empty title or sub_prompt")
                        continue

                    rag_params = item.get("rag_params")
                    if not isinstance(rag_params, dict):
                        logger.warning("Missing or invalid rag_params for '%s', using defaults", item["title"])
                        rag_params = {
                            "retrieval_mode": "hybrid",
                            "bm25_weight": 0.5,
                            "semantic_weight": 0.5,
                            "reasoning": "Default balanced weights (rag_params missing from LLM response)"
                        }
                        item["rag_params"] = rag_params

                    retrieval_mode_value = _normalize_retrieval_mode(rag_params.get("retrieval_mode"))
                    bm25_weight = rag_params.get("bm25_weight", 0.5)
                    semantic_weight = rag_params.get("semantic_weight", 0.5)

                    try:
                        bm25_weight = float(bm25_weight)
                        semantic_weight = float(semantic_weight)
                    except (ValueError, TypeError) as err:
                        logger.warning("Invalid RAG weight types for '%s': %s. Using defaults.", item["title"], err)
                        bm25_weight = 1.0 if retrieval_mode_value == "keyword" else 0.5
                        semantic_weight = 0.0 if retrieval_mode_value == "keyword" else 0.5
                    else:
                        if retrieval_mode_value == "keyword":
                            bm25_weight, semantic_weight = 1.0, 0.0
                        else:
                            if not (0.0 <= bm25_weight <= 1.0) or not (0.0 <= semantic_weight <= 1.0):
                                logger.warning(
                                    "RAG weights out of range for '%s': bm25=%s, semantic=%s. Using defaults.",
                                    item["title"],
                                    bm25_weight,
                                    semantic_weight,
                                )
                                bm25_weight, semantic_weight = 0.5, 0.5

                            weight_sum = bm25_weight + semantic_weight
                            if abs(weight_sum - 1.0) > 0.01:
                                logger.warning(
                                    "RAG weights don't sum to 1.0 for '%s' (sum=%.3f). Normalizing.",
                                    item["title"],
                                    weight_sum,
                                )
                                if weight_sum > 0:
                                    bm25_weight = bm25_weight / weight_sum
                                    semantic_weight = semantic_weight / weight_sum
                                else:
                                    bm25_weight, semantic_weight = 0.5, 0.5

                    rag_params["bm25_weight"] = bm25_weight
                    rag_params["semantic_weight"] = semantic_weight
                    rag_params["retrieval_mode"] = retrieval_mode_value

                    if "reasoning" not in rag_params or not isinstance(rag_params.get("reasoning"), str):
                        rag_params["reasoning"] = "No reasoning provided"

                    logger.info(
                        "RAG params for '%s': mode=%s, BM25=%.2f, Semantic=%.2f, Reasoning: %s",
                        item["title"],
                        retrieval_mode_value,
                        bm25_weight,
                        semantic_weight,
                        rag_params.get("reasoning", "N/A"),
                    )

                    keyword_candidates: List[str] = []
                    for key in ("keywords", "keyword_terms", "keyword_list", "keyword_synonyms"):
                        if key in item:
                            keyword_candidates.extend(_collect_keywords_from_value(item.get(key)))

                    if isinstance(rag_params.get("keywords"), (list, tuple, set, dict, str)):
                        keyword_candidates.extend(_collect_keywords_from_value(rag_params.get("keywords")))
                        rag_params.pop("keywords", None)

                    deduped_keywords: List[str] = []
                    seen_keyword_terms_local: Set[str] = set()
                    for candidate in keyword_candidates:
                        cleaned_candidate = _clean_keyword_text(candidate)
                        if not cleaned_candidate:
                            continue
                        lowered_candidate = cleaned_candidate.lower()
                        if lowered_candidate in seen_keyword_terms_local:
                            continue
                        seen_keyword_terms_local.add(lowered_candidate)
                        deduped_keywords.append(cleaned_candidate)

                    item["keywords"] = deduped_keywords

                    if retrieval_mode_value == "keyword" and not deduped_keywords:
                        logger.warning(
                            "retrieval_mode 'keyword' selected for '%s' but no keywords were extracted.",
                            item["title"],
                        )

                    bm25_terms = _normalize_bm25_terms_field(
                        rag_params.get("bm25_terms"),
                        fallback_terms=deduped_keywords,
                        fallback_prompt=item["sub_prompt"],
                    )
                    rag_params["bm25_terms"] = bm25_terms
                    item["bm25_terms"] = bm25_terms

                    # HyDE output disabled: remove any 'hyde' fields provided by the model
                    rag_params.pop("hyde", None)
                    item.pop("hyde", None)

                    valid_items.append(item)

                if not valid_items:
                    logger.warning("Decomposition resulted in an empty list after filtering. Falling back.")
                    return fallback_result

                user_request_context = parsed_json.get("user_request_context", "")
                user_request_context = parsed_json.get("user_request_context", "")

                if not isinstance(user_request_context, str):
                    user_request_context = ""
                else:
                    user_request_context = user_request_context.strip()

                result_payload = {
                    "user_request_context": user_request_context,
                    "decomposition": valid_items,
                }

                logger.info(
                    "Successfully decomposed prompt into %d sub-prompts (user_context_present=%s)",
                    len(valid_items),
                    bool(user_request_context),
                )
                return result_payload

            logger.warning("Decomposition JSON found, but 'decomposition' key is not a list. Falling back.")
            return fallback_result

        logger.warning("Decomposition JSON parsed, but missing 'decomposition' key or wrong structure. Falling back.")
        return fallback_result
    except Exception as parse_err:
        logger.error("Error parsing decomposition result structure: %s", parse_err, exc_info=True)
        logger.warning("Falling back to using the original prompt.")
        return fallback_result


# Backward compatibility alias
async def decompose_prompt(analyzer, user_prompt: str) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for decompose_ask_mode_prompt.
    This function is deprecated and will be removed in a future version.
    Use decompose_ask_mode_prompt instead.
    """
    logger.warning("decompose_prompt is deprecated. Use decompose_ask_mode_prompt instead.")
    return await decompose_ask_mode_prompt(analyzer, user_prompt)

# --- Review Mode Decomposition Models ---

class ReviewModeSubPrompt(BaseModel):
    """A single sub-prompt/rule for Review Mode validation."""
    title: str = Field(..., description="Concise title (max 5-6 words)")
    sub_prompt: str = Field(..., description="Re-written/clarified rule text")
    validation_type: Literal['regex', 'semantic'] = Field(..., description="Validation approach: regex or semantic")
    validation_reasoning: str = Field(..., description="Reasoning for choosing regex vs semantic")
    extracted_examples: List[str] = Field(default_factory=list, description="Examples extracted from the rule text (for illustration only)")
    violation_examples: List[str] = Field(default_factory=list, description="Generated examples of what WOULD violate this rule")
    compliance_examples: List[str] = Field(default_factory=list, description="Generated examples of what WOULD comply with this rule")


class ReviewModeDecomposition(BaseModel):
    """Complete decomposition result for Review Mode."""
    sub_prompts: List[ReviewModeSubPrompt] = Field(..., description="List of decomposed sub-prompts/rules")


async def decompose_review_mode_prompt(analyzer, user_prompt: str) -> List[ReviewModeSubPrompt]:
    """
    Decomposes a Review Mode prompt (validation rules) into structured sub-prompts.
    This function does NOT include document text and does NOT use RAG parameters.
    Instead, it focuses on:
    1. Breaking down the rule into sub-rules
    2. Clarifying each rule
    3. Determining validation approach (regex vs semantic)
    4. Extracting examples from rule text
    5. Generating violation and compliance examples

    Returns a list of ReviewModeSubPrompt objects.
    Returns a single sub-prompt with the original text on failure.
    """
    logger.info(f"[REVIEW MODE] Decomposing validation rules: '{user_prompt[:100]}...'")

    system_prompt = """You are an expert AI system that analyzes validation rules for financial and legal documents.

Your task is to break down the user's validation rule(s) into individual, structured sub-rules. For each sub-rule, you must:

1. **Create a concise title** (max 5-6 words)
2. **Rewrite/clarify the rule** to clearly explain:
   - What the rule is checking for
   - What constitutes a VIOLATION (non-compliant text)
   - What constitutes COMPLIANCE (correct text)
3. **Choose validation approach** (regex vs semantic) with reasoning
4. **Extract examples** from the rule text (e.g., text in parentheses like "(e.g., ...)")
5. **Generate violation examples** (2-3 examples of text that WOULD violate this rule)
6. **Generate compliance examples** (2-3 examples of text that WOULD comply with this rule)

## WHEN TO USE REGEX VS SEMANTIC:

**Use REGEX when:**
- The rule checks for a SPECIFIC, WELL-DEFINED pattern that can be exhaustively enumerated
- Examples: date formats (MM/DD/YYYY), specific numeric patterns (phone numbers, IDs), exact string matches
- The rule checks a SMALL, FIXED set of values (e.g., checking for 3-5 specific currency names)
- Pattern matching is deterministic and doesn't require understanding context or meaning

**Use SEMANTIC when:**
- The rule requires understanding MEANING, CONTEXT, or INTENT
- Examples: word confusion (decease vs decrease, principal vs principle, affect vs effect)
- The rule involves OPEN-ENDED sets that cannot be exhaustively listed
- Examples: "all currency references" (there are 180+ world currencies), "proper capitalization of country names"
- The rule requires CASE SENSITIVITY checks across diverse terms (e.g., "Indian rupee" vs "Indian Rupee")
- The rule involves CALCULATIONS, COMPARISONS, or LOGICAL REASONING
- The rule checks for TONE, STYLE, or APPROPRIATENESS

## SPECIFIC GUIDANCE FOR COMMON RULES:
- Currency case sensitivity (e.g., "Indian rupee" not "Indian Rupee"): Use SEMANTIC
  Reason: There are 180+ currencies; regex cannot enumerate all. Semantic can understand capitalization rules.
- Word confusion (decease/decrease, principal/principle): Use SEMANTIC
  Reason: Requires understanding context to determine if the word is used correctly.
- Specific date format (e.g., "Month DD, YYYY"): Use REGEX if checking format only; use SEMANTIC if also validating logical correctness
- Decimal precision for numbers (e.g., "1.0 billion" not "1 billion"): Use REGEX
  Reason: This is a specific numeric pattern that can be precisely matched.
- ISO currency codes (USD, EUR, GBP): Use REGEX if checking a small set; use SEMANTIC if checking all possible codes

## OUTPUT FORMAT:

Your entire response MUST be a single JSON object with this structure:
{
  "sub_prompts": [
    {
      "title": "Concise Title",
      "sub_prompt": "Clarified rule text explaining violations vs compliance",
      "validation_type": "regex" | "semantic",
      "validation_reasoning": "Brief explanation of why this approach was chosen",
      "extracted_examples": ["example1", "example2"],
      "violation_examples": ["text that violates", "another violation"],
      "compliance_examples": ["text that complies", "another compliant example"]
    }
  ]
}

Do not include any explanations, introductory text, or markdown formatting outside the JSON structure.

## EXAMPLE:

Input: "Check that currency names use proper case (e.g., 'U.S. dollar' not 'U.S. Dollar'). Also verify dates are in MM/DD/YYYY format."

Output:
{
  "sub_prompts": [
    {
      "title": "Currency Case Sensitivity",
      "sub_prompt": "Currency names must use proper case sensitivity where only the country/region name is capitalized, not the currency unit. VIOLATION: 'U.S. Dollar', 'Indian Rupee' (currency unit capitalized). COMPLIANT: 'U.S. dollar', 'Indian rupee' (only country capitalized).",
      "validation_type": "semantic",
      "validation_reasoning": "There are 180+ world currencies; regex cannot enumerate all. Semantic validation can understand capitalization rules across all currency names.",
      "extracted_examples": ["U.S. dollar", "U.S. Dollar"],
      "violation_examples": ["U.S. Dollar", "Indian Rupee", "European Euro"],
      "compliance_examples": ["U.S. dollar", "Indian rupee", "European euro"]
    },
    {
      "title": "Date Format MM/DD/YYYY",
      "sub_prompt": "Dates must be formatted as MM/DD/YYYY with two-digit month, two-digit day, and four-digit year, separated by forward slashes. VIOLATION: '2024-10-24', '10/24/24', 'October 24, 2024'. COMPLIANT: '10/24/2024', '01/15/2024'.",
      "validation_type": "regex",
      "validation_reasoning": "Date format MM/DD/YYYY is a specific, well-defined pattern that can be precisely matched with regex.",
      "extracted_examples": [],
      "violation_examples": ["2024-10-24", "10/24/24", "October 24, 2024"],
      "compliance_examples": ["10/24/2024", "01/15/2024", "12/31/2024"]
    }
  ]
}
"""

    human_prompt = f"""Analyze the following validation rule(s) and return the decomposed sub-rules with all required information as a JSON object according to the system instructions:

{user_prompt}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    # Fallback result in case of errors
    fallback_result = [ReviewModeSubPrompt(
        title="Overall Validation",
        sub_prompt=user_prompt,
        validation_type="semantic",
        validation_reasoning="Default semantic validation due to decomposition failure",
        extracted_examples=[],
        violation_examples=[],
        compliance_examples=[]
    )]

    try:
        try:
            parsed, raw_response = await analyzer._get_json_with_retries(
                messages=messages,
                model_name=DECOMPOSITION_MODEL_NAME,
                context="review mode prompt decomposition",
                return_raw=True,
            )
        except json.JSONDecodeError as json_err:
            logger.error("Failed to parse Review Mode decomposition response as JSON after retries: %s", json_err)
            logger.warning("Falling back to using the original prompt due to JSON parsing error.")
            return fallback_result

        if not raw_response:
            logger.warning("Empty response from Review Mode decomposition LLM. Falling back.")
            return fallback_result

        logger.debug(f"[REVIEW MODE] Raw decomposition response: {raw_response[:500]}...")

        # Validate structure
        if not isinstance(parsed, dict) or "sub_prompts" not in parsed:
            logger.error(
                "Invalid Review Mode decomposition structure. Expected dict with 'sub_prompts' key. Got: %s",
                type(parsed),
            )
            return fallback_result

        sub_prompts_data = parsed["sub_prompts"]
        if not isinstance(sub_prompts_data, list) or len(sub_prompts_data) == 0:
            logger.warning("Review Mode decomposition returned empty or invalid sub_prompts list. Falling back.")
            return fallback_result

        # Parse into Pydantic models
        result = []
        for item in sub_prompts_data:
            try:
                sub_prompt = ReviewModeSubPrompt(**item)
                result.append(sub_prompt)
                logger.info(
                    "[REVIEW MODE] Parsed sub-prompt: '%s' (validation_type=%s)",
                    sub_prompt.title,
                    sub_prompt.validation_type,
                )
            except Exception as e:
                logger.warning(f"Failed to parse Review Mode sub-prompt: {e}. Item: {item}")
                continue

        if not result:
            logger.warning("No valid Review Mode sub-prompts parsed. Falling back.")
            return fallback_result

        logger.info(f"[REVIEW MODE] Successfully decomposed into {len(result)} sub-prompts")
        return result

    except TimeoutError:
        logger.error("Review Mode decomposition request timed out. Falling back to original prompt.")
        return fallback_result
    except Exception as e:
        logger.error(f"Error during Review Mode decomposition LLM call: {str(e)}", exc_info=True)
        logger.warning("Falling back to using the original prompt.")
        return fallback_result


# --- Regex Generation Models ---

class RegexGenerationResult(BaseModel):
    """Result of regex pattern generation for a validation rule."""
    regex_pattern: str = Field(..., description="The generated Python-compatible regex pattern")
    explanation: str = Field(..., description="Explanation of what the regex matches and how it works")
    test_matches: List[str] = Field(default_factory=list, description="Example strings that SHOULD match this pattern")
    test_non_matches: List[str] = Field(default_factory=list, description="Example strings that should NOT match this pattern")


async def generate_regex_pattern(analyzer, sub_prompt: ReviewModeSubPrompt) -> Optional[RegexGenerationResult]:
    """
    Generates a regex pattern for a Review Mode sub-prompt that was flagged as 'regex' validation.

    Args:
        analyzer: The DocumentAnalyzer instance for making LLM calls
        sub_prompt: The ReviewModeSubPrompt containing the rule details

    Returns:
        RegexGenerationResult with the generated pattern, or None on failure
    """
    logger.info(f"[REGEX GENERATION] Generating regex pattern for rule: '{sub_prompt.title}'")

    system_prompt = """You are an expert at creating robust, production-ready Python regex patterns for document validation.

Your task is to generate a regex pattern based on the provided validation rule and examples.

## CRITICAL REGEX GUIDELINES:

### Word Boundaries and Decimal Numbers:
- PROBLEM: Word boundary \\b treats '.' as a boundary, so "67.3 billion" is seen as two tokens: "67" and "3"
  If you write \\b\\d+\\s+billion\\b, it will match BOTH "67 billion" AND "3 billion" (from "67.3 billion")

- SOLUTION: Use negative lookbehind and lookahead to prevent matching decimal parts:
  * (?<!\\.) - No decimal point immediately before the number
  * (?<!\\d\\.) - No digit-dot pattern before (prevents matching "3" in "67.3")
  * (?!\\.\\d+) - No decimal point after the number

- EXAMPLE: To match integers like "67 billion" but NOT decimals like "67.3 billion":
  Pattern: (?<!\\.)(?<!\\d\\.)\\b(?:\\d{1,3}(?:,\\d{3})*|\\d+)(?!\\.\\d+)\\s+billion\\b

  This pattern:
  Γ£ô Matches: "67 billion", "1,234 billion", "5 billion"
  Γ£ù Does NOT match: "67.3 billion", "1.0 billion", "5.5 billion"

### General Best Practices:
1. Use raw strings (r"...") for Python regex patterns
2. Be specific about boundaries to avoid partial matches
3. Handle common variations (spaces, punctuation, case if needed)
4. Test edge cases (numbers with commas, decimals, etc.)
5. Use non-capturing groups (?:...) when grouping is needed but capture isn't
6. Escape special regex characters: . ^ $ * + ? { } [ ] \\ | ( )

### Common Patterns:
- Date MM/DD/YYYY: \\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\\d{4}\\b
- Phone (US): \\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b
- Email: \\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b
- Currency codes: \\b(USD|EUR|GBP|JPY)\\b
- Numbers with commas: \\b\\d{1,3}(?:,\\d{3})*\\b

## OUTPUT FORMAT:

Your entire response MUST be a single JSON object with this structure:
{
  "regex_pattern": "your_regex_pattern_here",
  "explanation": "Clear explanation of what this pattern matches and how it works",
  "test_matches": ["example1 that should match", "example2 that should match"],
  "test_non_matches": ["example1 that should NOT match", "example2 that should NOT match"]
}

Do not include any explanations, introductory text, or markdown formatting outside the JSON structure.
The regex_pattern should be a valid Python regex string (without the r"" prefix - that will be added automatically).
"""

    # Build context from the sub_prompt
    examples_context = ""
    if sub_prompt.extracted_examples:
        examples_context += f"\n\nExtracted examples from rule text:\n" + "\n".join([f"- {ex}" for ex in sub_prompt.extracted_examples])

    if sub_prompt.violation_examples:
        examples_context += f"\n\nExamples that SHOULD be flagged (violations):\n" + "\n".join([f"- {ex}" for ex in sub_prompt.violation_examples])

    if sub_prompt.compliance_examples:
        examples_context += f"\n\nExamples that should NOT be flagged (compliant):\n" + "\n".join([f"- {ex}" for ex in sub_prompt.compliance_examples])

    human_prompt = f"""Generate a robust Python regex pattern for the following validation rule:

**Rule Title:** {sub_prompt.title}

**Clarified Rule:**
{sub_prompt.sub_prompt}

**Validation Reasoning:**
{sub_prompt.validation_reasoning}
{examples_context}

Please generate a regex pattern that will accurately identify violations of this rule. Return your response as a JSON object according to the system instructions."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    try:
        try:
            parsed, raw_response = await analyzer._get_json_with_retries(
                messages=messages,
                model_name=DECOMPOSITION_MODEL_NAME,
                context=f"regex generation for {sub_prompt.title}",
                return_raw=True,
            )
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse regex generation response as JSON after retries: {json_err}")
            return None

        if not raw_response:
            logger.warning(f"Empty response from regex generation for '{sub_prompt.title}'")
            return None

        logger.debug(f"[REGEX GENERATION] Raw response: {raw_response[:500]}...")

        try:
            result = RegexGenerationResult(**parsed)
        except Exception as parse_err:
            logger.error(f"Failed to parse regex generation result: {parse_err}")
            return None

        # Validate the regex pattern by attempting to compile it
        try:
            re.compile(result.regex_pattern)
            logger.info(f"[REGEX GENERATION] Successfully generated and validated regex for '{sub_prompt.title}'")
            logger.debug(f"[REGEX GENERATION] Pattern: {result.regex_pattern}")
            return result
        except re.error as regex_err:
            logger.error(f"Generated regex pattern is invalid: {regex_err}. Pattern: {result.regex_pattern}")
            return None

    except Exception as e:
        logger.error(f"Error during regex generation for '{sub_prompt.title}': {str(e)}", exc_info=True)
        return None
