"""
Prompt decomposition functionality.
"""

import json
import re
from typing import Any, Dict, List, Optional, Literal, Set

from pydantic import BaseModel, Field

from ..config import logger, DECOMPOSITION_MODEL_NAME, USE_DATABRICKS_LLM


async def decompose_ask_mode_prompt(analyzer, user_prompt: str) -> Dict[str, Any]:
    """Decompose the prompt and decide whether keyword mode should run."""

    logger.info("[ASK MODE] Decomposing prompt with RAG optimization: '%s'", user_prompt[:100])

    system_prompt = """You are a helpful assistant specializing in financial and legal document analysis. Your task is to analyze the user's prompt and identify distinct questions or analysis tasks within it.

First, decide whether the request is best served by a deterministic keyword-only retrieval (Keyword Mode).

Trigger Keyword Mode when ALL of the following are true:
- The user is asking for exact term lookups, specific form fields, codes, or literal phrases.
- The desired outputs are counts or locations of those exact terms.

Do NOT use Keyword Mode when the prompt requires interpretation, synthesis, summarization, or reasoning beyond literal keyword matches.

If you choose Keyword Mode you MUST also emit the explicit keywords that should be searched.

Return your answer as a single JSON object with the keys:
- "keyword_mode": boolean
- "keyword_reasoning": short string justifying your decision
- "keywords": list where each item is either a single keyword string or a list of related keywords/phrases that should be searched together (e.g., slash-separated synonyms)
- "user_request_context": short string capturing any explicit instructions the user included about how to handle the results (return an empty string if none were provided)
- "decomposition": list of sub-prompt objects exactly as described below

Break down the prompt into a list of self-contained, individual questions or tasks. For each task, provide:
1. A concise, descriptive title (max 5-6 words)
2. The full sub-prompt text
3. Optimal RAG (Retrieval-Augmented Generation) parameters for document retrieval

RAG Parameters Guidelines:
- For keyword-based queries (MT599 Swift, specific codes, exact terms): Use higher BM25 weight (0.7-0.8) for precise keyword matching
- For general legal/financial queries: Use balanced weights (BM25: 0.5, semantic: 0.5)
- For conceptual/interpretive queries: Use higher semantic weight (0.6-0.7) for meaning-based retrieval
- For technical terminology queries: Use slightly higher BM25 weight (0.6) for precise terms
- BM25 weight + semantic weight should always equal 1.0
- Provide brief reasoning for your weight selection

Your entire response MUST be a single JSON object with the keys described earlier. Each item in the "decomposition" list must have:
- "title": string (the concise title)
- "sub_prompt": string (the full sub-prompt text)
- "rag_params": object with:
  - "bm25_weight": number (0.0-1.0)
  - "semantic_weight": number (0.0-1.0)
  - "reasoning": string (brief explanation)

Do not include any explanations, introductory text, or markdown formatting outside the JSON structure.

Example Input Prompt:
"What is the defined / lawful loan currency?
What is the duration of the availability period?
What is the loan amount and currency?"

Example JSON Output:
{
  "keyword_mode": false,
  "keyword_reasoning": "Prompt requests broader analysis, not literal matches",
  "keywords": [],
  "user_request_context": "",
  "decomposition": [
    {
      "title": "Lawful Loan Currency",
      "sub_prompt": "What is the defined / lawful loan currency?",
      "rag_params": {
        "bm25_weight": 0.6,
        "semantic_weight": 0.4,
        "reasoning": "Legal terminology query benefits from keyword precision"
      }
    },
    {
      "title": "Availability Period Duration",
      "sub_prompt": "What is the duration of the availability period?",
      "rag_params": {
        "bm25_weight": 0.5,
        "semantic_weight": 0.5,
        "reasoning": "Balanced approach for standard financial term"
      }
    },
    {
      "title": "Loan Amount",
      "sub_prompt": "What is the loan amount?",
      "rag_params": {
        "bm25_weight": 0.6,
        "semantic_weight": 0.4,
        "reasoning": "Specific numerical data requires keyword matching"
      }
    },
    {
      "title": "Loan Currency",
      "sub_prompt": "What is the loan currency?",
      "rag_params": {
        "bm25_weight": 0.6,
        "semantic_weight": 0.4,
        "reasoning": "Currency codes are exact terms requiring keyword search"
      }
    }
  ]
}

Example Input Prompt:
"Analyze the termination clause and liability limitations in the loan agreement."

Example JSON Output:
{
  "keyword_mode": false,
  "keyword_reasoning": "Requires interpretive legal analysis",
  "keywords": [],
  "user_request_context": "",
  "decomposition": [
    {
      "title": "Termination Clause Analysis",
      "sub_prompt": "Analyze the termination clause in the loan agreement.",
      "rag_params": {
        "bm25_weight": 0.4,
        "semantic_weight": 0.6,
        "reasoning": "Conceptual legal analysis benefits from semantic understanding"
      }
    },
    {
      "title": "Liability Limitations Analysis",
      "sub_prompt": "Analyze the liability limitations in the loan agreement.",
      "rag_params": {
        "bm25_weight": 0.4,
        "semantic_weight": 0.6,
        "reasoning": "Interpretive legal query requires semantic retrieval"
      }
    }
  ]
}

Example Input Prompt:
"What is the MT599 Swift message format and field 79 content?"

Example JSON Output:
{
  "keyword_mode": true,
  "keyword_reasoning": "Prompt is focused on specific SWIFT terminology",
  "keywords": ["MT599", "Field 79"],
  "user_request_context": "",
  "decomposition": [
    {
      "title": "MT599 Swift Format",
      "sub_prompt": "What is the MT599 Swift message format?",
      "rag_params": {
        "bm25_weight": 0.8,
        "semantic_weight": 0.2,
        "reasoning": "MT599 Swift is highly specific terminology requiring exact keyword matching"
      }
    },
    {
      "title": "Field 79 Content",
      "sub_prompt": "What is the content of field 79?",
      "rag_params": {
        "bm25_weight": 0.75,
        "semantic_weight": 0.25,
        "reasoning": "Specific field number requires precise keyword search"
      }
    }
  ]
}

Example Input Prompt:
"What are the interest rates and fees for this loan?"

Example JSON Output:
{
  "keyword_mode": false,
  "keyword_reasoning": "Requires numerical comparison and interpretation",
  "keywords": [],
  "user_request_context": "",
  "decomposition": [
    {
      "title": "Loan Interest Rates",
      "sub_prompt": "What are the interest rates for this loan?",
      "rag_params": {
        "bm25_weight": 0.6,
        "semantic_weight": 0.4,
        "reasoning": "Numerical financial data requires keyword precision"
      }
    },
    {
      "title": "Loan Fees",
      "sub_prompt": "What are the fees for this loan?",
      "rag_params": {
        "bm25_weight": 0.6,
        "semantic_weight": 0.4,
        "reasoning": "Specific fee information benefits from keyword matching"
      }
    }
  ]
}
"""

    human_prompt = (
        "Analyze the following prompt and follow the system instructions to decide on Keyword Mode, "
        "identify keywords if applicable, and produce the decomposed sub-prompts in JSON format:\n\n"
        f"{user_prompt}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    fallback_result = {
        "keyword_mode": False,
        "keyword_reasoning": "Fallback: unable to run keyword decision.",
        "keywords": [],
        "keyword_groups": [],
        "user_request_context": "",
        "decomposition": [{
            "title": "Overall Analysis",
            "sub_prompt": user_prompt,
            "rag_params": {
                "bm25_weight": 0.5,
                "semantic_weight": 0.5,
                "reasoning": "Default balanced weights due to decomposition failure"
            }
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

    try:
        response_content = await analyzer._get_completion(messages, model_name=DECOMPOSITION_MODEL_NAME)

        try:
            cleaned_response = response_content.strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
            if match:
                json_str = match.group(1)
            elif cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                json_str = cleaned_response
            else:
                first_brace = cleaned_response.find("{")
                last_brace = cleaned_response.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = cleaned_response[first_brace:last_brace + 1]
                    logger.warning("Used basic brace finding for JSON extraction in decomposition.")
                else:
                    raise json.JSONDecodeError("Could not find JSON structure.", cleaned_response, 0)

            parsed_json = json.loads(json_str)

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

                        if "rag_params" not in item or not isinstance(item["rag_params"], dict):
                            logger.warning("Missing or invalid rag_params for '%s', using defaults", item["title"])
                            item["rag_params"] = {
                                "bm25_weight": 0.5,
                                "semantic_weight": 0.5,
                                "reasoning": "Default balanced weights (rag_params missing from LLM response)"
                            }
                        else:
                            rag_params = item["rag_params"]
                            bm25_weight = rag_params.get("bm25_weight", 0.5)
                            semantic_weight = rag_params.get("semantic_weight", 0.5)

                            try:
                                bm25_weight = float(bm25_weight)
                                semantic_weight = float(semantic_weight)

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

                                item["rag_params"]["bm25_weight"] = bm25_weight
                                item["rag_params"]["semantic_weight"] = semantic_weight

                                if "reasoning" not in rag_params or not isinstance(rag_params.get("reasoning"), str):
                                    item["rag_params"]["reasoning"] = "No reasoning provided"

                                logger.info(
                                    "RAG params for '%s': BM25=%.2f, Semantic=%.2f, Reasoning: %s",
                                    item["title"],
                                    bm25_weight,
                                    semantic_weight,
                                    rag_params.get("reasoning", "N/A"),
                                )

                            except (ValueError, TypeError) as err:
                                logger.warning("Invalid RAG weight types for '%s': %s. Using defaults.", item["title"], err)
                                item["rag_params"] = {
                                    "bm25_weight": 0.5,
                                    "semantic_weight": 0.5,
                                    "reasoning": "Default balanced weights (invalid weight values from LLM)"
                                }

                        valid_items.append(item)

                    if not valid_items:
                        logger.warning("Decomposition resulted in an empty list after filtering. Falling back.")
                        return fallback_result

                    keyword_mode_value = bool(parsed_json.get("keyword_mode", False))
                    raw_keywords = parsed_json.get("keywords", [])
                    keyword_reasoning = parsed_json.get("keyword_reasoning")
                    user_request_context = parsed_json.get("user_request_context", "")

                    if not isinstance(user_request_context, str):
                        user_request_context = ""
                    else:
                        user_request_context = user_request_context.strip()

                    keyword_groups: List[List[str]] = []
                    normalized_keywords: List[str] = []
                    seen_keywords: Set[str] = set()

                    if isinstance(raw_keywords, list):
                        for entry in raw_keywords:
                            group_terms: List[str] = []
                            if isinstance(entry, str):
                                group_terms.extend(_split_synonyms_if_needed(entry))
                            elif isinstance(entry, list):
                                local_seen: Set[str] = set()
                                for candidate in entry:
                                    for term in _split_synonyms_if_needed(candidate):
                                        lowered = term.lower()
                                        if lowered in local_seen:
                                            continue
                                        local_seen.add(lowered)
                                        group_terms.append(term)
                            elif isinstance(entry, dict):
                                group_terms.extend(_extract_terms_from_dict(entry))

                            if group_terms:
                                _add_keyword_group(group_terms, keyword_groups, normalized_keywords, seen_keywords)
                    elif isinstance(raw_keywords, str):
                        _add_keyword_group(_split_synonyms_if_needed(raw_keywords), keyword_groups, normalized_keywords, seen_keywords)

                    if keyword_mode_value and not normalized_keywords:
                        logger.warning("keyword_mode was true but no valid keywords were returned. Forcing keyword_mode to false.")
                        keyword_mode_value = False

                    result_payload = {
                        "keyword_mode": keyword_mode_value,
                        "keyword_reasoning": keyword_reasoning or "LLM did not supply keyword reasoning.",
                        "keywords": normalized_keywords,
                        "keyword_groups": keyword_groups,
                        "user_request_context": user_request_context,
                        "decomposition": valid_items,
                    }

                    logger.info(
                        "Successfully decomposed prompt into %d sub-prompts (keyword_mode=%s, keywords=%s, user_context_present=%s)",
                        len(valid_items),
                        keyword_mode_value,
                        normalized_keywords,
                        bool(user_request_context),
                    )
                    if keyword_groups:
                        logger.debug("Keyword groups extracted: %s", keyword_groups)
                    return result_payload

                logger.warning("Decomposition JSON found, but 'decomposition' key is not a list. Falling back.")
                return fallback_result

            logger.warning("Decomposition JSON parsed, but missing 'decomposition' key or wrong structure. Falling back.")
            return fallback_result

        except json.JSONDecodeError as json_err:
            logger.error("Failed to parse decomposition response as JSON: %s. Raw response: %s", json_err, response_content)
            logger.warning("Falling back to using the original prompt due to JSON parsing error.")
            return fallback_result

    except TimeoutError:
        logger.error("Prompt decomposition request timed out. Falling back to original prompt.")
        return fallback_result
    except Exception as err:
        logger.error("Error during prompt decomposition LLM call: %s", err, exc_info=True)
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
        response_content = await analyzer._get_completion(messages, model_name=DECOMPOSITION_MODEL_NAME)

        if not response_content:
            logger.warning("Empty response from Review Mode decomposition LLM. Falling back.")
            return fallback_result

        logger.debug(f"[REVIEW MODE] Raw decomposition response: {response_content[:500]}...")

        # Clean up response (remove markdown code blocks if present)
        cleaned_response = response_content.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            parsed = json.loads(cleaned_response)

            # Validate structure
            if not isinstance(parsed, dict) or "sub_prompts" not in parsed:
                logger.error(f"Invalid Review Mode decomposition structure. Expected dict with 'sub_prompts' key. Got: {type(parsed)}")
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
                    logger.info(f"[REVIEW MODE] Parsed sub-prompt: '{sub_prompt.title}' (validation_type={sub_prompt.validation_type})")
                except Exception as e:
                    logger.warning(f"Failed to parse Review Mode sub-prompt: {e}. Item: {item}")
                    continue

            if not result:
                logger.warning("No valid Review Mode sub-prompts parsed. Falling back.")
                return fallback_result

            logger.info(f"[REVIEW MODE] Successfully decomposed into {len(result)} sub-prompts")
            return result

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse Review Mode decomposition response as JSON: {json_err}. Raw response: {response_content}")
            logger.warning("Falling back to using the original prompt due to JSON parsing error.")
            return fallback_result

    except TimeoutError:
        logger.error(f"Review Mode decomposition request timed out. Falling back to original prompt.")
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
        response_content = await analyzer._get_completion(messages, model_name=DECOMPOSITION_MODEL_NAME)

        if not response_content:
            logger.warning(f"Empty response from regex generation for '{sub_prompt.title}'")
            return None

        logger.debug(f"[REGEX GENERATION] Raw response: {response_content[:500]}...")

        # Clean up response
        cleaned_response = response_content.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            parsed = json.loads(cleaned_response)
            result = RegexGenerationResult(**parsed)

            # Validate the regex pattern by attempting to compile it
            try:
                re.compile(result.regex_pattern)
                logger.info(f"[REGEX GENERATION] Successfully generated and validated regex for '{sub_prompt.title}'")
                logger.debug(f"[REGEX GENERATION] Pattern: {result.regex_pattern}")
                return result
            except re.error as regex_err:
                logger.error(f"Generated regex pattern is invalid: {regex_err}. Pattern: {result.regex_pattern}")
                return None

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse regex generation response as JSON: {json_err}. Raw response: {response_content}")
            return None
        except Exception as parse_err:
            logger.error(f"Failed to parse regex generation result: {parse_err}")
            return None

    except Exception as e:
        logger.error(f"Error during regex generation for '{sub_prompt.title}': {str(e)}", exc_info=True)
        return None
