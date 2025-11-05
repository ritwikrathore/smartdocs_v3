"""
SmartReview - AI-powered document validation tool.

This module provides functionality for creating validation templates and running
document compliance checks using regex and semantic validation.
"""

import re
import json
import asyncio
import fitz  # PyMuPDF
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# Import from centralized modules
from src.keyword_code.config import logger
from src.keyword_code.ai.databricks_llm import (
    get_databricks_llm,
    DATABRICKS_BASE_URL,
    DATABRICKS_LLM_MODEL
)
from src.keyword_code.utils.ui_helpers import (
    apply_ui_styling,
    render_branding
)

# --- Helper Functions ---

def _safe_str(obj, max_len=500):
    """Return a safe, truncated string representation for logging."""
    try:
        s = str(obj)
    except Exception:
        s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "... [truncated]"
    return s


async def _chat_completion_async(messages, model: Optional[str] = None, temperature: Optional[float] = None):
    """
    Async wrapper for Databricks LLM completion.
    Uses the centralized DatabricksLLMClient.
    """
    databricks_llm = get_databricks_llm()
    if not databricks_llm:
        logger.error("Databricks LLM client not initialized")
        raise RuntimeError("Databricks LLM client not initialized")

    # Use the async method from DatabricksLLMClient
    response_text = await databricks_llm.get_completion_async(messages, max_tokens=8192)

    # Wrap in a response-like object for compatibility
    class MockResponse:
        def __init__(self, content):
            self.choices = [type('obj', (object,), {
                'message': type('obj', (object,), {'content': content})()
            })()]

    return MockResponse(response_text)


def _parse_model_json(text: str) -> dict:
    """Attempt to extract JSON from a model output string.

    Strategies:
    - Try direct json.loads
    - Strip fenced code blocks (```json ... ``` or ``` ... ```)
    - Find the first {...} JSON object in the text and parse that

    Raises JSONDecodeError if parsing fails.
    """
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty model response", text, 0)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = text.strip()

    # If the model wrapped the JSON in fenced code blocks (e.g. ```json ... ```),
    # try to extract any fenced blocks and parse their contents.
    fence_blocks = []
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    for m in fence_pattern.finditer(stripped):
        fence_blocks.append(m.group(1).strip())

    for block in fence_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # try to be forgiving: sometimes models put plain text before/after
            # the JSON inside the fence
            start = block.find('{')
            end = block.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = block[start:end+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Try to locate the first '{' and the last '}' and parse that slice.
    # This is more reliable than regex for complex nested JSON with escaped characters.
    start = stripped.find('{')
    end = stripped.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error on candidate (first {{ to last }}): {e}")
            pass

    # If that fails, try to find the first JSON object or array using regex (less reliable for complex JSON).
    # Note: regex cannot fully validate nested JSON but works for simple model outputs.
    obj_pattern = re.compile(r"(\{(?:[^{}]|\{[^}]*\})*\})", re.DOTALL)
    arr_pattern = re.compile(r"(\[(?:[^\[\]]|\[[^\]]*\])*\])", re.DOTALL)

    for pattern in (obj_pattern, arr_pattern):
        for m in pattern.finditer(stripped):
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # try next match
                continue

    # If all attempts fail, raise with a helpful message including a short snippet
    # of the model output to aid debugging.
    snippet = (stripped[:1000] + '...') if len(stripped) > 1000 else stripped
    raise json.JSONDecodeError(f"Could not parse JSON from model output. Snippet: {snippet}", text, 0)


# --- Pydantic Models for Structured Data ---

class ProposedValidation(BaseModel):
    """AI's proposal for how to validate a user-defined rule."""
    explanation: str = Field(..., description="A clear, user-friendly explanation of how the validation will work.")
    validation_type: Literal['regex', 'semantic'] = Field(..., description="The type of validation method the AI has chosen.")
    validator: str = Field(..., description="The generated regex pattern or the detailed semantic prompt for the LLM.")
    example_finding: str = Field(..., description="A direct quote from the provided document text that this validator would find, to show the user it works as expected.")
    clarified_rule: str = Field(..., description="A rewritten, clarified version of the user's rule that clearly explains what constitutes a violation vs. compliance.")
    extracted_examples: List[str] = Field(default_factory=list, description="Examples extracted from the user's rule text (e.g., text in parentheses like '(e.g., ...)'). These should NOT be flagged as violations unless they actually appear in the document.")

class Rule(BaseModel):
    """A user-confirmed, executable validation rule."""
    description: str
    validation_type: Literal['regex', 'semantic']
    validator: str
    clarified_rule: Optional[str] = None  # Rewritten rule for clarity
    extracted_examples: List[str] = Field(default_factory=list)  # Examples to exclude from flagging

class ValidationTemplate(BaseModel):
    """A collection of rules saved by the user."""
    name: str
    rules: List[Rule]

class ValidationResult(BaseModel):
    """Represents a single issue found during validation."""
    page_num: int
    rule_description: str
    violation_type: Optional[str] = None
    finding: str
    analysis: Optional[str] = None
    context: str # A snippet of text around the finding for context

class DocumentChunk(BaseModel):
    """Represents a chunk of text from the document, typically a page."""
    content: str
    page_num: int


# --- Core Logic Functions ---

def decompose_rule_smartreview(rule_text: str) -> List[Rule]:
    """Decompose a plain-text rule into SmartReview validation tasks.
    - No RAG/retrieval. Single-step analysis to choose regex vs semantic.
    - Integrates validation type determination into decomposition.
    - Returns a list of Rule objects suitable for the SmartReview pipeline.
    """
    text = (rule_text or "").strip().lower()
    tasks: List[Rule] = []

    # Heuristics: precise formats → semantic with guardrails; qualitative/intent/tone → semantic
    # NOTE: We no longer hardcode regex patterns. Instead, use propose_validation_from_rule()
    # which has an AI agent that can generate optimal regex patterns with proper lookahead/lookbehind.
    format_keywords = [
        "yyyy", "mm", "dd", "date", "currency", "usd", "$", "eur", "€",
        "email", "e-mail", "phone", "ssn", "id", "identifier", "format",
        "digits", "characters", "alphanumeric", "exactly", "pattern",
    ]
    if any(k in text for k in format_keywords):
        # Default to semantic when we cannot synthesize a robust regex deterministically here.
        # The user can refine this into a regex later in the UI if desired.
        semantic_prompt = (
            f"You must check the text for violations of this rule:\n\"{rule_text}\"\n"
            "- Quote the exact offending text if any.\n"
            "- Provide a concise reason for the violation.\n"
            "- Do NOT flag compliant cases; only clear violations.\n"
            "- Be strict about numeric/word boundaries; avoid partial matches.\n"
            "- If the rule requires decimal precision, do NOT flag numbers that already include a decimal fraction (e.g., '67.3 billion', '1.0 billion').\n"
        )
        tasks.append(Rule(description=rule_text, validation_type='semantic', validator=semantic_prompt))
        return tasks

    # Default: semantic validation
    default_prompt = (
        f"You must check the text for violations of this rule:\n\"{rule_text}\"\n"
        "- Quote the exact offending text if any.\n"
        "- Provide a concise reason for the violation.\n"
        "- Do NOT flag compliant cases.\n"
        "- Be strict about boundaries; avoid partial/embedded matches.\n"
        "- If the rule mentions 'billion' and decimal precision, do NOT flag values with a decimal part (e.g., '67.3 billion', '1.0 billion').\n"
    )
    tasks.append(Rule(description=rule_text, validation_type='semantic', validator=default_prompt))
    return tasks



def extract_text_from_pdf(uploaded_file_bytes: bytes) -> List[DocumentChunk]:
    """Extracts text from each page of an uploaded PDF file."""
    logger.info("Starting PDF text extraction. bytes=%s", _safe_str(len(uploaded_file_bytes) if uploaded_file_bytes is not None else None))
    chunks = []
    try:
        pdf_document = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            logger.debug("Extracted page %s text length=%d", page_num + 1, len(text))
            chunks.append(DocumentChunk(content=text, page_num=page_num + 1))
        logger.info("Completed PDF extraction. pages=%d", len(chunks))
    except Exception as e:
        logger.exception("Error processing PDF file: %s", e)
        st.error(f"Error processing PDF file: {e}")
    return chunks


async def propose_validation_from_rule_v2(rule_text: str, example_text: str = "") -> Optional[ProposedValidation]:
    """
    NEW VERSION: AI agent to interpret a user's rule and propose a validation method.
    This version does NOT use document text in the decomposition step, following the refactored workflow.

    Workflow:
    1. Use decompose_review_mode_prompt to analyze the rule (no document text)
    2. For regex validation: call generate_regex_pattern to get the pattern
    3. For semantic validation: use the clarified rule as the validator

    Returns a ProposedValidation object compatible with the existing UI.
    """
    from src.keyword_code.ai.analyzer import DocumentAnalyzer
    from src.keyword_code.ai.decomposition import decompose_review_mode_prompt, generate_regex_pattern

    logger.info(f"[V2] Proposing validation for rule: '{_safe_str(rule_text, max_len=200)}'")

    # Step 1: Decompose the rule (no document text)
    analyzer = DocumentAnalyzer()

    # If example_text is provided, include it in the rule text for decomposition
    full_rule_text = rule_text
    if example_text:
        full_rule_text = f"{rule_text}\n\nUser-provided example: {example_text}"

    decomposed = await decompose_review_mode_prompt(analyzer, full_rule_text)

    if not decomposed or len(decomposed) == 0:
        logger.error("Decomposition failed or returned empty list")
        return None

    # Take the first sub-prompt (for single rule input, there should be only one)
    sub_prompt = decomposed[0]
    logger.info(f"[V2] Decomposed rule: type={sub_prompt.validation_type}, title='{sub_prompt.title}'")

    # Step 2: Generate validator based on validation type
    validator = ""
    example_finding = "No example available"

    if sub_prompt.validation_type == "regex":
        # Generate regex pattern
        regex_result = await generate_regex_pattern(analyzer, sub_prompt)
        if regex_result:
            validator = regex_result.regex_pattern
            # Use test_matches as example finding
            if regex_result.test_matches:
                example_finding = f"Would match: {regex_result.test_matches[0]}"
            logger.info(f"[V2] Generated regex pattern: {validator[:100]}...")
        else:
            logger.error("Regex generation failed, falling back to semantic")
            # Fallback to semantic if regex generation fails
            sub_prompt.validation_type = "semantic"
            sub_prompt.validation_reasoning = "Regex generation failed, using semantic validation as fallback"

    if sub_prompt.validation_type == "semantic":
        # Use clarified rule as the semantic validator prompt
        validator = (
            f"You must check the text for violations of this rule:\n\"{sub_prompt.sub_prompt}\"\n"
            "- Quote the exact offending text if any.\n"
            "- Provide a concise reason for the violation.\n"
            "- Do NOT flag compliant cases; only clear violations.\n"
            "- Be strict about boundaries; avoid partial/embedded matches.\n"
        )
        # Use violation examples as example finding
        if sub_prompt.violation_examples:
            example_finding = f"Would flag: {sub_prompt.violation_examples[0]}"
        logger.info(f"[V2] Using semantic validation with clarified rule")

    # Step 3: Build ProposedValidation object
    proposal = ProposedValidation(
        explanation=sub_prompt.validation_reasoning,
        validation_type=sub_prompt.validation_type,
        validator=validator,
        example_finding=example_finding,
        clarified_rule=sub_prompt.sub_prompt,
        extracted_examples=sub_prompt.extracted_examples
    )

    logger.info(f"[V2] Proposal created: type={proposal.validation_type}")
    return proposal


async def propose_validation_from_rule(rule_text: str, example_text: str, doc_chunks: List[DocumentChunk]) -> Optional[ProposedValidation]:
    """
    LEGACY VERSION: AI agent to interpret a user's rule and propose a validation method.
    This version includes document text in the API call (original implementation).

    NOTE: This function is kept for backward compatibility with the interactive UI workflow.
    For automated workflows, use propose_validation_from_rule_v2 instead.
    """

    # Combine document chunks for context, but limit the size to avoid excessive token usage
    full_text_context = "\n".join([chunk.content for chunk in doc_chunks])
    # Truncate context to a reasonable length for the API call
    max_context_length = 12000
    if len(full_text_context) > max_context_length:
        full_text_context = full_text_context[:max_context_length] + "\n... [document truncated for brevity]"

    system_prompt = """
    You are an expert AI system that converts a user's plain-text rule into a structured, machine-executable validation. You are primarily working with financial documents.

    Steps:
    1) Extract any examples from the rule text (e.g., text in parentheses like "(e.g., ...)" or similar patterns).
    2) Rewrite the rule to clearly explain the user's intent, removing ambiguity and clarifying what constitutes a violation vs. compliance.
    3) Analyze the rule and any provided example.
    4) Choose validation_type: 'regex' (precise patterns like dates/currency/IDs) or 'semantic' (intent/tone/meaning/context).
    5) Generate validator:
       - If 'regex': produce a robust Python-compatible regex pattern string.
       - If 'semantic': produce a clear, concise evaluation prompt for another AI to check violations.
    6) Find an example from the provided document text that your validator would identify.
    7) Explain the approach in one or two sentences.

    WHEN TO USE REGEX VS SEMANTIC:

    Use REGEX when:
    - The rule checks for a SPECIFIC, WELL-DEFINED pattern that can be exhaustively enumerated
    - Examples: date formats (MM/DD/YYYY), specific numeric patterns (phone numbers, IDs), exact string matches
    - The rule checks a SMALL, FIXED set of values (e.g., checking for 3-5 specific currency names)
    - Pattern matching is deterministic and doesn't require understanding context or meaning

    Use SEMANTIC when:
    - The rule requires understanding MEANING, CONTEXT, or INTENT
    - Examples: word confusion (decease vs decrease, principal vs principle, affect vs effect)
    - The rule involves OPEN-ENDED sets that cannot be exhaustively listed
    - Examples: "all currency references" (there are 180+ world currencies), "proper capitalization of country names"
    - The rule requires CASE SENSITIVITY checks across diverse terms (e.g., "Indian rupee" vs "Indian Rupee")
    - The rule involves CALCULATIONS, COMPARISONS, or LOGICAL REASONING
    - The rule checks for TONE, STYLE, or APPROPRIATENESS

    SPECIFIC GUIDANCE FOR COMMON RULES:
    - Currency case sensitivity (e.g., "Indian rupee" not "Indian Rupee"): Use SEMANTIC
      Reason: There are 180+ currencies; regex cannot enumerate all. Semantic can understand capitalization rules.
    - Word confusion (decease/decrease, principal/principle): Use SEMANTIC
      Reason: Requires understanding context to determine if the word is used correctly.
    - Specific date format (e.g., "Month DD, YYYY"): Use REGEX if checking format only; use SEMANTIC if also validating logical correctness
    - Decimal precision for numbers (e.g., "1.0 billion" not "1 billion"): Use REGEX
      Reason: This is a specific numeric pattern that can be precisely matched.
    - ISO currency codes (USD, EUR, GBP): Use REGEX if checking a small set; use SEMANTIC if checking all possible codes

    CRITICAL REGEX GUIDELINES:
    When creating regex patterns, be aware of word boundaries and decimal numbers:

    - PROBLEM: Word boundary \\b treats '.' as a boundary, so "67.3 billion" is seen as two tokens: "67" and "3"
      If you write \\b\\d+\\s+billion\\b, it will match BOTH "67 billion" AND "3 billion" (from "67.3 billion")

    - SOLUTION: Use negative lookbehind and lookahead to prevent matching decimal parts:
      * (?<!\\.) - No decimal point immediately before the number
      * (?<!\\d\\.) - No digit-dot pattern before (prevents matching "3" in "67.3")
      * (?!\\.\\d+) - No decimal point after the number

    - EXAMPLE: To match integers like "67 billion" but NOT decimals like "67.3 billion":
      Pattern: (?<!\\.)(?<!\\d\\.)\\b(?:\\d{1,3}(?:,\\d{3})*|\\d+)(?!\\.\\d+)\\s+billion\\b

      This pattern:
      ✓ Matches: "67 billion", "1,234 billion", "5 billion"
      ✗ Does NOT match: "67.3 billion", "1.0 billion", "5.5 billion"

    - GENERAL RULE: When matching numbers that should NOT include decimals:
      1. Add (?<!\\.) and (?<!\\d\\.) before your number pattern
      2. Add (?!\\.\\d+) after your number pattern
      3. This prevents matching decimal parts as separate integers

    EXAMPLE EXTRACTION:
    - Look for examples in the rule text, typically in parentheses like "(e.g., ...)" or "(for example, ...)"
    - Extract these examples as a list of strings
    - These examples are for ILLUSTRATION ONLY and should NOT be flagged as violations unless they actually appear in the document
    - Example: "Check currency case (e.g., 'U.S. dollar' not 'U.S. Dollar')" should extract: ["U.S. dollar", "U.S. Dollar"]

    RULE CLARIFICATION:
    - Rewrite the user's rule to clearly state:
      * What the rule is checking for
      * What constitutes a VIOLATION (non-compliant text)
      * What constitutes COMPLIANCE (correct text)
    - Remove ambiguity and make the intent crystal clear
    - Example: "Check currency case (e.g., 'U.S. dollar' not 'U.S. Dollar')" becomes:
      "Currency names must use proper case sensitivity where only the country/region name is capitalized, not the currency unit.
       VIOLATION: 'U.S. Dollar', 'Indian Rupee' (currency unit capitalized).
       COMPLIANT: 'U.S. dollar', 'Indian rupee' (only country capitalized)."

    STRICT OUTPUT REQUIREMENTS:
    - Output MUST be a single JSON object with EXACTLY these keys:
      - "explanation": string
      - "validation_type": "regex" | "semantic"
      - "validator": string
      - "example_finding": string
      - "clarified_rule": string (the rewritten, clarified rule)
      - "extracted_examples": array of strings (examples found in the rule text)
    - Do not include any additional keys, prose, comments, markdown, or code fences.
    - Return ONLY the JSON object.
    """

    user_prompt = f"""
    Here is the document context I am working with:
    --- DOCUMENT TEXT ---
    {full_text_context}
    --- END DOCUMENT TEXT ---

    Please create a validation proposal for the following rule:
    Rule Description: "{rule_text}"
    User-provided Example: "{example_text if example_text else 'No example provided.'}"
    """

    logger.info("Requesting proposal for rule. rule_text=%s example_provided=%s doc_chunks=%d", _safe_str(rule_text), bool(example_text), len(doc_chunks))
    try:
        # Log the full request in DEBUG mode
        logger.debug("=" * 80)
        logger.debug("SMARTREVIEW AI API REQUEST")
        logger.debug("=" * 80)
        logger.debug(f"Endpoint: {DATABRICKS_BASE_URL}")
        logger.debug(f"Model: {DATABRICKS_LLM_MODEL}")
        logger.debug("System Prompt:")
        logger.debug(system_prompt)
        logger.debug("User Prompt:")
        logger.debug(user_prompt)
        logger.debug("=" * 80)

        response = await _chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=DATABRICKS_LLM_MODEL
        )
        raw = response.choices[0].message.content

        # Log the full response in DEBUG mode
        logger.debug("=" * 80)
        logger.debug("SMARTREVIEW AI API RESPONSE")
        logger.debug("=" * 80)
        logger.debug(f"Response Length: {len(raw)} characters")
        logger.debug("Response Content:")
        logger.debug(raw)
        logger.debug("=" * 80)
        try:
            response_json = _parse_model_json(raw)

            # Normalize validator field: the model sometimes returns a dict
            # like {"regex_pattern": "..."} or {"pattern": "..."}.
            # Ensure `validator` is a string as required by ProposedValidation.
            if isinstance(response_json, dict) and 'validator' in response_json:
                val = response_json['validator']
                if isinstance(val, dict):
                    # Common keys the model might use
                    for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                        if key in val:
                            response_json['validator'] = val[key]
                            break
                    else:
                        # Fallback to JSON string representation
                        try:
                            response_json['validator'] = json.dumps(val)
                        except Exception:
                            response_json['validator'] = str(val)

            # Post-parse guard: ensure required keys exist
            required_keys = {"explanation", "validation_type", "validator", "example_finding"}
            if not isinstance(response_json, dict):
                response_json = {}
            missing = required_keys - set(response_json.keys())
            if missing:
                logger.warning("ProposedValidation missing keys %s; attempting structured retry", missing)
                retry_system_prompt = (
                    "Your previous output was missing required keys. Return ONLY a single JSON object with EXACTLY the keys: "
                    "explanation, validation_type, validator, example_finding. No prose, no markdown, no extra keys."
                )
                retry_user_prompt = f"""
                Rule: "{rule_text}"
                Example: "{example_text if example_text else 'No example provided.'}"
                Previous_output (verbatim):
                {raw}

                Document context (truncated):
                {full_text_context[:2000]}
                """
                retry_resp = await _chat_completion_async(
                    messages=[
                        {"role": "system", "content": retry_system_prompt},
                        {"role": "user", "content": retry_user_prompt},
                    ],
                    model=DATABRICKS_LLM_MODEL,
                    temperature=0.0,
                )
                raw2 = retry_resp.choices[0].message.content
                try:
                    response_json = _parse_model_json(raw2)
                    # Re-normalize validator if needed
                    if isinstance(response_json, dict) and 'validator' in response_json:
                        val2 = response_json['validator']
                        if isinstance(val2, dict):
                            for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                                if key in val2:
                                    response_json['validator'] = val2[key]
                                    break
                            else:
                                try:
                                    response_json['validator'] = json.dumps(val2)
                                except Exception:
                                    response_json['validator'] = str(val2)
                except Exception:
                    logger.exception("Retry parse failed; will attempt fallback synthesis if possible")

                # Check again
                if not isinstance(response_json, dict):
                    response_json = {}
                missing = required_keys - set(response_json.keys())
                if missing:
                    if missing == {"explanation"}:
                        # Synthesize minimal explanation and proceed
                        vtype = response_json.get("validation_type", "semantic")
                        response_json["explanation"] = (
                            f"This {vtype} validator checks the document for: {rule_text.strip()[:200]}."
                        )
                    else:
                        raise ValueError(f"Missing required keys after retry: {missing}")

            proposal = ProposedValidation(**response_json)
        except Exception as e:
            logger.exception("Failed to parse JSON from model output: %s", e)
            logger.debug("Raw model output: %s", _safe_str(raw, max_len=2000))
            st.error("AI returned an unparsable response. Check logs for the raw output.")
            return None
        logger.info("Generated proposal type=%s example_finding=%s", proposal.validation_type, _safe_str(proposal.example_finding, max_len=200))
        return proposal
    except Exception as e:
        logger.exception("An AI communication error occurred while proposing validation: %s", e)
        st.error(f"An AI communication error occurred: {e}")
        return None


async def refine_validation_from_chat(chat_history: List[dict], original_proposal: ProposedValidation, doc_chunks: List[DocumentChunk]) -> Optional[ProposedValidation]:
    """AI agent that refines a proposal based on user chat feedback."""
    full_text_context = "\n".join([chunk.content for chunk in doc_chunks])
    max_context_length = 12000
    if len(full_text_context) > max_context_length:
        full_text_context = full_text_context[:max_context_length] + "\n... [document truncated for brevity]"

    system_prompt = """
    You are an expert AI system that refines validation rules based on user feedback.
    The user was not satisfied with your previous proposal and will provide feedback.
    Your task is to generate a *new* `ProposedValidation` JSON object that incorporates the user's feedback.
    Carefully read the chat history to understand what the user wants to change. Then, generate a completely new proposal that addresses their concerns.

    CRITICAL REGEX GUIDELINES:
    When creating regex patterns, be aware of word boundaries and decimal numbers:

    - PROBLEM: Word boundary \\b treats '.' as a boundary, so "67.3 billion" is seen as two tokens: "67" and "3"
      If you write \\b\\d+\\s+billion\\b, it will match BOTH "67 billion" AND "3 billion" (from "67.3 billion")

    - SOLUTION: Use negative lookbehind and lookahead to prevent matching decimal parts:
      * (?<!\\.) - No decimal point immediately before the number
      * (?<!\\d\\.) - No digit-dot pattern before (prevents matching "3" in "67.3")
      * (?!\\.\\d+) - No decimal point after the number

    - EXAMPLE: To match integers like "67 billion" but NOT decimals like "67.3 billion":
      Pattern: (?<!\\.)(?<!\\d\\.)\\b(?:\\d{1,3}(?:,\\d{3})*|\\d+)(?!\\.\\d+)\\s+billion\\b

      This pattern:
      ✓ Matches: "67 billion", "1,234 billion", "5 billion"
      ✗ Does NOT match: "67.3 billion", "1.0 billion", "5.5 billion"

    - GENERAL RULE: When matching numbers that should NOT include decimals:
      1. Add (?<!\\.) and (?<!\\d\\.) before your number pattern
      2. Add (?!\\.\\d+) after your number pattern
      3. This prevents matching decimal parts as separate integers
    """

    chat_transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    user_prompt = f"""
    --- DOCUMENT TEXT ---
    {full_text_context}
    --- END DOCUMENT TEXT ---

    This was my original proposal that the user wants to refine:
    --- ORIGINAL PROPOSAL ---
    {original_proposal.model_dump_json(indent=2)}
    --- END ORIGINAL PROPOSAL ---

    Here is the conversation so far:
    --- CHAT HISTORY ---
    {chat_transcript}
    --- END CHAT HISTORY ---

    Based on the user's feedback in the chat, please generate a new and improved validation proposal in the required JSON format.
    """

    logger.info("Refining validation from chat. chat_length=%d original_type=%s", len(chat_history), original_proposal.validation_type if original_proposal else None)
    try:
        logger.debug("Calling LLM at %s model=%s", DATABRICKS_BASE_URL, DATABRICKS_LLM_MODEL)
        response = await _chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=DATABRICKS_LLM_MODEL
        )
        raw = response.choices[0].message.content
        logger.debug("Raw AI refinement response length=%d", len(_safe_str(raw)))
        try:
            response_json = _parse_model_json(raw)

            # Normalize validator field similar to propose_validation_from_rule
            if isinstance(response_json, dict) and 'validator' in response_json:
                val = response_json['validator']
                if isinstance(val, dict):
                    for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                        if key in val:
                            response_json['validator'] = val[key]
                            break
                    else:
                        try:
                            response_json['validator'] = json.dumps(val)
                        except Exception:
                            response_json['validator'] = str(val)

            proposal = ProposedValidation(**response_json)
        except Exception as e:
            # Parsing failed; attempt a forgiving fallback before giving up.
            logger.exception("Failed to parse JSON from refinement output: %s", e)
            logger.debug("Raw model output (snippet): %s", _safe_str(raw, max_len=2000))

            # Try to be forgiving: remove common leading phrases and code fences then retry
            cleaned = raw
            # Remove leading assistant/commentary lines like 'Here's a new proposal:'
            cleaned = re.sub(r"^\s*[A-Za-z\s\'\":,.-]{0,80}:?\s*\n", "", cleaned)
            # Remove any surrounding triple-backtick fences
            if cleaned.strip().startswith("```") and cleaned.strip().endswith("```"):
                inner_lines = cleaned.strip().splitlines()[1:-1]
                cleaned = "\n".join(inner_lines)

            try:
                response_json = _parse_model_json(cleaned)
                # Normalize as below
                if isinstance(response_json, dict) and 'validator' in response_json:
                    val = response_json['validator']
                    if isinstance(val, dict):
                        for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                            if key in val:
                                response_json['validator'] = val[key]
                                break
                        else:
                            try:
                                response_json['validator'] = json.dumps(val)
                            except Exception:
                                response_json['validator'] = str(val)

                proposal = ProposedValidation(**response_json)
                logger.info("Refined proposal parsed after fallback. type=%s", proposal.validation_type)
                return proposal
            except Exception:
                logger.exception("Fallback parsing also failed for refinement output.")
                st.error("AI returned an unparsable refinement. Check logs for the raw output.")
                return None
        logger.info("Refined proposal generated type=%s", proposal.validation_type)
        return proposal
    except Exception as e:
        logger.exception("An AI communication error occurred during refinement: %s", e)
        st.error(f"An AI communication error occurred during refinement: {e}")
        return None



async def execute_validation_template(template: ValidationTemplate, doc_chunks: List[DocumentChunk]) -> List[ValidationResult]:
    """Runs all rules in a template against a document and collects the results.
    Uses the new Orchestrator (parallel multi-tool + evaluator). Falls back to legacy execution on error.
    """
    logger.info(
        "Executing validation template '%s' with %d rules on %d document chunks.",
        template.name if template else None,
        len(template.rules) if template else 0,
        len(doc_chunks),
    )

    # Log rule types for debugging
    if template and template.rules:
        rule_types = {}
        for rule in template.rules:
            vtype = getattr(rule, "validation_type", "unknown")
            rule_types[vtype] = rule_types.get(vtype, 0) + 1
        logger.info(f"Rule breakdown: {dict(rule_types)}")

    try:
        from src.keyword_code.agents.review_orchestrator import orchestrate_review
        ranked = await orchestrate_review(template, doc_chunks)
        logger.info(f"Orchestrated review completed. Found {len(ranked)} ranked findings.")

        results: List[ValidationResult] = []
        for r in ranked:
            results.append(
                ValidationResult(
                    page_num=r.page_num,
                    rule_description=r.rule_description,
                    violation_type=r.violation_type,
                    finding=r.finding,
                    analysis=r.analysis,
                    context=r.context,
                )
            )
        logger.info(f"Validation template execution completed successfully. Returning {len(results)} results.")
        return results
    except Exception as e:
        logger.exception("Orchestrated review failed, falling back to legacy execution: %s", e)
        all_results: List[ValidationResult] = []
        tasks = []
        for rule in template.rules:
            for chunk in doc_chunks:
                tasks.append(run_rule_on_chunk(rule, chunk))
        list_of_results_per_task = await asyncio.gather(*tasks)
        for result_list in list_of_results_per_task:
            all_results.extend(result_list)
        logger.info(f"Legacy execution completed. Returning {len(all_results)} results.")
        return all_results


async def run_rule_on_chunk(rule: Rule, chunk: DocumentChunk) -> List[ValidationResult]:
    """Helper function to run a single rule on a single chunk."""
    results = []
    logger.debug("Running rule on chunk. rule='%s' type=%s page=%d", _safe_str(rule.description, max_len=200), rule.validation_type, chunk.page_num)
    if rule.validation_type == 'regex':
        try:
            matches = re.finditer(rule.validator, chunk.content)
            for match in matches:
                # Create a context snippet around the finding
                start = max(0, match.start() - 50)
                end = min(len(chunk.content), match.end() + 50)
                context_snippet = f"...{chunk.content[start:end]}..."
                logger.debug("Regex match on page %d: %s", chunk.page_num, _safe_str(match.group(0), max_len=200))
                results.append(ValidationResult(
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    violation_type='regex',
                    finding=f"Found violation: '{match.group(0)}'",
                    analysis=f"Regex matched the pattern for this rule, indicating non-compliance: {rule.description}",
                    context=context_snippet
                ))
        except re.error as e:
            logger.exception("Invalid regex pattern for rule '%s': %s", rule.description, e)
            st.warning(f"Invalid regex pattern for rule '{rule.description}': {e}")

    elif rule.validation_type == 'semantic':
        system_prompt = """
        You are an AI document validation assistant. You will be given a chunk of text and a rule.
        Your task is to check if the text violates the rule.

        CRITICAL - ONLY FLAG TRUE VIOLATIONS:
        - Do NOT flag text that is COMPLIANT with the rule
        - Do NOT flag text where there is NO CONFUSION or error
        - Do NOT flag text that MEETS the requirements
        - Do NOT flag text that is NOT RELEVANT to the rule (e.g., proper acronyms like 'U.S. GAAP' when checking capitalization)
        - Do NOT flag numeric expressions that already satisfy the rule (e.g., '67.3 billion' is compliant if rule requires decimal precision)

        RESPONSE FORMAT:
        - If you find a TRUE VIOLATION, respond with ONLY the exact string of text from the document that violates the rule. Do NOT include explanations, commentary, or corrections. Extract and return ONLY the verbatim erroneous text.
        - If there are NO VIOLATIONS (including compliant cases, correct usage, or irrelevant matches), respond *only* with the text "No violation found.".

        Do not be conversational. Provide only the exact erroneous text or "No violation found.".
        """
        prompt = f"""
        --- RULE ---
        {rule.validator}

        --- TEXT TO VALIDATE ---
        {chunk.content}
        """
        try:
            logger.debug("Calling semantic AI for page %d rule=%s; base_url=%s model=%s", chunk.page_num, _safe_str(rule.description, max_len=200), DATABRICKS_BASE_URL, DATABRICKS_LLM_MODEL)
            response = await _chat_completion_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=DATABRICKS_LLM_MODEL,
                temperature=0.1,
            )
            message_content = response.choices[0].message.content
            logger.debug("Semantic AI response for page %d length=%d", chunk.page_num, len(_safe_str(message_content)))
            if message_content.lower().strip() != "no violation found.":
                logger.info("Semantic violation found on page %d: %s", chunk.page_num, _safe_str(message_content, max_len=300))
                results.append(ValidationResult(
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    violation_type='semantic',
                    finding=message_content,
                    analysis=f"Semantic evaluation reason: {message_content}",
                    context=f"Semantic check on page {chunk.page_num}."
                ))
        except Exception as e:
            logger.exception("API call failed for semantic rule on page %d: %s", chunk.page_num, e)
            st.warning(f"API call failed for semantic rule on page {chunk.page_num}: {e}")

    return results

# --- UI Rendering Functions ---


def render_validation_view():
    """Renders the main UI for validating documents against saved templates."""
    logger.debug("Rendering validation view")
    # Top-level title and short description
    st.title("SmartReview — Document Validation")
    st.caption("Upload a PDF, pick a template, and run fast AI- or regex-based checks.")

    # Minimal sidebar: navigation and a small help/credits area
    with st.sidebar:
        st.markdown("### Navigation")
        if st.button("Create Template"):
            st.session_state.app_mode = 'rule_definition'
            st.rerun()
        st.markdown("---")
        st.markdown("#### Tips")
        st.write("Keep templates focused: 3-8 rules works well.")
        st.markdown("---")
        if st.checkbox("Show debug logs", value=False, key="show_logs"):
            st.write("Logs are printed to the server console.")

    # If there are no templates, show a friendly prompt
    if not st.session_state.templates:
        st.info("No validation templates yet. Click 'Create Template' in the sidebar to get started.")
        return

    # Main validation area laid out in two columns: inputs (left) and results (right)
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Run Validation")
        template_names = list(st.session_state.templates.keys())
        selected_template_name = st.selectbox("Select a Template", options=template_names)
        uploaded_file = st.file_uploader("Upload PDF to validate", type="pdf", key="validation_uploader")
        run_disabled = (not selected_template_name) or (not uploaded_file)
        if st.button("Run Validation", disabled=run_disabled, type="primary"):
            with st.spinner("Analyzing document... this may take a moment"):
                template = st.session_state.templates[selected_template_name]
                pdf_bytes = uploaded_file.getvalue()
                doc_chunks = extract_text_from_pdf(pdf_bytes)
                if doc_chunks:
                    results = asyncio.run(execute_validation_template(template, doc_chunks))
                    st.session_state.validation_results = results
                    st.success("Validation complete.")
                else:
                    st.error("Could not extract text from the PDF.")

    # Results column: shows run status and collapsible results per finding
    with right_col:
        st.subheader("Validation Report")
        if st.session_state.validation_results is None:
            st.info("No validation run yet. Results will appear here.")
        elif not st.session_state.validation_results:
            st.success("No issues were found for the selected template.")
        else:
            for result in st.session_state.validation_results:
                header = f"[{(result.violation_type or 'violation').upper()}] Page {result.page_num} — {result.rule_description}"
                with st.expander(header, expanded=False):
                    if getattr(result, 'analysis', None):
                        st.markdown(result.analysis)
                    st.error(result.finding)
                    st.caption("Context")
                    st.markdown(f"> {result.context.replace('...', ' ... ')}")


def render_rule_definition_view():
    """Renders the UI for the interactive rule creation process."""
    logger.debug("Rendering rule definition view")
    # Rule definition screen: upload a sample doc, then create rules with AI assistance
    st.header("Rule Template Definition")

    # Compact sidebar control to go back to validation
    with st.sidebar:
        if st.button("Back to Validation"):
            st.session_state.app_mode = 'validation'
            st.rerun()
        st.markdown("---")

    # Step 1: Upload a sample document if none exists
    if not st.session_state.definition_pdf_bytes:
        st.info("Upload a sample PDF to base your template on.")
        uploaded_file = st.file_uploader("Upload sample PDF", type="pdf", key="definition_uploader")
        if uploaded_file:
            st.session_state.definition_pdf_bytes = uploaded_file.getvalue()
            st.session_state.definition_doc_chunks = extract_text_from_pdf(st.session_state.definition_pdf_bytes)
            st.rerun()

    if not st.session_state.definition_pdf_bytes:
        return

    st.success("Sample document loaded.")

    # Layout: left column shows accepted rules, right column is for creating/confirming a new rule
    left, right = st.columns([1, 1.2])

    with right:
        st.subheader("Accepted Rules")
        if not st.session_state.current_rules:
            st.caption("No rules added yet.")
        else:
            for i, rule in enumerate(st.session_state.current_rules):
                with st.expander(f"{i+1}. {rule.description}"):
                    st.code(f"Type: {rule.validation_type}\nValidator: {rule.validator}", language="text")

    with left:
        st.subheader("Create a New Rule")

        # If AI has proposed a validation, show a compact confirmation card
        if st.session_state.proposed_validation:
            proposal = st.session_state.proposed_validation
            st.markdown("**AI Proposal**")
            st.markdown(f"**Explanation:** {proposal.explanation}")
            st.markdown(f"**Type:** `{proposal.validation_type}`")
            st.code(proposal.validator, language='text')
            st.markdown(f"**Example finding:** {proposal.example_finding}")

            a_col, b_col = st.columns([1, 1])
            with a_col:
                if st.button("Accept Proposal"):
                    accepted_rule = Rule(
                        description=st.session_state.current_rule_text,
                        validation_type=proposal.validation_type,
                        validator=proposal.validator,
                        clarified_rule=getattr(proposal, 'clarified_rule', None),
                        extracted_examples=getattr(proposal, 'extracted_examples', []),
                    )
                    st.session_state.current_rules.append(accepted_rule)
                    st.session_state.proposed_validation = None
                    st.session_state.current_rule_text = ""
                    st.session_state.current_rule_example = ""
                    st.session_state.refinement_chat_history = []
                    st.rerun()
            with b_col:
                if st.button("Refine"):
                    st.session_state.is_refining = True
                    st.session_state.refinement_chat_history.append({"role": "assistant", "content": "I've created a proposal. How would you like to refine it?"})
                    st.rerun()

        # Refinement chat
        if st.session_state.is_refining:
            with st.expander("Refine with AI", expanded=True):
                for message in st.session_state.refinement_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("Tell the AI how to change the rule..."):
                    st.session_state.refinement_chat_history.append({"role": "user", "content": prompt})
                    with st.spinner("AI is thinking..."):
                        new_proposal = asyncio.run(refine_validation_from_chat(
                            st.session_state.refinement_chat_history,
                            st.session_state.proposed_validation,
                            st.session_state.definition_doc_chunks,
                        ))
                        if new_proposal:
                            st.session_state.proposed_validation = new_proposal
                            st.session_state.refinement_chat_history.append({"role": "assistant", "content": "Based on your feedback, here is my new proposal."})
                        else:
                            st.session_state.refinement_chat_history.append({"role": "assistant", "content": "I had trouble processing that refinement. Could you rephrase?"})
                    st.rerun()

        # Main form for new rules (shown when not confirming/refining)
        if not st.session_state.proposed_validation and not st.session_state.is_refining:
            rule_text = st.text_input("Rule (plain text)", placeholder="e.g., Dates must be YYYY-MM-DD", key="rule_text_input")
            example_text = st.text_input("Example from document (optional)", placeholder="e.g., 2025-10-27", key="rule_example_input")
            if st.button("Get AI Suggestion", disabled=not rule_text):
                st.session_state.current_rule_text = rule_text
                st.session_state.current_rule_example = example_text
                with st.spinner("AI is analyzing your rule..."):
                    proposal = asyncio.run(propose_validation_from_rule(rule_text, example_text, st.session_state.definition_doc_chunks))
                    if proposal:
                        st.session_state.proposed_validation = proposal
                st.rerun()

        # Save template area
        st.markdown("---")
        template_name = st.text_input("Template name", placeholder="e.g., Financial Report Standard")
        if st.button("Save Template", disabled=(not st.session_state.current_rules or not template_name)):
            normalized_rules = []
            for r in st.session_state.current_rules:
                try:
                    if hasattr(r, 'model_dump'):
                        normalized_rules.append(Rule(**r.model_dump()))
                    elif isinstance(r, dict):
                        normalized_rules.append(Rule(**r))
                    else:
                        normalized_rules.append(Rule(description=getattr(r, 'description', str(r)), validation_type=getattr(r, 'validation_type', 'semantic'), validator=getattr(r, 'validator', '')))
                except Exception:
                    # skip invalid rule
                    pass
            new_template = ValidationTemplate(name=template_name, rules=normalized_rules)
            st.session_state.templates[template_name] = new_template
            st.session_state.definition_pdf_bytes = None
            st.session_state.definition_doc_chunks = []
            st.session_state.current_rules = []
            st.success(f"Template '{template_name}' saved!")
            st.balloons()
            st.session_state.app_mode = 'validation'
            st.rerun()


# --- Session State Initialization ---

def initialize_smartreview_session_state():
    """Initialize SmartReview-specific session state variables."""
    logger.debug("Initializing SmartReview session state")

    # SmartReview-specific state defaults
    smartreview_defaults = {
        'app_mode': 'validation',  # 'validation' or 'rule_definition'
        'templates': {},
        'validation_results': None,
        # State for rule definition view
        'definition_pdf_bytes': None,
        'definition_doc_chunks': [],
        'current_rules': [],
        'current_rule_text': "",
        'current_rule_example': "",
        'proposed_validation': None,
        'is_refining': False,
        'refinement_chat_history': []
    }

    for key, value in smartreview_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
