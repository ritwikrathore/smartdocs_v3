"""
Document analyzer functionality.
"""

import json
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..config import logger, ANALYSIS_MODEL_NAME, USE_DATABRICKS_LLM, LLM_MAX_RETRIES

# Import Databricks LLM client
from .databricks_llm import get_databricks_llm

# Import interaction logger
from ..utils.interaction_logger import log_llm_interaction


_thread_local = threading.local()


def _sanitize_unescaped_control_chars(json_str: str) -> Tuple[str, bool]:
    """Escape control characters that appear unescaped within JSON strings."""

    sanitized_chars: List[str] = []
    in_string = False
    escape_next = False
    sanitized = False

    for ch in json_str:
        if in_string:
            if escape_next:
                sanitized_chars.append(ch)
                escape_next = False
                continue

            if ch == "\\":
                sanitized_chars.append(ch)
                escape_next = True
                continue

            if ch == '"':
                sanitized_chars.append(ch)
                in_string = False
                continue

            code_point = ord(ch)
            if ch == "\n":
                sanitized_chars.append("\\n")
                sanitized = True
                continue
            if ch == "\r":
                sanitized_chars.append("\\r")
                sanitized = True
                continue
            if ch == "\t":
                sanitized_chars.append("\\t")
                sanitized = True
                continue
            if code_point < 0x20:
                sanitized_chars.append(f"\\u{code_point:04x}")
                sanitized = True
                continue

            sanitized_chars.append(ch)
            continue

        sanitized_chars.append(ch)
        if ch == '"':
            in_string = True
            escape_next = False

    return "".join(sanitized_chars), sanitized


class DocumentAnalyzer:
    def __init__(self):
        # Initialize Databricks LLM client
        self.databricks_client = get_databricks_llm() if USE_DATABRICKS_LLM else None

        # Log client initialization status
        if self.databricks_client:
            logger.info("DocumentAnalyzer initialized with Databricks LLM client")
        else:
            logger.error("DocumentAnalyzer failed to initialize Databricks LLM client")

    def _ensure_client(self, model_name: str):
        # Return client configuration
        if self.databricks_client:
            return {"client": self.databricks_client, "model_name": model_name, "type": "databricks"}
        else:
            raise ValueError("No LLM client available - Databricks LLM client failed to initialize")

    async def _get_completion(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        *,
        max_attempts: Optional[int] = None,
    ) -> str:
        """Helper method to get completion from Databricks LLM with retry support."""

        attempts = max_attempts or LLM_MAX_RETRIES

        for attempt in range(1, attempts + 1):
            try:
                # Ensure Databricks client is available
                if not self.databricks_client:
                    raise ValueError("Databricks LLM client not initialized")

                logger.info("Sending request to Databricks LLM model")
                # Databricks client handles message formatting internally
                response_content = await self.databricks_client.get_completion_async(messages, max_tokens=8192)

                if not response_content:
                    raise ValueError("Failed to get response from Databricks LLM")

                logger.info("Received response from Databricks LLM model")

                # Log the LLM interaction
                interaction_type = "analysis"
                if model_name == "databricks-gpt-oss-120b":
                    if any("decompose" in msg.get("content", "").lower() for msg in messages if msg.get("role") == "system"):
                        interaction_type = "decomposition"
                    elif any("chat" in msg.get("content", "").lower() for msg in messages if msg.get("role") == "system"):
                        interaction_type = "chat"

                log_llm_interaction(messages, response_content, interaction_type)

                return response_content

            except Exception as e:
                logger.error(
                    "Error getting completion from Databricks LLM (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    e,
                    exc_info=attempt == attempts,
                )
                if attempt == attempts:
                    raise
                logger.info(
                    "Retrying Databricks LLM call (attempt %d/%d)",
                    attempt + 1,
                    attempts,
                )

    async def _get_json_with_retries(
        self,
        *,
        messages: List[Dict[str, str]],
        model_name: str,
        context: str,
        extractor: Optional[Callable[[str], Any]] = None,
        max_attempts: Optional[int] = None,
        return_raw: bool = False,
    ) -> Any:
        """Call the LLM and parse the response as JSON with retry logic."""

        attempts = max_attempts or LLM_MAX_RETRIES
        extractor = extractor or self._extract_json_from_response
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                response_content = await self._get_completion(
                    messages,
                    model_name,
                    max_attempts=1,
                )
            except Exception as err:
                last_error = err
                logger.error(
                    "LLM call failed during %s (attempt %d/%d): %s",
                    context,
                    attempt,
                    attempts,
                    err,
                    exc_info=attempt == attempts,
                )
                if attempt == attempts:
                    raise
                logger.info(
                    "Retrying %s after call failure (attempt %d/%d)",
                    context,
                    attempt + 1,
                    attempts,
                )
                continue

            logger.debug(
                "Raw LLM response for %s (attempt %d/%d): %s",
                context,
                attempt,
                attempts,
                response_content[:500] if response_content else "",
            )

            try:
                parsed = extractor(response_content)
                if return_raw:
                    return parsed, response_content
                return parsed
            except json.JSONDecodeError as json_err:
                last_error = json_err
                logger.error(
                    "Failed to parse LLM response as JSON during %s (attempt %d/%d): %s",
                    context,
                    attempt,
                    attempts,
                    json_err,
                    exc_info=attempt == attempts,
                )
                if attempt == attempts:
                    raise
                logger.info(
                    "Retrying %s due to JSON parsing error (attempt %d/%d)",
                    context,
                    attempt + 1,
                    attempts,
                )

        if last_error:
            raise last_error
        raise RuntimeError(f"LLM retries exhausted for {context}")

    @staticmethod
    def _extract_json_from_response(response_content: str) -> Any:
        """Extract the first JSON object from a model response."""

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
                logger.warning("Used brace slicing heuristic to extract JSON from model response.")
            else:
                raise json.JSONDecodeError("Could not locate JSON payload in response.", cleaned_response, 0)

        sanitized_json_str, sanitized = _sanitize_unescaped_control_chars(json_str)
        if sanitized:
            logger.debug("Escaped control characters in JSON payload before parsing.")

        return json.loads(sanitized_json_str)

    @property
    def output_schema_analysis(self) -> dict:
        """Defines the expected JSON structure for document analysis."""
        # Keep this schema definition as it's used in the analysis prompt
        return {
            "title": "Concise Title for the Analysis Section based on the specific sub-prompt",
            "analysis_sections": {
                "descriptive_section_name_1": {
                    "Analysis": "Detailed analysis text for this section...",
                    "Supporting_Phrases": [
                        "Exact quote 1 from the document text...",
                        "Exact quote 2, potentially longer...",
                    ],
                    "Context": "Optional context about this section (e.g., source sections)",
                },
                # Add more sections as identified by the AI FOR THIS SUB-PROMPT
            },
        }

    async def analyze_document_with_all_contexts(
        self,
        filename: str,
        main_prompt: str,
        sub_prompts_with_contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes all sub-prompts with their relevant contexts in a single LLM call.

        Args:
            filename: Name of the document being analyzed
            main_prompt: The original user prompt
            sub_prompts_with_contexts: List of dictionaries, each containing:
                - 'title': Title of the sub-prompt
                - 'sub_prompt': The sub-prompt text
                - 'relevant_chunks': List of relevant chunks for this sub-prompt

        Returns:
            List of dictionaries, each containing the analysis for a sub-prompt
        """
        try:
            if not sub_prompts_with_contexts:
                logger.warning(f"No sub-prompts with contexts provided for {filename}")
                return []

            # Format all sub-prompts and their contexts
            formatted_sub_prompts = []
            for i, item in enumerate(sub_prompts_with_contexts):
                sub_prompt = item.get('sub_prompt', '')
                title = item.get('title', f'Sub-prompt {i+1}')
                relevant_chunks = item.get('relevant_chunks', [])

                if not relevant_chunks:
                    logger.warning(f"No relevant chunks for sub-prompt '{title}' in {filename}")
                    formatted_context = "No relevant text found for this sub-prompt."
                else:
                    # Format relevant chunks for this sub-prompt
                    # Note: We provide page numbers but not chunk IDs to encourage natural section references
                    formatted_context = "\n\n---\n\n".join([
                        f"Page: {chunk.get('page_num', -1) + 1}\n"
                        f"TEXT: {chunk.get('text', '')}"
                        for chunk in relevant_chunks
                    ])

                formatted_sub_prompts.append({
                    "index": i + 1,
                    "title": title,
                    "sub_prompt": sub_prompt,
                    "context": formatted_context
                })

            # Create the system prompt for the comprehensive analysis
            system_prompt = """You are an intelligent document analyzer specializing in legal and financial documents. You will be given a main prompt, multiple sub-prompts derived from it, and relevant document excerpts for each sub-prompt. Your task is to analyze each sub-prompt using its specific context and provide a structured response.

### IMPORTANT Core Instructions:
1. **Analyze Each Sub-prompt Separately:** For each sub-prompt, provide a detailed analysis using ONLY the context provided for that specific sub-prompt.
2. **Structured Response:** Your response must follow the JSON structure specified below, with an analysis for each sub-prompt.
3. **Direct Answers:** For each sub-prompt, provide a comprehensive analysis that directly answers the question. If the question is not answerable, clearly state this in the analysis.
4. **Exact Supporting Quotes:** For each sub-prompt, include direct, verbatim quotes from its context that support your analysis.
5. **No Cross-Referencing:** Do not use context from one sub-prompt to answer another sub-prompt, even if they seem related.
6. **No Information Found:** If the context for a sub-prompt does not contain information to answer it, clearly state this in the analysis.
7. **Natural Section References:** When referring to where information is found in the document, use natural language references to document sections (e.g., "Section 9 of the Loan Agreement", "Definitions Section", "Article 5", "Schedule A") based on the content of the text excerpts. Do NOT mention chunk IDs or technical identifiers.

### IMPORTANT Formatting Guidelines:
*YOU MUST ALWAYS USE MARKDOWN* in your analysis_summary to improve readability. Apply the following options **only inside the analysis_summary field**; other JSON fields such as supporting_quotes must remain plain strings without Markdown tables or list scaffolding:
- **Bullet points** (using *, -, or +) for concise lists of up to 10 items (e.g., schedule of dates, list of violations, enumerated findings)
- **Numbered lists** (using 1., 2., etc.) for sequential or ordered information such as repayment schedules, payment terms, or fee structures and tables
- **Bold text** (using **text**) to emphasize important information like dates, percentages, amounts, facts, or key definitions
- **Italic text** (using *text*) for subtle emphasis or document references
- **Tables** (Markdown table syntax) only when the provided excerpts contain inherently tabular information that benefits from a structured layout. Keep tables concise (no more than 12 columns or 20 rows), reuse the source column labels when available, never fabricate tables when list formats suffice, and do not place tables in supporting_quotes or other JSON properties.

DO NOT use any of the following:
- Headers (#, ##, etc.) - the UI provides its own section headers
- Code blocks (```) - not applicable for document analysis
- Tables unless the relevant context clearly contains tabular data as described above
- Other complex Markdown elements

### JSON Output Schema:
```json
{
  "analyses": [
    {
      "sub_prompt_index": 1,
      "sub_prompt_title": "Title of the first sub-prompt",
      "sub_prompt_analyzed": "The exact first sub-prompt being analyzed",
      "analysis_summary": "Detailed analysis directly answering the **first sub-prompt**...",
      "supporting_quotes": [
        "Exact quote 1 from the document text for the first sub-prompt...",
        "Exact quote 2, potentially longer..."
      ],
      "analysis_context": "Natural language description of where the information was found (e.g., 'Section 9 of the Loan Agreement', 'Definitions Section', 'Article 5 - Payment Terms')"
    },
    {
      "sub_prompt_index": 2,
      "sub_prompt_title": "Title of the second sub-prompt",
      "sub_prompt_analyzed": "The exact second sub-prompt being analyzed",
      "analysis_summary": "Detailed analysis directly answering the **second sub-prompt**...",
      "supporting_quotes": [
        "Exact quote 1 from the document text for the second sub-prompt...",
        "Exact quote 2, potentially longer..."
      ],
      "analysis_context": "Natural language description of where the information was found (e.g., 'Section 9 of the Loan Agreement', 'Definitions Section', 'Article 5 - Payment Terms')"
    }
    // Additional analyses for each sub-prompt...
  ]
}
```

Your entire response MUST be a single JSON object following this schema and Formatting Guidelines. Do not include any introductory text, explanations, or markdown formatting outside the JSON structure.
"""

            # Create the human prompt with all sub-prompts and their contexts
            human_prompt = f"""Please analyze the following document based on the main prompt and its derived sub-prompts, using the relevant excerpts provided for each sub-prompt.

Document Name:
{filename}

Main Prompt:
{main_prompt}

Sub-prompts and their contexts:
"""

            # Add each sub-prompt and its context
            for item in formatted_sub_prompts:
                human_prompt += f"""
--- SUB-PROMPT {item['index']} ---
Title: {item['title']}
Sub-prompt: {item['sub_prompt']}

Relevant Document Excerpts for Sub-prompt {item['index']}:
{item['context']}

"""

            human_prompt += """
Generate a structured analysis for EACH sub-prompt, strictly following the JSON schema and Formatting Guidelines provided in the system instructions. Ensure each analysis only addresses its specific sub-prompt and uses only the context provided for that sub-prompt.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]

            logger.info(f"Sending comprehensive analysis request for {len(formatted_sub_prompts)} sub-prompts in {filename} to AI")
            try:
                parsed_json = await self._get_json_with_retries(
                    messages=messages,
                    model_name=ANALYSIS_MODEL_NAME,
                    context=f"comprehensive analysis for {filename}",
                )
                logger.info(f"Received comprehensive AI analysis response for {filename}")
            except json.JSONDecodeError as json_err:
                logger.error(
                    "Failed to parse comprehensive AI analysis response as JSON after %d attempt(s): %s",
                    LLM_MAX_RETRIES,
                    json_err,
                )
                return self._create_fallback_analyses(sub_prompts_with_contexts)
            except Exception as e:
                logger.error(
                    "Error retrieving comprehensive AI analysis response: %s",
                    e,
                    exc_info=True,
                )
                return self._create_fallback_analyses(sub_prompts_with_contexts)

            # Validate the response structure
            if not isinstance(parsed_json, dict) or "analyses" not in parsed_json:
                logger.error("Invalid response format: 'analyses' key missing")
                return self._create_fallback_analyses(sub_prompts_with_contexts)

            analyses = parsed_json["analyses"]
            if not isinstance(analyses, list):
                logger.error("Invalid response format: 'analyses' is not a list")
                return self._create_fallback_analyses(sub_prompts_with_contexts)

            # Process each analysis
            results = []
            for analysis in analyses:
                # Basic validation
                if not isinstance(analysis, dict):
                    logger.warning("Skipping invalid analysis entry (not a dict)")
                    continue

                # Extract the analysis data
                sub_prompt_index = analysis.get("sub_prompt_index")
                if sub_prompt_index is None:
                    logger.warning("Analysis missing sub_prompt_index, using position in list")
                    sub_prompt_index = len(results) + 1

                # Find the original sub-prompt data
                original_index = sub_prompt_index - 1
                if 0 <= original_index < len(sub_prompts_with_contexts):
                    original_sub_prompt = sub_prompts_with_contexts[original_index].get('sub_prompt', '')
                    original_title = sub_prompts_with_contexts[original_index].get('title', '')
                else:
                    # This is expected when retrying a single sub-prompt from a multi-prompt analysis
                    # The LLM may return the original index (e.g., 2) but we only have 1 item in the list
                    logger.debug(f"Sub-prompt index {sub_prompt_index} out of range for {len(sub_prompts_with_contexts)} sub-prompts (expected for retry scenarios), using LLM-provided values")
                    original_sub_prompt = analysis.get("sub_prompt_analyzed", "Unknown sub-prompt")
                    original_title = analysis.get("sub_prompt_title", "Unknown title")

                # Create the result in the expected format
                result = {
                    "sub_prompt_analyzed": analysis.get("sub_prompt_analyzed", original_sub_prompt),
                    "analysis_summary": analysis.get("analysis_summary", "No analysis provided"),
                    "supporting_quotes": analysis.get("supporting_quotes", ["No relevant phrase found."]),
                    "analysis_context": analysis.get("analysis_context", ""),
                    "title": analysis.get("sub_prompt_title", original_title)
                }

                # Ensure supporting_quotes is a list
                if not isinstance(result["supporting_quotes"], list):
                    result["supporting_quotes"] = [str(result["supporting_quotes"])]

                # Convert to JSON string for compatibility with existing code
                results.append({
                    "title": result["title"],
                    "sub_prompt": result["sub_prompt_analyzed"],
                    "analysis_json": json.dumps(result, indent=2)
                })

            # Check if we have results for all sub-prompts
            if len(results) < len(sub_prompts_with_contexts):
                logger.warning(f"Missing analyses for some sub-prompts: got {len(results)}, expected {len(sub_prompts_with_contexts)}")
                # Add fallback analyses for missing sub-prompts
                existing_indices = {r.get("sub_prompt") for r in results}
                for i, item in enumerate(sub_prompts_with_contexts):
                    if item.get("sub_prompt") not in existing_indices:
                        fallback = self._create_fallback_analysis(item)
                        results.append(fallback)

            return results

        except Exception as e:
            logger.error(f"Error during comprehensive AI document analysis for {filename}: {str(e)}", exc_info=True)
            return self._create_fallback_analyses(sub_prompts_with_contexts)

    async def analyze_keyword_occurrences(
        self,
        *,
        filename: str,
        main_prompt: str,
        keyword_sections: Dict[str, Dict[str, Any]],
        total_occurrences: int,
        keyword_reasoning: Optional[str] = None,
        user_request_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the analysis model to summarise keyword occurrences and return structured citations.

        If the decomposition captured additional instructions from the user (user_request_context),
        pass them through so the analysis summary can reflect that guidance.
        """

        keyword_payload = []
        for section_key, section in keyword_sections.items():
            keyword = section.get("keyword")
            occurrences = section.get("occurrences", []) or []
            keyword_payload.append({
                "section_key": section_key,
                "keyword": keyword,
                "occurrences": [
                    {
                        "occurrence_id": occ.get("id"),
                        "page_label": occ.get("page_label"),
                        "match_score": occ.get("match_score"),
                        "snippet": occ.get("snippet"),
                    }
                    for occ in occurrences if isinstance(occ, dict)
                ],
            })

        structured_input = {
            "document": filename,
            "user_prompt": main_prompt,
            "keyword_reasoning": keyword_reasoning,
            "total_occurrences": total_occurrences,
            "keywords": keyword_payload,
        }

        if isinstance(user_request_context, str) and user_request_context.strip():
            structured_input["user_request_context"] = user_request_context.strip()

        system_prompt = """You are assisting with keyword-based document analysis. The decomposition model has already determined that keyword mode should run and has supplied exact keyword occurrences (IDs, page information, and snippets).

You MUST follow these requirements:
1. For each keyword, write an "analysis_summary" that references the provided occurrences and mentions the exact count.
2. Produce a "occurrence_citations" list that contains one entry per occurrence. Do not omit or add occurrences.
3. Each citation entry must include:
   - "occurrence_id" (exactly as provided)
   - "page_label" (use the provided number)
   - "match_score" (reuse the provided value)
   - "citation_text" (a concise supporting quote or paraphrase grounded in the provided snippet)
   - "context_note" (short explanation of why this occurrence matters)
4. Do NOT invent new snippets, page numbers, or occurrences. Use only the supplied data.
5. Write clear, professional prose; reference page numbers naturally (e.g., "on page 4").
6. If "user_request_context" is provided, incorporate its guidance into the analysis_summary phrasing while staying grounded in the supplied occurrences.

Return a single JSON object with the structure:
{
  "keywords": [
    {
      "keyword": "...",
      "total_occurrences": <int>,
      "analysis_summary": "...",
      "occurrence_citations": [
        {
          "occurrence_id": "...",
          "page_label": <int>,
          "match_score": <float>,
          "citation_text": "...",
          "context_note": "..."
        }
      ]
    }
  ]
}

Do not include any additional keys and do not wrap the JSON in Markdown fences."""

        additional_context = ""
        if isinstance(user_request_context, str) and user_request_context.strip():
            additional_context = (
                "\n\nAdditional user context to respect in your analysis summary: "
                f"{user_request_context.strip()}"
            )

        human_prompt = (
            "Summarise the provided keyword occurrences. Here is the structured data you must use:\n\n"
            f"{json.dumps(structured_input, indent=2)}"
            f"{additional_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        fallback_result = self._keyword_analysis_fallback(keyword_sections, keyword_reasoning)

        try:
            parsed_json = await self._get_json_with_retries(
                messages=messages,
                model_name=ANALYSIS_MODEL_NAME,
                context=f"keyword analysis for {filename}",
            )
            return self._sanitize_keyword_analysis(parsed_json, keyword_sections, keyword_reasoning)
        except Exception as err:
            logger.error("Keyword mode analysis failed: %s", err, exc_info=True)
            return fallback_result

    def _create_fallback_analyses(self, sub_prompts_with_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback analyses for all sub-prompts when the main analysis fails."""
        return [self._create_fallback_analysis(item) for item in sub_prompts_with_contexts]

    def _create_fallback_analysis(self, sub_prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback analysis for a single sub-prompt."""
        sub_prompt = sub_prompt_data.get("sub_prompt", "Unknown sub-prompt")
        title = sub_prompt_data.get("title", "Unknown title")

        error_response = {
            "sub_prompt_analyzed": sub_prompt,
            "analysis_summary": "An error occurred while analyzing this sub-prompt.",
            "supporting_quotes": ["No relevant phrase found."],
            "analysis_context": "Analysis Error"
        }

        return {
            "title": title,
            "sub_prompt": sub_prompt,
            "analysis_json": json.dumps(error_response, indent=2)
        }

    def _sanitize_keyword_analysis(
        self,
        parsed_json: Dict[str, Any],
        keyword_sections: Dict[str, Dict[str, Any]],
        keyword_reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ensure keyword analysis output matches available occurrences."""

        sanitized_entries: List[Dict[str, Any]] = []

        parsed_lookup: Dict[str, Dict[str, Any]] = {}
        for item in parsed_json.get("keywords", []) or []:
            keyword = item.get("keyword")
            if isinstance(keyword, str) and keyword.strip():
                parsed_lookup[keyword.strip()] = item

        for section_key, section in keyword_sections.items():
            keyword = section.get("keyword")
            occurrences = section.get("occurrences", []) or []
            expected_by_id = {occ.get("id"): occ for occ in occurrences if isinstance(occ, dict) and occ.get("id")}

            parsed_item = parsed_lookup.get(keyword)
            sanitized_citations: List[Dict[str, Any]] = []
            used_occurrence_ids = set()

            if parsed_item:
                for citation in parsed_item.get("occurrence_citations", []) or []:
                    occurrence_id = citation.get("occurrence_id")
                    if occurrence_id not in expected_by_id:
                        continue
                    occ_data = expected_by_id[occurrence_id]
                    sanitized_citations.append({
                        "occurrence_id": occurrence_id,
                        "page_label": occ_data.get("page_label"),
                        "match_score": occ_data.get("match_score", 0.0),
                        "citation_text": citation.get("citation_text") or occ_data.get("snippet"),
                        "context_note": citation.get("context_note", ""),
                        "snippet": occ_data.get("snippet"),
                    })
                    used_occurrence_ids.add(occurrence_id)

            for occurrence_id, occ_data in expected_by_id.items():
                if occurrence_id in used_occurrence_ids:
                    continue
                sanitized_citations.append({
                    "occurrence_id": occurrence_id,
                    "page_label": occ_data.get("page_label"),
                    "match_score": occ_data.get("match_score", 0.0),
                    "citation_text": occ_data.get("snippet"),
                    "context_note": "Automatically generated citation based on provided snippet.",
                    "snippet": occ_data.get("snippet"),
                })

            sanitized_citations.sort(key=lambda entry: entry.get("occurrence_id", ""))

            analysis_summary = None
            if parsed_item and isinstance(parsed_item.get("analysis_summary"), str):
                analysis_summary = parsed_item["analysis_summary"].strip()

            if not analysis_summary:
                analysis_summary = self._build_keyword_summary(keyword, sanitized_citations)

            sanitized_entries.append({
                "section_key": section_key,
                "keyword": keyword,
                "analysis_summary": analysis_summary,
                "occurrence_citations": sanitized_citations,
                "total_occurrences": len(sanitized_citations),
            })

        return {
            "keywords": sanitized_entries,
            "total_occurrences": sum(entry["total_occurrences"] for entry in sanitized_entries),
            "keyword_reasoning": keyword_reasoning,
        }

    def _keyword_analysis_fallback(
        self,
        keyword_sections: Dict[str, Dict[str, Any]],
        keyword_reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        fallback_entries: List[Dict[str, Any]] = []
        for section_key, section in keyword_sections.items():
            keyword = section.get("keyword")
            occurrences = section.get("occurrences", []) or []
            citations = []
            for occ in occurrences:
                if not isinstance(occ, dict):
                    continue
                citations.append({
                    "occurrence_id": occ.get("id"),
                    "page_label": occ.get("page_label"),
                    "match_score": occ.get("match_score", 0.0),
                    "citation_text": occ.get("snippet"),
                    "context_note": "Automatically generated citation based on provided snippet.",
                    "snippet": occ.get("snippet"),
                })

            fallback_entries.append({
                "section_key": section_key,
                "keyword": keyword,
                "analysis_summary": self._build_keyword_summary(keyword, citations),
                "occurrence_citations": citations,
                "total_occurrences": len(citations),
            })

        return {
            "keywords": fallback_entries,
            "total_occurrences": sum(entry["total_occurrences"] for entry in fallback_entries),
            "keyword_reasoning": keyword_reasoning,
        }

    @staticmethod
    def _build_keyword_summary(keyword: Optional[str], citations: List[Dict[str, Any]]) -> str:
        keyword_label = keyword or "the keyword"
        count = len(citations)
        if not citations:
            return f"No occurrences of '{keyword_label}' were detected in the supplied snippets." if keyword else "No keyword occurrences were detected."

        pages = sorted({citation.get("page_label") for citation in citations if isinstance(citation.get("page_label"), (int, float))})
        if pages:
            page_part = "page" if len(pages) == 1 else "pages"
            page_numbers = ", ".join(str(int(p)) for p in pages)
            return f"Found {count} occurrence(s) of '{keyword_label}' on {page_part} {page_numbers}."
        return f"Found {count} occurrence(s) of '{keyword_label}'."