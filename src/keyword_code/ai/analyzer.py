"""
Document analyzer functionality.
"""

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from ..config import logger, ANALYSIS_MODEL_NAME, USE_DATABRICKS_LLM

# Import Databricks LLM client
from .databricks_llm import get_databricks_llm

# Import interaction logger
from ..utils.interaction_logger import log_llm_interaction


_thread_local = threading.local()


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
        self, messages: List[Dict[str, str]], model_name: str
    ) -> str:
        """Helper method to get completion from Databricks LLM."""
        try:
            # Ensure Databricks client is available
            if not self.databricks_client:
                raise ValueError("Databricks LLM client not initialized")

            logger.info(f"Sending request to Databricks LLM model")
            # Databricks client handles message formatting internally
            response_content = await self.databricks_client.get_completion_async(messages, max_tokens=8192)

            if not response_content:
                raise ValueError("Failed to get response from Databricks LLM")

            logger.info(f"Received response from Databricks LLM model")

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
                f"Error getting completion from Databricks LLM: {str(e)}", exc_info=True
            )
            raise

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
                    "Context": "Optional context about this section (e.g., source sub-prompt)",
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
                    formatted_context = "\n\n---\n\n".join([
                        f"Chunk ID: {chunk.get('chunk_id', 'unknown')}, Page: {chunk.get('page_num', -1) + 1}, Score: {chunk.get('score', 0):.3f}\n"
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

### Core Instructions:
1. **Analyze Each Sub-prompt Separately:** For each sub-prompt, provide a focused analysis using ONLY the context provided for that specific sub-prompt.
2. **Structured Response:** Your response must follow the JSON structure specified below, with an analysis for each sub-prompt.
3. **Direct Answers:** For each sub-prompt, provide a comprehensive analysis that directly answers the question.
4. **Exact Supporting Quotes:** For each sub-prompt, include direct, verbatim quotes from its context that support your analysis.
5. **No Cross-Referencing:** Do not use context from one sub-prompt to answer another sub-prompt, even if they seem related.
6. **No Information Found:** If the context for a sub-prompt does not contain information to answer it, clearly state this in the analysis.
7. **Do not mention chunk ids in your analysis text:** Do not include references to chunk id in your analysis text.

### JSON Output Schema:
```json
{
  "analyses": [
    {
      "sub_prompt_index": 1,
      "sub_prompt_title": "Title of the first sub-prompt",
      "sub_prompt_analyzed": "The exact first sub-prompt being analyzed",
      "analysis_summary": "Detailed analysis directly answering the first sub-prompt...",
      "supporting_quotes": [
        "Exact quote 1 from the document text for the first sub-prompt...",
        "Exact quote 2, potentially longer..."
      ],
      "analysis_context": "Optional context about the analysis (e.g., document section names)"
    },
    {
      "sub_prompt_index": 2,
      "sub_prompt_title": "Title of the second sub-prompt",
      "sub_prompt_analyzed": "The exact second sub-prompt being analyzed",
      "analysis_summary": "Detailed analysis directly answering the second sub-prompt...",
      "supporting_quotes": [
        "Exact quote 1 from the document text for the second sub-prompt...",
        "Exact quote 2, potentially longer..."
      ],
      "analysis_context": "Optional context about the analysis (e.g., document section names)"
    }
    // Additional analyses for each sub-prompt...
  ]
}
```

Your entire response MUST be a single JSON object following this schema. Do not include any introductory text, explanations, or markdown formatting outside the JSON structure.
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
Generate a structured analysis for EACH sub-prompt, strictly following the JSON schema provided in the system instructions. Ensure each analysis only addresses its specific sub-prompt and uses only the context provided for that sub-prompt.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]

            logger.info(f"Sending comprehensive analysis request for {len(formatted_sub_prompts)} sub-prompts in {filename} to AI")

            # Call the LLM with all sub-prompts and contexts
            response_content = await self._get_completion(messages, model_name=ANALYSIS_MODEL_NAME)
            logger.info(f"Received comprehensive AI analysis response for {filename}")

            # Parse the JSON response
            try:
                cleaned_response = response_content.strip()
                match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                elif cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                    json_str = cleaned_response
                else:
                    first_brace = cleaned_response.find('{')
                    last_brace = cleaned_response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = cleaned_response[first_brace:last_brace+1]
                        logger.warning("Used basic brace finding for JSON extraction in comprehensive analysis.")
                    else:
                        raise json.JSONDecodeError("Could not find JSON structure in analysis response.", cleaned_response, 0)

                parsed_json = json.loads(json_str)

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
                        logger.warning(f"Invalid sub_prompt_index: {sub_prompt_index}, using provided values")
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

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse comprehensive AI analysis response as JSON: {json_err}")
                return self._create_fallback_analyses(sub_prompts_with_contexts)
            except Exception as e:
                logger.error(f"Error processing comprehensive analysis response: {str(e)}", exc_info=True)
                return self._create_fallback_analyses(sub_prompts_with_contexts)

        except Exception as e:
            logger.error(f"Error during comprehensive AI document analysis for {filename}: {str(e)}", exc_info=True)
            return self._create_fallback_analyses(sub_prompts_with_contexts)

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


