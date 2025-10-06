"""
Prompt decomposition functionality.
"""

import json
import re
from typing import Dict, List
from ..config import logger, DECOMPOSITION_MODEL_NAME, USE_DATABRICKS_LLM


async def decompose_prompt(analyzer, user_prompt: str) -> List[Dict[str, str]]:
    """
    Analyzes the user prompt with an LLM to break it down into individual questions/tasks,
    each with a suggested concise title and optimal RAG retrieval parameters.
    Returns a list of dictionaries, each containing 'sub_prompt', 'title', and 'rag_params'.
    Returns [{'sub_prompt': user_prompt, 'title': 'Overall Analysis', 'rag_params': {...}}] on failure.
    """
    logger.info(f"Decomposing prompt with RAG optimization: '{user_prompt[:100]}...'")
    system_prompt = """You are a helpful assistant specializing in financial and legal document analysis. Your task is to analyze the user's prompt and identify distinct questions or analysis tasks within it.

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

Your entire response MUST be a single JSON object containing a single key "decomposition", whose value is a list of JSON objects. Each object must have:
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
    human_prompt = f"Analyze the following prompt and return the decomposed questions/tasks with their titles and optimal RAG parameters as a JSON list of objects according to the system instructions:\n\n{user_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    # Fallback result in case of errors - includes default balanced RAG params
    fallback_result = [{
        "title": "Overall Analysis",
        "sub_prompt": user_prompt,
        "rag_params": {
            "bm25_weight": 0.5,
            "semantic_weight": 0.5,
            "reasoning": "Default balanced weights due to decomposition failure"
        }
    }]

    try:
        response_content = await analyzer._get_completion(messages, model_name=DECOMPOSITION_MODEL_NAME)

        # Attempt to parse the JSON
        try:
            # Clean potential markdown fences (same logic as before)
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
                     logger.warning("Used basic brace finding for JSON extraction in decomposition.")
                 else:
                     raise json.JSONDecodeError("Could not find JSON structure.", cleaned_response, 0)

            parsed_json = json.loads(json_str)

            # Validate the new structure with RAG params
            if isinstance(parsed_json, dict) and "decomposition" in parsed_json:
                decomposition_list = parsed_json["decomposition"]
                if isinstance(decomposition_list, list):
                    valid_items = []

                    for item in decomposition_list:
                        # Validate basic structure
                        if not isinstance(item, dict) or "title" not in item or "sub_prompt" not in item:
                            logger.warning(f"Skipping invalid decomposition item (missing title or sub_prompt): {item}")
                            continue

                        if not isinstance(item["title"], str) or not isinstance(item["sub_prompt"], str):
                            logger.warning(f"Skipping invalid decomposition item (non-string title or sub_prompt): {item}")
                            continue

                        if not item["title"].strip() or not item["sub_prompt"].strip():
                            logger.warning(f"Skipping decomposition item with empty title or sub_prompt")
                            continue

                        # Validate and normalize RAG params
                        if "rag_params" not in item or not isinstance(item["rag_params"], dict):
                            logger.warning(f"Missing or invalid rag_params for '{item['title']}', using defaults")
                            item["rag_params"] = {
                                "bm25_weight": 0.5,
                                "semantic_weight": 0.5,
                                "reasoning": "Default balanced weights (rag_params missing from LLM response)"
                            }
                        else:
                            rag_params = item["rag_params"]

                            # Validate weights
                            bm25_weight = rag_params.get("bm25_weight", 0.5)
                            semantic_weight = rag_params.get("semantic_weight", 0.5)

                            # Ensure weights are numbers in valid range
                            try:
                                bm25_weight = float(bm25_weight)
                                semantic_weight = float(semantic_weight)

                                if not (0.0 <= bm25_weight <= 1.0) or not (0.0 <= semantic_weight <= 1.0):
                                    logger.warning(f"RAG weights out of range for '{item['title']}': bm25={bm25_weight}, semantic={semantic_weight}. Using defaults.")
                                    bm25_weight, semantic_weight = 0.5, 0.5

                                # Normalize weights to sum to 1.0
                                weight_sum = bm25_weight + semantic_weight
                                if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
                                    logger.warning(f"RAG weights don't sum to 1.0 for '{item['title']}' (sum={weight_sum:.3f}). Normalizing.")
                                    if weight_sum > 0:
                                        bm25_weight = bm25_weight / weight_sum
                                        semantic_weight = semantic_weight / weight_sum
                                    else:
                                        bm25_weight, semantic_weight = 0.5, 0.5

                                # Update normalized weights
                                item["rag_params"]["bm25_weight"] = bm25_weight
                                item["rag_params"]["semantic_weight"] = semantic_weight

                                # Ensure reasoning exists
                                if "reasoning" not in rag_params or not isinstance(rag_params["reasoning"], str):
                                    item["rag_params"]["reasoning"] = "No reasoning provided"

                                logger.info(f"RAG params for '{item['title']}': BM25={bm25_weight:.2f}, Semantic={semantic_weight:.2f}, Reasoning: {rag_params.get('reasoning', 'N/A')}")

                            except (ValueError, TypeError) as e:
                                logger.warning(f"Invalid RAG weight types for '{item['title']}': {e}. Using defaults.")
                                item["rag_params"] = {
                                    "bm25_weight": 0.5,
                                    "semantic_weight": 0.5,
                                    "reasoning": "Default balanced weights (invalid weight values from LLM)"
                                }

                        valid_items.append(item)

                    if not valid_items:
                        logger.warning("Decomposition resulted in an empty list after filtering. Falling back.")
                        return fallback_result

                    logger.info(f"Successfully decomposed prompt into {len(valid_items)} sub-prompts with titles and RAG parameters.")
                    return valid_items
                else:
                    logger.warning("Decomposition JSON found, but 'decomposition' key is not a list. Falling back.")
                    return fallback_result
            else:
                logger.warning("Decomposition JSON parsed, but missing 'decomposition' key or wrong structure. Falling back.")
                return fallback_result

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse decomposition response as JSON: {json_err}. Raw response: {response_content}")
            logger.warning("Falling back to using the original prompt due to JSON parsing error.")
            return fallback_result

    except TimeoutError:
        logger.error(f"Prompt decomposition request timed out. Falling back to original prompt.")
        return fallback_result
    except Exception as e:
        logger.error(f"Error during prompt decomposition LLM call: {str(e)}", exc_info=True)
        logger.warning("Falling back to using the original prompt.")
        return fallback_result
