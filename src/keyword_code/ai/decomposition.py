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
    each with a suggested concise title.
    Returns a list of dictionaries, each containing 'sub_prompt' and 'title'.
    Returns [{'sub_prompt': user_prompt, 'title': 'Overall Analysis'}] on failure.
    """
    logger.info(f"Decomposing prompt: '{user_prompt[:100]}...'")
    system_prompt = """You are a helpful assistant specializing in financial and legal document analysis. Your task is to analyze the user's prompt and identify distinct questions or analysis tasks within it.
Break down the prompt into a list of self-contained, individual questions or tasks. For each task, also provide a concise, descriptive title (max 5-6 words).
Your entire response MUST be a single JSON object containing a single key "decomposition", whose value is a list of JSON objects. Each object in the list must have two keys: "title" (the concise title) and "sub_prompt" (the full sub-prompt text).
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
      "sub_prompt": "What is the defined / lawful loan currency?"
    },
    {
      "title": "Availability Period Duration",
      "sub_prompt": "What is the duration of the availability period?"
    },
    {
      "title": "Loan Amount",
      "sub_prompt": "What is the loan amount?"
    },
    {
      "title": "Loan Currency",
      "sub_prompt": "What is the loan currency?"
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
      "sub_prompt": "Analyze the termination clause in the loan agreement."
    },
    {
      "title": "Liability Limitations Analysis",
      "sub_prompt": "Analyze the liability limitations in the loan agreement."
    }
  ]
}

Example Input Prompt:
"Did Microsoft or Google have higher revenue growth last quarter?"

Example JSON Output:
{
  "decomposition": [
    {
      "title": "Microsoft Revenue Growth",
      "sub_prompt": "What was Microsoft's revenue growth last quarter?"
    },
    {
      "title": "Google Revenue Growth",
      "sub_prompt": "What was Google's revenue growth last quarter?"
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
      "sub_prompt": "What are the interest rates for this loan?"
    },
    {
      "title": "Loan Fees",
      "sub_prompt": "What are the fees for this loan?"
    }
  ]
}
"""
    human_prompt = f"Analyze the following prompt and return the decomposed questions/tasks and their titles as a JSON list of objects according to the system instructions:\n\n{user_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    # Fallback result in case of errors
    fallback_result = [{"title": "Overall Analysis", "sub_prompt": user_prompt}]

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

            # Validate the new structure
            if isinstance(parsed_json, dict) and "decomposition" in parsed_json:
                decomposition_list = parsed_json["decomposition"]
                if (isinstance(decomposition_list, list) and
                    all(isinstance(item, dict) and "title" in item and "sub_prompt" in item
                        and isinstance(item["title"], str) and isinstance(item["sub_prompt"], str)
                        for item in decomposition_list)):

                    logger.info(f"Successfully decomposed prompt into {len(decomposition_list)} sub-prompts with titles.")
                    # Filter out items with empty titles or sub-prompts
                    valid_items = [item for item in decomposition_list if item["title"].strip() and item["sub_prompt"].strip()]
                    if not valid_items:
                         logger.warning("Decomposition resulted in an empty list after filtering. Falling back.")
                         return fallback_result
                    return valid_items
                else:
                    logger.warning("Decomposition JSON found, but 'decomposition' key is not a list of valid {'title': str, 'sub_prompt': str} objects. Falling back.")
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
