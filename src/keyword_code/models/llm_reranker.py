"""
LLM-based fallback reranker for when the Databricks reranker API is unavailable.

This module provides a fallback reranker that uses the Databricks LLM to score
query-document pairs when the dedicated reranker API is unavailable. It maintains
the same interface as DatabricksRerankerModel for seamless integration.
"""

import numpy as np
from typing import List, Tuple
from ..config import logger
from ..ai.databricks_llm import get_databricks_llm_client
from ..text_utils import clean_and_parse_json
from ..utils.langfuse_tracing import (
    optional_context,
    record_generation_error,
    set_generation_output,
    start_generation,
)


class LLMRerankerModel:
    """
    LLM-based reranker that uses Databricks LLM to score query-document pairs.
    
    This class provides a fallback reranking mechanism when the dedicated
    reranker API is unavailable. It uses the LLM to assess relevance and
    returns scores in the same format as DatabricksRerankerModel.
    """

    def __init__(self, max_length: int = None):
        """
        Initialize the LLM-based reranker.

        Args:
            max_length: Maximum character length for document text.
                       If None, no truncation is applied (recommended).
                       The LLM has a large context window (~8K tokens = ~32K chars),
                       so truncation is typically not needed.
        """
        self.client = get_databricks_llm_client()
        self.max_length = max_length  # None by default - no truncation
        self.model_name = "databricks-llama-4-maverick"

        if self.client is None:
            logger.error("Failed to initialize LLM client for fallback reranker")
            raise ValueError("LLM client not available for fallback reranker")

        if self.max_length:
            logger.info(f"LLM-based fallback reranker initialized with max_length={self.max_length}")
        else:
            logger.info("LLM-based fallback reranker initialized (no truncation)")

    def _truncate_text(self, text: str, max_chars: int = None) -> str:
        """
        Truncate text to a maximum character length if needed.

        Note: Truncation is typically not needed for the LLM reranker as it has
        a large context window (~8K tokens = ~32K chars). This method is only
        used if max_length is explicitly set or for extreme edge cases.

        Args:
            text: The text to truncate
            max_chars: Maximum number of characters (uses self.max_length if None)

        Returns:
            Original or truncated text
        """
        max_chars = max_chars or self.max_length

        # If no max_chars specified, return text as-is (no truncation)
        if max_chars is None:
            return text

        if len(text) <= max_chars:
            return text

        # Only truncate if text exceeds max_chars
        logger.warning(f"Truncating document from {len(text)} to {max_chars} characters")
        truncated = text[:max_chars]

        # Try to truncate at a sentence boundary
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.7:  # Only use period if it's not too far back
            return truncated[:last_period + 1]

        # Try to truncate at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space]

        return truncated

    def _score_single_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair using the LLM.

        Args:
            query: The search query
            document: The document text to score

        Returns:
            Relevance score between 0 and 1
        """
        # Only truncate if max_length is set (typically not needed)
        truncated_doc = self._truncate_text(document)
        
        # Create a prompt for the LLM to score relevance
        prompt = f"""You are a relevance scoring system. Given a query and a chunk of text, score how relevant the chunk is to answering the query, take the metadata into account.

Query: {query}

Document: {truncated_doc}

Provide a relevance score between 0.0 and 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat relevant
- 1.0 = highly relevant and directly answers the query

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        messages = [
            {"role": "system", "content": "You are a precise relevance scoring system. Respond only with a single number between 0.0 and 1.0."},
            {"role": "user", "content": prompt}
        ]

        with optional_context(
            start_generation(
                name="reranker.llm.score_single",
                input_data={"query": query, "document_preview": truncated_doc[:200]},
                metadata={"operation": "llm_reranker.single_pair"},
                model=self.model_name,
            )
        ) as generation:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=10,
                    temperature=0.0  # Use deterministic scoring
                )
                
                score_text = response.choices[0].message.content.strip()
                
                # Extract the numeric score
                try:
                    score = float(score_text)
                    # Clamp score to [0, 1] range
                    score = max(0.0, min(1.0, score))
                    
                    set_generation_output(
                        generation,
                        output={"score": score, "raw_response": score_text},
                        metadata={"score": score},
                    )
                    return score
                except ValueError:
                    logger.warning(f"LLM reranker returned non-numeric score: {score_text}, using 0.5")
                    set_generation_output(
                        generation,
                        output={"score": 0.5, "raw_response": score_text, "parse_error": True},
                        metadata={"score": 0.5, "error": "non_numeric_response"},
                    )
                    return 0.5
                    
            except Exception as e:
                logger.error(f"Error scoring pair with LLM: {e}")
                record_generation_error(
                    generation,
                    e,
                    metadata={"stage": "llm_reranker.single_pair"},
                )
                return 0.5  # Return neutral score on error

    def _score_batch_optimized(self, sentence_pairs: List[List[str]]) -> np.ndarray:
        """
        Score multiple pairs in a single LLM call for efficiency.

        This method batches ALL pairs into a single API call, which is much faster
        than making individual calls per chunk. The LLM has a large context window
        (~8K tokens) so it can handle 15-20+ pairs in one call.

        Note: For batch scoring, we use the full document text without truncation
        since the LLM has a large context window. Only truncate if explicitly needed.

        Args:
            sentence_pairs: List of [query, document] pairs

        Returns:
            numpy array of scores
        """
        if not sentence_pairs:
            return np.array([])

        # IMPORTANT: This should show ONE batch call, not multiple individual calls!
        logger.error(f"============ BATCH OPTIMIZATION START: {len(sentence_pairs)} pairs ============")

        # Try to batch ALL pairs in a single call (much faster!)
        # The LLM can handle 15-20+ pairs easily with its large context window
        logger.info(f"🚀 LLM reranker: Attempting to score {len(sentence_pairs)} pairs in ONE batch call")

        try:
            # Create a batch prompt with all pairs
            batch_prompt = "You are a relevance scoring system. Score each query-document pair below.\n\n"

            for i, (query, document) in enumerate(sentence_pairs):
                # Use full document text - no truncation unless max_length is set
                doc_text = self._truncate_text(document)
                batch_prompt += f"Pair {i+1}:\nQuery: {query}\nDocument: {doc_text}\n\n"

            batch_prompt += f"""For each of the {len(sentence_pairs)} pairs above, provide a relevance score between 0.0 and 1.0.
Respond with ONLY a JSON object containing a list of scores. The JSON should have this structure:
{{
  "scores": [
    {{"index": 1, "score": 0.8}},
    {{"index": 2, "score": 0.3}},
    ...
  ]
}}
Ensure you include all {len(sentence_pairs)} pairs."""

            messages = [
                {"role": "system", "content": "You are a precise relevance scoring system. Respond only with valid JSON."},
                {"role": "user", "content": batch_prompt}
            ]

            # Calculate appropriate max_tokens based on number of pairs
            # Each score object {"index": 12, "score": 0.85} is ~15 tokens
            max_tokens = min(20 * len(sentence_pairs) + 100, 2000)

            # Log the full request in DEBUG mode
            logger.debug("=" * 80)
            logger.debug("LLM RERANKER API REQUEST")
            logger.debug("=" * 80)
            logger.debug(f"Model: {self.model_name}")
            logger.debug(f"Max Tokens: {max_tokens}")
            logger.debug(f"Temperature: 0.0")
            logger.debug(f"Batch Size: {len(sentence_pairs)} pairs")
            logger.debug("System Message:")
            logger.debug(messages[0]["content"])
            logger.debug("User Message (Batch Prompt):")
            logger.debug(batch_prompt)
            logger.debug("=" * 80)

            with optional_context(
                start_generation(
                    name="reranker.llm.batch_score",
                    input_data={
                        "num_pairs": len(sentence_pairs),
                        "queries_preview": [pair[0][:100] for pair in sentence_pairs[:3]],
                        "pairs": [
                            {
                                "query": pair[0],
                                "document": pair[1][:500] + ("..." if len(pair[1]) > 500 else ""),  # Truncate for display
                            }
                            for pair in sentence_pairs
                        ],
                    },
                    metadata={
                        "operation": "llm_reranker.batch",
                        "batch_size": len(sentence_pairs),
                    },
                    model=self.model_name,
                )
            ) as generation:
                try:
                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=0.0
                    )

                    score_text = response.choices[0].message.content.strip()

                    # Extract and log token usage
                    usage_info = None
                    if hasattr(response, 'usage') and response.usage:
                        usage_info = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        # Update generation with usage info
                        from ..utils.langfuse_tracing import set_generation_output
                        set_generation_output(generation, usage=usage_info)

                    # Log the full response in DEBUG mode
                    logger.debug("=" * 80)
                    logger.debug("LLM RERANKER API RESPONSE")
                    logger.debug("=" * 80)
                    logger.debug(f"Response Length: {len(score_text)} characters")
                    if usage_info:
                        logger.debug(f"Token Usage: {usage_info}")
                    logger.debug("Response Content:")
                    logger.debug(score_text)
                    logger.debug("=" * 80)

                    import json
                    import re
                    
                    score_map = {}
                    
                    # Use the centralized helper to clean and parse JSON
                    data = clean_and_parse_json(score_text)
                    
                    if data is not None:
                        # Handle different possible JSON structures
                        items = []
                        if isinstance(data, dict):
                            if "scores" in data and isinstance(data["scores"], list):
                                items = data["scores"]
                            else:
                                # Maybe the dict itself is the list or has other keys?
                                # Try to find any list in values
                                for val in data.values():
                                    if isinstance(val, list):
                                        items = val
                                        break
                        elif isinstance(data, list):
                            items = data
                            
                        # Process items
                        for item in items:
                            idx = -1
                            score_val = -1.0
                            
                            if isinstance(item, dict):
                                # Try to find index
                                if "index" in item:
                                    idx = int(item["index"]) - 1  # Convert 1-based to 0-based
                                elif "pair" in item:
                                    idx = int(item["pair"]) - 1
                                    
                                # Try to find score
                                if "score" in item:
                                    score_val = float(item["score"])
                                elif "relevance" in item:
                                    score_val = float(item["relevance"])
                            elif isinstance(item, (int, float)):
                                # Simple list of scores?
                                # We can't be sure of index, but let's assume sequential if we have to
                                pass
                                
                            if 0 <= idx < len(sentence_pairs) and score_val >= 0:
                                score_map[idx] = max(0.0, min(1.0, score_val))
                    
                    # If JSON parsing failed or returned empty/invalid structure, try regex fallback
                    if not score_map:
                        logger.warning(f"JSON parsing failed or yielded no scores. Trying regex fallback.")
                        # Fallback to regex parsing if JSON fails
                        lines = score_text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            
                            # Try to match "Pair N: score" or similar patterns inside JSON-like text
                            pair_match = re.search(r'(?:index|pair)["\s:]+(\d+).*?(?:score|relevance)["\s:]+([\d\.]+)', line, re.IGNORECASE)
                            if pair_match:
                                try:
                                    idx = int(pair_match.group(1)) - 1
                                    score_val = float(pair_match.group(2))
                                    if 0 <= idx < len(sentence_pairs):
                                        score_map[idx] = max(0.0, min(1.0, score_val))
                                except ValueError:
                                    pass

                    # Check if we have scores for all pairs
                    scores = []
                    missing_indices = []
                    
                    for i in range(len(sentence_pairs)):
                        if i in score_map:
                            scores.append(score_map[i])
                        else:
                            missing_indices.append(i)
                            
                    # If we have all scores, return them
                    if not missing_indices:
                        logger.info(f"✓ LLM reranker: Successfully scored {len(scores)} pairs in ONE batch call")
                        set_generation_output(
                            generation,
                            output={
                                "scores": scores,
                                "mean_score": float(np.mean(scores)),
                                "scored_pairs": [
                                    {
                                        "query": pair[0],
                                        "document": pair[1][:500] + ("..." if len(pair[1]) > 500 else ""),
                                        "score": float(score),
                                    }
                                    for pair, score in zip(sentence_pairs, scores)
                                ],
                            },
                            metadata={
                                "num_scores": len(scores),
                                "mean_score": float(np.mean(scores)),
                                "min_score": float(np.min(scores)),
                                "max_score": float(np.max(scores)),
                            },
                        )
                        return np.array(scores)
                    else:
                        logger.warning(f"✗ LLM batch scoring missing {len(missing_indices)} scores. Found: {len(scores)}")
                        logger.warning(f"Response was: {score_text[:200]}...")
                        logger.warning(f"Falling back to individual scoring for {len(sentence_pairs)} pairs")
                        record_generation_error(
                            generation,
                            ValueError(f"Batch scoring incomplete. Missing indices: {missing_indices}"),
                            metadata={"expected": len(sentence_pairs), "found": len(scores), "missing": missing_indices},
                        )

                except Exception as e:
                    logger.error(f"✗ Error in batch scoring: {e}")
                    logger.error(f"Falling back to individual scoring for {len(sentence_pairs)} pairs", exc_info=True)
                    record_generation_error(
                        generation,
                        e,
                        metadata={"stage": "llm_reranker.batch_scoring"},
                    )

        except Exception as e:
            logger.error(f"✗ Error setting up batch scoring: {e}")
            logger.error(f"Falling back to individual scoring for {len(sentence_pairs)} pairs", exc_info=True)

        # Fallback: score each pair individually (only if batch fails)
        logger.error(f"============ FALLBACK TO INDIVIDUAL SCORING: {len(sentence_pairs)} pairs ============")
        logger.error(f"⚠️ BATCH OPTIMIZATION FAILED! Scoring {len(sentence_pairs)} pairs individually (this will be slow!)")
        scores = []
        for i, (query, document) in enumerate(sentence_pairs, 1):
            if i == 1 or i == len(sentence_pairs):
                logger.error(f"Scoring pair {i}/{len(sentence_pairs)} individually")
            score = self._score_single_pair(query, document)
            scores.append(score)

        logger.error(f"============ COMPLETED INDIVIDUAL SCORING: {len(sentence_pairs)} pairs ============")
        return np.array(scores)

    def predict(self, sentence_pairs: List[List[str]]) -> np.ndarray:
        """
        Predict the relevance scores for a list of sentence pairs.

        This method maintains the same interface as DatabricksRerankerModel.predict()
        for seamless integration.

        IMPORTANT: All pairs are scored in a SINGLE batch API call for efficiency.
        For example, if reranking 15 chunks for a subprompt, this makes 1 API call
        instead of 15 individual calls, resulting in ~15x speedup.

        Args:
            sentence_pairs: List of [query, document] pairs (typically 10-20 pairs per subprompt)

        Returns:
            numpy array of scores between 0 and 1
        """
        if not sentence_pairs:
            return np.array([])

        logger.info(f"LLM fallback reranker scoring {len(sentence_pairs)} pairs")
        scores = self._score_batch_optimized(sentence_pairs)
        logger.info(f"LLM fallback reranker completed scoring with mean score: {scores.mean():.4f}")
        return scores


def load_llm_reranker_model():
    """
    Loads the LLM-based fallback reranker model.

    Returns:
        LLMRerankerModel instance or None if initialization fails
    """
    try:
        logger.info("Loading LLM-based fallback reranker model")
        model = LLMRerankerModel()
        logger.info("LLM fallback reranker loaded successfully")
        return model
            
    except Exception as e:
        logger.error(f"Error loading LLM fallback reranker: {e}")
        return None

