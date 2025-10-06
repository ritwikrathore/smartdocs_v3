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
        prompt = f"""You are a relevance scoring system. Given a query and a document, score how relevant the document is to answering the query.

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
                return score
            except ValueError:
                logger.warning(f"LLM reranker returned non-numeric score: {score_text}, using 0.5")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error scoring pair with LLM: {e}")
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
Respond with ONLY the scores, one per line, in order. Example format:
0.8
0.3
0.9"""

            messages = [
                {"role": "system", "content": "You are a precise relevance scoring system. Respond only with numbers, one per line."},
                {"role": "user", "content": batch_prompt}
            ]

            logger.debug(f"Batch prompt length: {len(batch_prompt)} characters")

            # Calculate appropriate max_tokens based on number of pairs
            # Each score is ~4 tokens (e.g., "0.85\n"), so we need ~4 * num_pairs tokens
            max_tokens = min(4 * len(sentence_pairs) + 20, 500)

            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.0
            )

            score_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {score_text[:500]}...")

            lines = score_text.split('\n')

            scores = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Try to extract just the number (handle cases like "1. 0.85" or "Score: 0.85")
                    # First try direct float conversion
                    score = float(line)
                    scores.append(max(0.0, min(1.0, score)))
                except ValueError:
                    # Try to extract number from text
                    import re
                    numbers = re.findall(r'0?\.\d+|[01]\.?\d*', line)
                    if numbers:
                        try:
                            score = float(numbers[0])
                            scores.append(max(0.0, min(1.0, score)))
                            logger.debug(f"Extracted score {score} from line: {line}")
                        except ValueError:
                            logger.warning(f"Could not parse score from line {line_num}: {line}")
                            continue
                    else:
                        logger.warning(f"No number found in line {line_num}: {line}")
                        continue

            # If we got the right number of scores, return them
            if len(scores) == len(sentence_pairs):
                logger.info(f"✓ LLM reranker: Successfully scored {len(scores)} pairs in ONE batch call")
                return np.array(scores)
            else:
                logger.warning(f"✗ LLM batch scoring returned {len(scores)} scores, expected {len(sentence_pairs)}")
                logger.warning(f"Response was: {score_text[:200]}...")
                logger.warning(f"Falling back to individual scoring for {len(sentence_pairs)} pairs")

        except Exception as e:
            logger.error(f"✗ Error in batch scoring: {e}")
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
        
        # Test the model with a simple input
        test_pairs = [
            ["What is the capital of France?", "Paris is the capital and largest city of France."],
            ["What is the capital of France?", "The Eiffel Tower is a famous landmark."]
        ]
        
        try:
            test_scores = model.predict(test_pairs)
            
            if isinstance(test_scores, np.ndarray) and len(test_scores) == 2:
                logger.info(f"LLM fallback reranker loaded successfully. Test scores: {test_scores}")
                # Verify that the first score is higher (more relevant)
                if test_scores[0] > test_scores[1]:
                    logger.info("LLM fallback reranker test passed: correctly ranked relevant document higher")
                else:
                    logger.warning(f"LLM fallback reranker test warning: expected first score > second score, got {test_scores}")
                return model
            else:
                logger.error(f"LLM fallback reranker test failed: Invalid score format")
                return None
                
        except Exception as e:
            logger.error(f"LLM fallback reranker test failed: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading LLM fallback reranker: {e}")
        return None

