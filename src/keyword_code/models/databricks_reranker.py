"""
Databricks reranker model integration.

This module provides a wrapper around the Databricks reranker API to make it
compatible with the CrossEncoder interface used in the application.

The client is cached using Streamlit's cache_resource decorator to avoid
recreating it on each Streamlit rerun, which is a performance optimization.
"""

import os
import numpy as np
import streamlit as st
import pandas as pd
import json
import requests
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path
from ..config import (
    logger,
    USE_DATABRICKS_RERANKER,
    RERANKER_MAX_TOKENS,
    RERANKER_API_TIMEOUT,
    ENABLE_LLM_RERANKER_FALLBACK
)
from transformers import AutoTokenizer

# Databricks endpoint URL
DATABRICKS_BASE_URL = "https://int.api.worldbank.org/portfoliointelligence/serving-endpoints"
DATABRICKS_RERANKER_MODEL_NAME = "cpm-marco-ms"
DATABRICKS_RERANKER_ENDPOINT = f"{DATABRICKS_BASE_URL}/{DATABRICKS_RERANKER_MODEL_NAME}/invocations"


def create_tf_serving_json(data):
    """
    Create the JSON payload for TensorFlow Serving.

    Args:
        data: The data to convert to TensorFlow Serving format

    Returns:
        Dict with the formatted data
    """
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}


def get_databricks_reranker_token():
    """
    Gets the Databricks token (not cached to ensure fresh token retrieval).

    Returns:
        The Databricks token or None if not available
    """
    try:
        # Check if Databricks reranker is enabled
        if not USE_DATABRICKS_RERANKER:
            logger.info("Databricks reranker is disabled in configuration")
            return None

        # Reload environment variables to ensure we have the latest values
        root_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        env_path = root_dir / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.debug(f"Reloaded environment variables from: {env_path}")

        # Get token from environment variables only
        databricks_token = os.environ.get("DATABRICKS_API_KEY")

        # Add detailed debugging
        logger.debug(f"Reranker token retrieval - USE_DATABRICKS_RERANKER: {USE_DATABRICKS_RERANKER}")
        logger.debug(f"Reranker token retrieval - Token found: {'Yes' if databricks_token else 'No'}")
        if databricks_token:
            logger.debug(f"Reranker token retrieval - Token starts with: {databricks_token[:4]}...")

        if not databricks_token:
            logger.error("DATABRICKS_API_KEY not found in environment variables for reranker")
            # Only show Streamlit error if we're in a Streamlit context
            try:
                st.error("Databricks API token not found. Please add DATABRICKS_API_KEY to your .env file")
            except:
                # Not in Streamlit context, just log the error
                pass
            return None

        return databricks_token
    except Exception as e:
        logger.error(f"Error getting Databricks token: {e}")
        return None


class DatabricksRerankerModel:
    """
    Wrapper class for Databricks reranker model to provide a similar interface
    to CrossEncoder for easy integration with existing code.

    This class wraps the Databricks reranker API to make it compatible with
    the CrossEncoder interface used in the application. It maintains
    the same method signatures to ensure compatibility with existing code.
    """

    def __init__(self, max_length: int = None):
        """
        Initialize the Databricks reranker model.

        Args:
            max_length: Maximum token length for inputs to the model
                        If None, uses RERANKER_MAX_TOKENS from config (default: 512)
        """
        self.endpoint_url = DATABRICKS_RERANKER_ENDPOINT
        self.model_name = DATABRICKS_RERANKER_MODEL_NAME
        self.max_length = max_length if max_length is not None else RERANKER_MAX_TOKENS
        # Load the tokenizer once per instance
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L12-v2')

    def _truncate_text_pair(self, query: str, document: str) -> Tuple[str, str]:
        """
        Truncate a query-document pair to fit within the model's maximum token length using the BERT tokenizer.
        Always include at least 1 token from the document, truncating the query as needed. Logs truncation details.
        """
        max_length = self.max_length
        margin = 8  # Small margin to avoid edge-case overflows
        max_length_with_margin = max_length - margin
        special_tokens_count = 3

        # Tokenize query and document separately
        query_tokens = self.tokenizer.tokenize(query)
        document_tokens = self.tokenizer.tokenize(document)

        # Always reserve at least 1 token for the document
        min_doc_tokens = 1
        max_query_tokens = max_length_with_margin - special_tokens_count - min_doc_tokens

        if len(query_tokens) > max_query_tokens:
            truncated_query_tokens = query_tokens[:max_query_tokens]
            logger.warning(f"Query truncated from {len(query_tokens)} to {len(truncated_query_tokens)} tokens (BERT tokens)")
        else:
            truncated_query_tokens = query_tokens

        # Now allocate the rest to the document
        remaining_tokens = max_length_with_margin - special_tokens_count - len(truncated_query_tokens)
        truncated_document_tokens = document_tokens[:max(remaining_tokens, min_doc_tokens)]
        if len(document_tokens) > len(truncated_document_tokens):
            logger.warning(f"Document truncated from {len(document_tokens)} to {len(truncated_document_tokens)} tokens (BERT tokens)")

        # Detokenize
        truncated_query = self.tokenizer.convert_tokens_to_string(truncated_query_tokens)
        truncated_document = self.tokenizer.convert_tokens_to_string(truncated_document_tokens)
        return truncated_query, truncated_document

    def _truncate_by_chars(self, text: str, max_chars: int = 1500) -> str:
        """
        Fallback method to truncate text by character count when tokenizer is unavailable.

        Args:
            text: The text to truncate
            max_chars: Maximum number of characters

        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text

        # Simple truncation - take the first max_chars characters
        truncated = text[:max_chars]

        # Try to truncate at a sentence or word boundary if possible
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # Only use period if it's not too far back
            return truncated[:last_period + 1]

        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space]

        return truncated

    def predict(self, sentence_pairs: List[List[str]]) -> np.ndarray:
        """
        Predict the relevance scores for a list of sentence pairs.

        Args:
            sentence_pairs: List of [query, document] pairs

        Returns:
            numpy array of scores
        """
        # Get a fresh token each time to avoid caching issues
        token = get_databricks_reranker_token()
        if token is None:
            logger.error("Cannot predict: Databricks token not available")
            return np.zeros(len(sentence_pairs))

        try:
            # Convert sentence pairs to DataFrame format expected by the model
            data = []
            for query, document in sentence_pairs:
                # Truncate both query and document to fit within model's token limit
                truncated_query, truncated_document = self._truncate_text_pair(query, document)

                data.append({
                    'text': truncated_query,
                    'text_pair': truncated_document
                })

            df = pd.DataFrame(data)

            # Create the request payload
            ds_dict = {'dataframe_split': df.to_dict(orient='split')}
            data_json = json.dumps(ds_dict, allow_nan=True)

            # Set up headers
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            # Make the API call with timeout
            logger.info(f"Calling Databricks reranker API with {len(sentence_pairs)} pairs")
            response = requests.post(
                url=self.endpoint_url,
                headers=headers,
                data=data_json,
                timeout=RERANKER_API_TIMEOUT
            )

            # Check for errors
            if response.status_code != 200:
                logger.error(f"Databricks reranker API error: {response.status_code}, {response.text}")
                return np.zeros(len(sentence_pairs))

            # Parse the response
            result = response.json()

            # Extract scores from the response
            if 'predictions' in result:
                predictions = result['predictions']

                # Handle the case where predictions is a list of dictionaries with 'score' field
                if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
                    logger.debug(f"Databricks reranker returned predictions as list of dictionaries")

                    # Extract just the scores from the dictionaries
                    if all('score' in p for p in predictions):
                        scores = np.array([float(p['score']) for p in predictions])
                        logger.info(f"Databricks reranker returned {len(scores)} scores")
                        return scores
                    else:
                        logger.error("Error: 'score' field not found in all prediction items")
                        # Try to extract whatever we can
                        scores = []
                        for i, p in enumerate(predictions):
                            if 'score' in p:
                                scores.append(float(p['score']))
                            else:
                                logger.warning(f"No score in prediction {i}: {p}")
                                scores.append(0.0)
                        return np.array(scores)
                else:
                    # Try to handle as a simple list of scores
                    try:
                        scores = np.array([float(score) for score in predictions])
                        logger.info(f"Databricks reranker returned {len(scores)} scores")
                        return scores
                    except Exception as e:
                        logger.error(f"Error converting predictions to scores: {e}")
                        return np.zeros(len(sentence_pairs))
            else:
                logger.error(f"Unexpected response format. 'predictions' key not found. Keys: {list(result.keys())}")
                return np.zeros(len(sentence_pairs))

        except requests.exceptions.Timeout:
            logger.error(f"Databricks reranker API timeout after {RERANKER_API_TIMEOUT} seconds")
            return np.zeros(len(sentence_pairs))
        except requests.exceptions.RequestException as e:
            logger.error(f"Databricks reranker API request error: {e}")
            return np.zeros(len(sentence_pairs))
        except Exception as e:
            logger.error(f"Error in Databricks reranker prediction: {e}")
            return np.zeros(len(sentence_pairs))


@st.cache_resource
def load_databricks_reranker_model() -> Optional[object]:
    """
    Loads and caches the Databricks reranker model with timeout and fallback support.

    The model has a maximum context window of 512 tokens, so inputs will be
    automatically truncated if they exceed this limit. The truncation is done
    by the DatabricksRerankerModel class using a BERT tokenizer to accurately
    count tokens and ensure the combined length of query and document doesn't
    exceed the model's maximum token length.

    If the Databricks reranker API fails at startup (timeout, 403 error, etc.),
    this function will automatically fall back to the LLM-based reranker.

    Returns:
        DatabricksRerankerModel or LLMRerankerModel instance, or None if both fail
    """
    try:
        logger.info(f"Loading Databricks reranker model: {DATABRICKS_RERANKER_MODEL_NAME}")
        logger.info(f"Using max token length: {RERANKER_MAX_TOKENS}")
        logger.info(f"API timeout set to: {RERANKER_API_TIMEOUT} seconds")

        model = DatabricksRerankerModel()  # Uses RERANKER_MAX_TOKENS from config

        # Test the model with a simple input to verify API connection
        # This test will timeout after RERANKER_API_TIMEOUT seconds
        test_pairs = [["How many people live in Berlin?", "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."]]

        try:
            logger.info("Testing Databricks reranker API connection...")
            test_scores = model.predict(test_pairs)

            if isinstance(test_scores, np.ndarray) and test_scores.shape[0] > 0:
                # Check if we got a zero score (which indicates an API error)
                if test_scores[0] == 0.0:
                    logger.warning("Databricks reranker returned zero score, likely due to API error")
                    raise ValueError("Databricks reranker API returned error response")

                # Check if the score is a reasonable value (should be between 0 and 1)
                if 0 <= test_scores[0] <= 1:
                    logger.info(f"✓ Databricks reranker model loaded successfully with score: {test_scores[0]:.4f}")
                    return model
                else:
                    logger.warning(f"Databricks reranker returned an unusual score: {test_scores[0]}")
                    # Still return the model as it might be a valid score outside the expected range
                    return model
            else:
                logger.error(f"Databricks reranker model test failed: Invalid score format: {type(test_scores)}")
                raise ValueError("Invalid score format from Databricks reranker")

        except requests.exceptions.Timeout:
            logger.error(f"✗ Databricks reranker API timeout after {RERANKER_API_TIMEOUT} seconds at startup")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Databricks reranker API request error at startup: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Databricks reranker model test failed with error: {e}")
            raise

    except Exception as e:
        logger.error(f"Failed to load Databricks reranker model: {e}")

        # Check if fallback is enabled
        if not ENABLE_LLM_RERANKER_FALLBACK:
            logger.warning("LLM-based fallback reranker is disabled in configuration")
            logger.info("Set ENABLE_LLM_RERANKER_FALLBACK=True in config.py to enable fallback")
            return None

        logger.warning("Attempting to load LLM-based fallback reranker...")

        # Try to load the LLM-based fallback reranker
        try:
            from .llm_reranker import load_llm_reranker_model
            fallback_model = load_llm_reranker_model()

            if fallback_model:
                logger.info("✓ Successfully loaded LLM-based fallback reranker")
                return fallback_model
            else:
                logger.error("✗ Failed to load LLM-based fallback reranker")
                return None

        except Exception as fallback_error:
            logger.error(f"✗ Error loading LLM-based fallback reranker: {fallback_error}")
            return None
