"""
Databricks embedding model integration.

This module provides a wrapper around the Databricks embedding API to provide
a consistent embedding interface for the application.

The client is cached using Streamlit's cache_resource decorator to avoid
recreating it on each Streamlit rerun, which is a performance optimization.
"""

import os
import numpy as np
import streamlit as st
from openai import OpenAI
from typing import List, Union
from ..config import logger, USE_DATABRICKS_EMBEDDING

# Databricks endpoint URL - Make sure this exactly matches what worked in your test scripts
DATABRICKS_BASE_URL = "https://adb-3858882779799477.17.azuredatabricks.net/serving-endpoints"
DATABRICKS_MODEL_NAME = "databricks-gte-large-en"

# Debug flag to print detailed API call information
DEBUG_API_CALLS = True


@st.cache_resource
def get_databricks_client():
    """
    Creates and caches an OpenAI client configured for Databricks.

    This function is cached using Streamlit's cache_resource decorator to avoid
    recreating the client on each Streamlit rerun, which is a performance optimization.
    The caching is important even though we're making API calls because:
    1. It avoids recreating the client object on each rerun
    2. It allows us to reuse the same client across multiple API calls
    3. It ensures we only validate the token once per session

    Returns:
        OpenAI client or None if token is not available
    """
    try:
        # Check if Databricks embedding is enabled
        if not USE_DATABRICKS_EMBEDDING:
            logger.info("Databricks embedding is disabled in configuration")
            return None

        # Get token from environment variables only
        databricks_token = os.environ.get("DATABRICKS_API_KEY")

        # Debug logging for token retrieval
        if databricks_token:
            logger.info("DATABRICKS_API_KEY found in environment variables")
            # Don't log the full token for security reasons
            logger.info(f"Token starts with: {databricks_token[:4]}...")
        else:
            logger.error("DATABRICKS_API_KEY not found in environment variables")

            # Try to get token from Streamlit secrets as a fallback (for compatibility)
            try:
                if hasattr(st, "secrets") and "DATABRICKS_API_KEY" in st.secrets:
                    databricks_token = st.secrets["DATABRICKS_API_KEY"]
                    logger.info("DATABRICKS_API_KEY found in Streamlit secrets")
                    # Don't log the full token for security reasons
                    logger.info(f"Token from secrets starts with: {databricks_token[:4]}...")
                else:
                    logger.error("DATABRICKS_API_KEY not found in Streamlit secrets either")
            except Exception as secret_error:
                logger.error(f"Error accessing Streamlit secrets: {secret_error}")

            if not databricks_token:
                st.error("Databricks API token not found. Please add DATABRICKS_API_KEY to your .env file")
                return None

        # Create OpenAI client with Databricks configuration
        logger.info(f"Creating OpenAI client with Databricks base URL: {DATABRICKS_BASE_URL}")

        # Clean the token to ensure there are no whitespace or quotes
        clean_token = databricks_token.strip().replace('"', '').replace("'", "")

        if DEBUG_API_CALLS:
            logger.info(f"Original token starts with: {databricks_token[:4]}...")
            logger.info(f"Cleaned token starts with: {clean_token[:4]}...")

        client = OpenAI(
            api_key=clean_token,
            base_url=DATABRICKS_BASE_URL
        )
        logger.info("Databricks OpenAI client created successfully")

        # Test the client with a simple API call to verify it works
        try:
            logger.info("Testing Databricks client with a simple API call...")
            # This is a minimal API call to test if the client is working
            test_response = client.embeddings.create(
                input=["Test sentence"],
                model=DATABRICKS_MODEL_NAME
            )
            # Check if we got a valid response
            if test_response and hasattr(test_response, 'data') and len(test_response.data) > 0:
                logger.info(f"Databricks client test successful! Received embedding with {len(test_response.data[0].embedding)} dimensions")
            else:
                logger.info("Databricks client test successful but received unexpected response format")
            return client
        except Exception as api_error:
            logger.error(f"Error testing Databricks client API call: {api_error}")
            st.error(f"Databricks API test failed: {api_error}")
            return None

    except Exception as e:
        logger.error(f"Error creating Databricks client: {e}")
        st.error(f"Failed to initialize Databricks client: {e}")
        return None


class DatabricksEmbeddingModel:
    """
    Wrapper class for Databricks embedding model to provide a consistent
    embedding interface for the application.

    This class wraps the Databricks embedding API and provides standard
    embedding methods like encode() for easy integration with existing code.
    """

    # Class-level cache to persist across instances
    _embedding_cache = {}

    def __init__(self):
        self.client = get_databricks_client()
        self.model_name = DATABRICKS_MODEL_NAME
        self.embedding_dimension = 1024  # GTE-large has 1024 dimensions
        # Use the class-level cache
        self.embedding_cache = DatabricksEmbeddingModel._embedding_cache

    def encode(self,
               sentences: Union[str, List[str]],
               convert_to_tensor: bool = True,
               # Additional parameters for compatibility
               show_progress_bar: bool = False,
               batch_size: int = 100) -> Union[List[List[float]], np.ndarray]:
        """
        Generate embeddings for the given sentences using Databricks API.

        Args:
            sentences: String or list of strings to encode
            convert_to_tensor: If True, returns a numpy array (ignored if client is None)
            show_progress_bar: Ignored, kept for compatibility
            batch_size: Size of batches to process, default 100

        Returns:
            List of embeddings or numpy array if convert_to_tensor=True
        """
        if self.client is None:
            logger.error("Cannot encode text: Databricks client not initialized")
            # Return zero embeddings as fallback
            if isinstance(sentences, str):
                return np.zeros(self.embedding_dimension) if convert_to_tensor else [0.0] * self.embedding_dimension
            else:
                empty_embeddings = [np.zeros(self.embedding_dimension) for _ in range(len(sentences))]
                return np.array(empty_embeddings) if convert_to_tensor else [[0.0] * self.embedding_dimension for _ in range(len(sentences))]

        try:
            # Handle single string vs list of strings
            is_single_sentence = isinstance(sentences, str)
            input_texts = [sentences] if is_single_sentence else sentences

            # Process in batches if needed (OpenAI API may have limits)
            # Using the batch_size parameter passed to the function
            all_embeddings = []

            # Ignore show_progress_bar parameter to avoid IDE warning
            _ = show_progress_bar

            # Use a smaller batch size for API calls to reduce timeouts and improve reliability
            api_batch_size = min(batch_size, 32)  # Limit to 32 items per API call

            # Cache embeddings to avoid redundant API calls
            # This is especially helpful when processing the same text multiple times
            embedding_cache = {}

            for i in range(0, len(input_texts), api_batch_size):
                batch = input_texts[i:i+api_batch_size]

                # Check cache for each text in the batch
                uncached_texts = []
                uncached_indices = []
                batch_results = [None] * len(batch)

                for j, text in enumerate(batch):
                    # Use a hash of the text as the cache key
                    text_hash = hash(text)
                    if text_hash in embedding_cache:
                        batch_results[j] = embedding_cache[text_hash]
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)

                # Only make API call if there are uncached texts
                if uncached_texts:
                    try:
                        # Get a fresh token directly from environment for this call
                        # This is to ensure we're using the most up-to-date token
                        current_token = os.environ.get("DATABRICKS_API_KEY")

                        if DEBUG_API_CALLS:
                            logger.info(f"Making API call to Databricks embedding endpoint")
                            logger.info(f"Base URL: {DATABRICKS_BASE_URL}")
                            logger.info(f"Model: {self.model_name}")
                            logger.info(f"Number of texts: {len(uncached_texts)}")
                            logger.info(f"Token available: {'Yes' if current_token else 'No'}")
                            if current_token:
                                logger.info(f"Token starts with: {current_token[:4]}...")

                        # Make the API call
                        response = self.client.embeddings.create(
                            input=uncached_texts,
                            model=self.model_name
                        )

                        # Extract embeddings from response
                        api_embeddings = [item.embedding for item in response.data]

                        if DEBUG_API_CALLS:
                            logger.info(f"API call successful, received {len(api_embeddings)} embeddings")

                        # Update cache and batch results
                        for j, embedding in enumerate(api_embeddings):
                            text_idx = uncached_indices[j]
                            text_hash = hash(batch[text_idx])
                            embedding_cache[text_hash] = embedding
                            batch_results[text_idx] = embedding
                    except Exception as e:
                        logger.error(f"Error in API call for batch {i//api_batch_size}: {e}")
                        # Try to get more detailed error information
                        if hasattr(e, 'response') and hasattr(e.response, 'text'):
                            logger.error(f"Response text: {e.response.text}")
                        # Fill in missing embeddings with zeros
                        for j in uncached_indices:
                            batch_results[j] = [0.0] * self.embedding_dimension

                all_embeddings.extend(batch_results)

            # Return appropriate format
            if is_single_sentence:
                result = all_embeddings[0]
                return np.array(result) if convert_to_tensor else result
            else:
                return np.array(all_embeddings) if convert_to_tensor else all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            if isinstance(sentences, str):
                return np.zeros(self.embedding_dimension) if convert_to_tensor else [0.0] * self.embedding_dimension
            else:
                empty_embeddings = [np.zeros(self.embedding_dimension) for _ in range(len(sentences))]
                return np.array(empty_embeddings) if convert_to_tensor else [[0.0] * self.embedding_dimension for _ in range(len(sentences))]


@st.cache_resource
def load_databricks_embedding_model():
    """
    Loads and caches the Databricks embedding model.

    This function is cached using Streamlit's cache_resource decorator to avoid
    recreating the model on each Streamlit rerun, which is a performance optimization.
    The caching is important even though we're making API calls because:
    1. It avoids recreating the model wrapper object on each rerun
    2. It allows us to test the API connection once and reuse the validated model
    3. It ensures consistent behavior across the application

    Returns:
        DatabricksEmbeddingModel instance or None if initialization fails
    """
    try:
        logger.info(f"Loading Databricks embedding model: {DATABRICKS_MODEL_NAME}")

        # Check if the token is available before creating the model
        databricks_token = os.environ.get("DATABRICKS_API_KEY")
        if not databricks_token:
            logger.error("Cannot load Databricks embedding model: DATABRICKS_API_KEY not found in environment variables")
            st.error("Databricks API token not found. Please add DATABRICKS_API_KEY to your .env file")
            return None

        # Create the model
        model = DatabricksEmbeddingModel()

        # Check if the client was created successfully
        if model.client is None:
            logger.error("Databricks embedding model initialization failed: Client is None")
            return None

        # Test the model with a simple input to verify API connection
        logger.info("Testing Databricks embedding model with a sample sentence...")
        test_embedding = model.encode("Test sentence", convert_to_tensor=True)

        if isinstance(test_embedding, np.ndarray) and test_embedding.shape[0] > 0:
            logger.info(f"Databricks embedding model loaded successfully. Embedding dimension: {test_embedding.shape[0]}")
            return model
        else:
            logger.error("Databricks embedding model test failed: Invalid embedding format")
            st.error("Databricks embedding model test failed: Invalid embedding format")
            return None
    except Exception as e:
        logger.error(f"Error loading Databricks embedding model: {e}")
        st.error(f"Error loading Databricks embedding model: {e}")
        return None
