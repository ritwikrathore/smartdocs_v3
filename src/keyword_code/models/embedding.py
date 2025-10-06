"""
Embedding model loading and management.
"""

import streamlit as st
from ..config import logger

# Import Databricks embedding model
from .databricks_embedding import load_databricks_embedding_model


@st.cache_resource  # Use cache_resource for non-data objects like models
def load_embedding_model():
    """
    Loads the Databricks embedding model and caches it.

    Note: This function is cached using Streamlit's cache_resource decorator
    to avoid recreating the client on each Streamlit rerun, which is a
    performance optimization.
    """
    logger.info("Loading Databricks embedding model...")
    databricks_model = load_databricks_embedding_model()

    if databricks_model is not None:
        logger.info("Successfully loaded Databricks embedding model")
        return databricks_model
    else:
        error_msg = "Failed to load Databricks embedding model. Please check your DATABRICKS_API_KEY in .env file."
        logger.error(error_msg)
        st.error(f"Fatal Error: {error_msg}")
        return None

# --- Load Reranker Model (Shared) ---
def load_reranker_model():
    """Loads the Databricks reranker model and returns it, or None if not available."""
    from ..config import USE_DATABRICKS_RERANKER
    from ..config import logger
    model = None
    if USE_DATABRICKS_RERANKER:
        try:
            from .databricks_reranker import load_databricks_reranker_model
            logger.info("Loading Databricks reranker model...")
            model = load_databricks_reranker_model()
            if model:
                logger.info("Databricks reranker model loaded successfully.")
                return model
            else:
                logger.error("Failed to load Databricks reranker model. Reranking will be disabled.")
                return None
        except Exception as e:
            logger.error(f"Error loading Databricks reranker model: {e}", exc_info=True)
            logger.error("Reranking will be disabled.")
            return None
    else:
        logger.info("Databricks reranker is disabled in configuration. Reranking will be disabled.")
        return None



