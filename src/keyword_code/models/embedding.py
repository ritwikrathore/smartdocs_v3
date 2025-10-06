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
    """
    Loads the reranker model (Databricks API or LLM fallback) and returns it, or None if not available.

    This function will first attempt to load the Databricks reranker API. If that fails
    (due to timeout, 403 error, or other issues), it will automatically fall back to
    the LLM-based reranker.

    Returns:
        Reranker model instance (DatabricksRerankerModel or LLMRerankerModel) or None
    """
    from ..config import USE_DATABRICKS_RERANKER
    from ..config import logger
    model = None
    if USE_DATABRICKS_RERANKER:
        try:
            from .databricks_reranker import load_databricks_reranker_model
            logger.info("=" * 60)
            logger.info("RERANKER INITIALIZATION")
            logger.info("=" * 60)
            logger.info("Attempting to load Databricks reranker model...")

            model = load_databricks_reranker_model()

            if model:
                # Check which type of model was loaded
                model_type = type(model).__name__
                if model_type == "DatabricksRerankerModel":
                    logger.info("✓ Databricks reranker API loaded successfully")
                elif model_type == "LLMRerankerModel":
                    logger.info("✓ LLM-based fallback reranker loaded successfully")
                else:
                    logger.info(f"✓ Reranker loaded successfully (type: {model_type})")
                logger.info("=" * 60)
                return model
            else:
                logger.error("✗ Failed to load any reranker model. Reranking will be disabled.")
                logger.info("=" * 60)
                return None
        except Exception as e:
            logger.error(f"✗ Error loading reranker model: {e}", exc_info=True)
            logger.error("Reranking will be disabled.")
            logger.info("=" * 60)
            return None
    else:
        logger.info("Databricks reranker is disabled in configuration. Reranking will be disabled.")
        return None



