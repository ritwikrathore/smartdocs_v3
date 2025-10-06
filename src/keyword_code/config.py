"""
Configuration settings for the keyword_code package.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Get the project root directory
root_dir = Path(__file__).parent.parent.parent  # This should point to the project root
env_path = root_dir / '.env'

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Temporarily set to DEBUG to help diagnose reranker token issue
    format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"  # Added threadName
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file with explicit path
load_dotenv(dotenv_path=env_path)

# Verify if DATABRICKS_API_KEY is loaded
databricks_token = os.environ.get("DATABRICKS_API_KEY")
if databricks_token:
    logger.info("DATABRICKS_API_KEY loaded successfully")
    # Don't log the full token for security reasons
    logger.info(f"Token starts with: {databricks_token[:4]}...")
else:
    logger.error("DATABRICKS_API_KEY not found in environment variables after load_dotenv()")

# --- Worker Configuration ---
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "true").lower() == "true"

# --- RAG Configuration ---
FUZZY_MATCH_THRESHOLD = 85  # Lowered threshold (0-100) to better handle quotation mark differences
RAG_TOP_K = 15  # Number of relevant chunks to retrieve per sub-prompt
RAG_WORKERS = 4  # Number of workers for parallel RAG processing

# --- Chunking Configuration ---
# Sentence chunker parameters
SENTENCES_PER_CHUNK = 6  # Number of sentences per chunk
MIN_CHUNK_CHAR_LENGTH = 50  # Minimum character length for a chunk to be valid

# --- Model Paths ---
# Using Databricks for all models, no local models needed
# RERANKER_MODEL_PATH is kept for backward compatibility but not used anymore
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "src/keyword_code/reranking_model_local")

# --- Interaction Logging Configuration ---
# Set to True to enable detailed logging of BM25, semantic search, reranker, and LLM interactions
ENABLE_INTERACTION_LOGGING = True  # Disabled by default

# --- Databricks Models ---
# Configuration for Databricks services
USE_DATABRICKS_EMBEDDING = True  # Use Databricks for embeddings
USE_DATABRICKS_LLM = True  # Use Databricks for LLM
USE_DATABRICKS_RERANKER = True  # Use Databricks for reranking

# --- Reranker Configuration ---
# The Databricks reranker model has a maximum context window of 512 tokens
# Inputs longer than this will be automatically truncated
RERANKER_MAX_TOKENS = 512  # Maximum token length for the reranker model

# --- LLM Configuration ---
# Using Databricks LLM
DECOMPOSITION_MODEL_NAME = "databricks-llama-4-maverick"  # Databricks model name
ANALYSIS_MODEL_NAME = "databricks-llama-4-maverick"  # Databricks model name

# --- UI Configuration ---
# Define primary colors
PROCESS_CYAN = "#00ADE4"
DARK_BLUE = "#002345"
LIGHT_BLUE_TINT = "#E6F7FD"  # Example tint (adjust as needed)
VERY_LIGHT_GRAY = "#FAFAFA"
