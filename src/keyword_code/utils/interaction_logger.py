"""
Interaction logger for debugging and analysis.

This module provides functions to log various interactions in the RAG pipeline:
- BM25 search results
- Semantic search results
- Reranker classifications
- LLM prompts and responses

All logs are written to a file for later analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import the configuration
from ..config import ENABLE_INTERACTION_LOGGING

# Create a dedicated logger for interactions
interaction_logger = logging.getLogger("interaction_logger")
interaction_logger.setLevel(logging.DEBUG)

# Flag to enable/disable interaction logging - initialized from config
INTERACTION_LOGGING_ENABLED = ENABLE_INTERACTION_LOGGING
# Log file path
INTERACTION_LOG_FILE = None
# File handler for the interaction logger
_file_handler = None


def setup_interaction_logging(log_file_path: str = None) -> None:
    """
    Set up interaction logging to a file.

    Args:
        log_file_path: Path to the log file. If None, a default path will be used.
    """
    global INTERACTION_LOGGING_ENABLED, INTERACTION_LOG_FILE, _file_handler

    # If already set up, remove existing handler
    if _file_handler is not None:
        interaction_logger.removeHandler(_file_handler)
        _file_handler = None

    # Create logs directory if it doesn't exist
    if log_file_path is None:
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(logs_dir, f"rag_interactions_{timestamp}.log")

    # Create file handler
    _file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    _file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _file_handler.setFormatter(formatter)

    # Add handler to logger
    interaction_logger.addHandler(_file_handler)

    # Enable logging
    INTERACTION_LOGGING_ENABLED = True
    INTERACTION_LOG_FILE = log_file_path

    interaction_logger.info(f"Interaction logging enabled. Log file: {log_file_path}")


def disable_interaction_logging() -> None:
    """Disable interaction logging."""
    global INTERACTION_LOGGING_ENABLED, _file_handler

    if _file_handler is not None:
        interaction_logger.removeHandler(_file_handler)
        _file_handler = None

    INTERACTION_LOGGING_ENABLED = False
    interaction_logger.info("Interaction logging disabled.")


def log_bm25_results(prompt: str, results: List[Tuple[int, float]], chunks: List[Dict[str, Any]]) -> None:
    """
    Log BM25 search results.

    Args:
        prompt: The search prompt
        results: List of (chunk_index, score) tuples
        chunks: The original chunks list
    """
    if not INTERACTION_LOGGING_ENABLED:
        return

    try:
        # Format results for logging
        formatted_results = []
        for chunk_idx, score in results:
            chunk = chunks[chunk_idx]
            formatted_results.append({
                "chunk_id": chunk.get("chunk_id", f"unknown_{chunk_idx}"),
                "page_num": chunk.get("page_num", -1),
                "score": float(score),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            })

        # Log as JSON
        log_entry = {
            "type": "bm25_results",
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "num_results": len(results),
            "results": formatted_results
        }

        interaction_logger.debug(f"BM25_RESULTS: {json.dumps(log_entry)}")
    except Exception as e:
        interaction_logger.error(f"Error logging BM25 results: {e}")


def log_semantic_search_results(prompt: str, indices: Set[int], chunks: List[Dict[str, Any]], scores: Optional[List[float]] = None) -> None:
    """
    Log semantic search results.

    Args:
        prompt: The search prompt
        indices: Set of chunk indices
        chunks: The original chunks list
        scores: Optional list of scores corresponding to indices
    """
    if not INTERACTION_LOGGING_ENABLED:
        return

    try:
        # Format results for logging
        formatted_results = []
        for i, chunk_idx in enumerate(indices):
            chunk = chunks[chunk_idx]
            result = {
                "chunk_id": chunk.get("chunk_id", f"unknown_{chunk_idx}"),
                "page_num": chunk.get("page_num", -1),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            }

            # Add score if available
            if scores is not None and i < len(scores):
                result["score"] = float(scores[i])

            formatted_results.append(result)

        # Log as JSON
        log_entry = {
            "type": "semantic_search_results",
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "num_results": len(indices),
            "results": formatted_results
        }

        interaction_logger.debug(f"SEMANTIC_RESULTS: {json.dumps(log_entry)}")
    except Exception as e:
        interaction_logger.error(f"Error logging semantic search results: {e}")


def log_reranker_results(prompt: str, results: List[Dict[str, Any]]) -> None:
    """
    Log reranker results.

    Args:
        prompt: The search prompt
        results: List of reranked chunk dictionaries with scores
    """
    if not INTERACTION_LOGGING_ENABLED:
        return

    try:
        # Format results for logging
        formatted_results = []
        for chunk in results:
            formatted_results.append({
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "score": float(chunk.get("score", 0)),
                "retrieval_method": chunk.get("retrieval_method", "unknown"),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            })

        # Log as JSON
        log_entry = {
            "type": "reranker_results",
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "num_results": len(results),
            "results": formatted_results
        }

        interaction_logger.debug(f"RERANKER_RESULTS: {json.dumps(log_entry)}")
    except Exception as e:
        interaction_logger.error(f"Error logging reranker results: {e}")


def log_llm_ranking_results(prompt: str, results: List[Dict[str, Any]]) -> None:
    """
    Log LLM ranking results.

    Args:
        prompt: The search prompt
        results: List of LLM-ranked chunk dictionaries with scores
    """
    if not INTERACTION_LOGGING_ENABLED:
        return

    try:
        # Format results for logging
        formatted_results = []
        for chunk in results:
            formatted_results.append({
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "score": float(chunk.get("score", 0)),
                "retrieval_method": chunk.get("retrieval_method", "llm_ranking"),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            })

        # Log as JSON
        log_entry = {
            "type": "llm_ranking_results",
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "num_results": len(results),
            "results": formatted_results
        }

        interaction_logger.debug(f"LLM_RANKING_RESULTS: {json.dumps(log_entry)}")
    except Exception as e:
        interaction_logger.error(f"Error logging LLM ranking results: {e}")


def log_llm_interaction(messages: List[Dict[str, str]], response: str, interaction_type: str = "general") -> None:
    """
    Log LLM prompt and response.

    Args:
        messages: The messages sent to the LLM
        response: The response from the LLM
        interaction_type: Type of interaction (e.g., "analysis", "decomposition", "chat")
    """
    if not INTERACTION_LOGGING_ENABLED:
        return

    try:
        # Log as JSON
        log_entry = {
            "type": "llm_interaction",
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "response": response
        }

        interaction_logger.debug(f"LLM_INTERACTION: {json.dumps(log_entry)}")
    except Exception as e:
        interaction_logger.error(f"Error logging LLM interaction: {e}")
