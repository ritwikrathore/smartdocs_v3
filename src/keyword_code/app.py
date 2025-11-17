"""
Main application file for the keyword_code package.
This is the entry point for the Streamlit application.

This file has been refactored to be more modular, with functionality split into
separate modules for better maintainability and readability.
"""

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import re
import tempfile
import threading
import queue
import atexit
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber  # For PDF viewer rendering - Can likely be removed if fitz rendering is stable
import streamlit as st
from docx import Document as DocxDocument  # Renamed to avoid conflict
from docx import Document # Import for Word export
from docx.shared import Pt, RGBColor # Import for Word export styling
from docx.enum.text import WD_ALIGN_PARAGRAPH # Import for Word export alignment
from dotenv import load_dotenv
from thefuzz import fuzz
import openai  # Used for Databricks API integration
import zipfile
import urllib.parse
import streamlit_pills as stp # For clickable prompt suggestions

# No longer using CrossEncoder for reranking, using Databricks reranker instead

# Import from our modules
from .config import (
    logger, MAX_WORKERS, ENABLE_PARALLEL, RAG_TOP_K, RERANKER_MODEL_PATH,
    USE_DATABRICKS_RERANKER, ENABLE_INTERACTION_LOGGING, SAVED_PROMPTS
)
from .utils.helpers import get_base64_encoded_image, normalize_text, remove_markdown_formatting
from .utils.async_utils import run_async
from .utils.ui_helpers import (
    apply_ui_styling, render_branding, initialize_session_state,
    display_welcome_features, clear_session_for_new_query, clear_incompatible_embeddings
)
from .utils.file_manager import (
    create_temp_file, create_temp_dir, remove_temp_file, remove_temp_dir,
    cleanup_session_files, cleanup_all_temp_files, create_session_temp_file,
    get_session_id, update_session_access, cleanup_expired_sessions
)
from .utils.memory_monitor import (
    monitor_memory_usage, cleanup_memory, get_memory_usage, format_bytes
)
from .utils.interaction_logger import (
    setup_interaction_logging, disable_interaction_logging, INTERACTION_LOGGING_ENABLED,
    log_rag_parameters
)
from .utils.langfuse_tracing import (
    optional_context,
    set_span_output,
    start_span,
    start_trace,
    update_current_trace,
    update_current_span,
    get_langfuse_client_cached,
)
from .models.embedding import load_embedding_model, load_reranker_model
from .processors.pdf_processor import PDFProcessor
from .processors.word_processor import WordProcessor
from .rag.retrieval import retrieve_relevant_chunks, retrieve_relevant_chunks_for_chat
from .ai.analyzer import DocumentAnalyzer
from .ai.chat import generate_chat_response
from .services.keyword_mode_service import KeywordModeService


def get_embedding_model():
    """Lazily load and return the shared embedding model."""
    return load_embedding_model()


@st.cache_resource(show_spinner="Calibrating the reranker ⚙️…")
def get_reranker_model():
    """Load the reranker model on demand and cache the instance."""
    return load_reranker_model()


@st.cache_resource(show_spinner="Preparing keyword highlighter ✨…")
def get_keyword_mode_service():
    """Return a cached KeywordModeService instance."""
    return KeywordModeService()


_interaction_logging_initialized = False


def ensure_interaction_logging():
    """Enable detailed interaction logging once, when first needed."""
    global _interaction_logging_initialized
    if _interaction_logging_initialized:
        return
    if ENABLE_INTERACTION_LOGGING:
        enable_interaction_logging()
        _interaction_logging_initialized = True


_CHUNK_TRACE_SKIP_METADATA_KEYS = {
    "embedding",
    "chunk_embedding",
    "tokens",
    "raw_text",
}


def _truncate_chunk_text(text: Optional[str]) -> str:
    return text if isinstance(text, str) else ""


def _coerce_chunk_metadata_value(value: Any, *, max_length: int = 500, max_items: int = 25) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + "..."
        return value

    if isinstance(value, list):
        items = [
            _coerce_chunk_metadata_value(item, max_length=max_length, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append("...")
        return items

    if isinstance(value, tuple):
        items = [
            _coerce_chunk_metadata_value(item, max_length=max_length, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append("...")
        return items

    if isinstance(value, set):
        items = list(value)[:max_items]
        coerced = [
            _coerce_chunk_metadata_value(item, max_length=max_length, max_items=max_items)
            for item in items
        ]
        if len(value) > max_items:
            coerced.append("...")
        return coerced

    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_items:
                sanitized["..."] = f"{len(value) - max_items} more keys"
                break
            sanitized[str(key)] = _coerce_chunk_metadata_value(
                val,
                max_length=max_length,
                max_items=max_items,
            )
        return sanitized

    value_str = str(value)
    if len(value_str) > max_length:
        return value_str[:max_length] + "..."
    return value_str


def _build_chunk_trace_payload(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata")

        chunk_entry: Dict[str, Any] = {
            "index": index,
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type"),
            "page_num": chunk.get("page_num"),
            "page_label": chunk.get("page_label"),
            "char_count": len(text) if isinstance(text, str) else None,
            "text": _truncate_chunk_text(text),
        }

        if isinstance(metadata, dict):
            sanitized_metadata: Dict[str, Any] = {}
            for key, value in metadata.items():
                if key in _CHUNK_TRACE_SKIP_METADATA_KEYS:
                    continue
                sanitized_metadata[str(key)] = _coerce_chunk_metadata_value(value)
            if sanitized_metadata:
                chunk_entry["metadata"] = sanitized_metadata

        payload.append(chunk_entry)

    return payload


def _record_chunking_trace(
    *,
    chunks: Sequence[Dict[str, Any]],
    source: str,
    chunk_embeddings: Any = None,
    valid_chunk_indices: Optional[Sequence[int]] = None,
):
    span_cm = start_span(
        name="preprocessing.chunking",
        input_data={
            "chunk_source": source,
            "num_chunks": len(chunks),
            "has_embeddings": chunk_embeddings is not None,
            "has_valid_indices": valid_chunk_indices is not None,
        },
        metadata={
            "operation": "preprocessing.chunking",
            "chunk_source": source,
        },
    )

    with optional_context(span_cm) as chunk_span:
        if chunk_span is None:
            return

        try:
            chunk_payload = _build_chunk_trace_payload(chunks)
            embedding_shape: Optional[List[int]] = None
            if chunk_embeddings is not None:
                if hasattr(chunk_embeddings, "shape"):
                    embedding_shape = list(chunk_embeddings.shape)
                elif isinstance(chunk_embeddings, (list, tuple)):
                    embedding_shape = [len(chunk_embeddings)]

            valid_indices_list = (
                list(valid_chunk_indices) if valid_chunk_indices is not None else None
            )

            set_span_output(
                chunk_span,
                output={
                    "chunk_source": source,
                    "num_chunks": len(chunks),
                    "chunks": chunk_payload,
                    "valid_chunk_indices": valid_indices_list,
                    "chunk_embedding_shape": embedding_shape,
                },
                metadata={
                    "chunk_source": source,
                    "num_chunks": len(chunks),
                },
            )
        except Exception as err:  # pragma: no cover - defensive logging
            logger.debug("Unable to record chunking trace output: %s", err)
            try:
                chunk_span.update(level="ERROR", status_message=str(err))
            except Exception:
                pass

# --- OpenAI client for Databricks API ---
# The OpenAI client is configured in the Databricks LLM client

# --- Load images as base64 for embedding in CSS ---
try:
    mascot_base64 = get_base64_encoded_image("src/keyword_code/assets/mascot.png")
    ifcontrollers_base64 = get_base64_encoded_image("src/keyword_code/assets/ifcontrollers.png")
    images_loaded = True
    logger.info("Successfully loaded all brand images")
except Exception as e:
    logger.error(f"Failed to load one or more images: {e}")
    images_loaded = False
    mascot_base64 = ""
    ifcontrollers_base64 = ""

# Check image is now loaded directly in the display_welcome_features function

# --- End Custom CSS Styling ---


def slugify_keyword_label(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "keyword"

# --- Function to enable interaction logging ---
def enable_interaction_logging():
    """
    Enable detailed logging of all interactions (BM25, semantic search, reranker, LLM).
    Creates a timestamped log file in the logs directory.

    This function is called automatically at startup if ENABLE_INTERACTION_LOGGING is True.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"rag_interactions_{timestamp}.log"

        # Set up interaction logging
        setup_interaction_logging(str(log_file_path))

        logger.info(f"Interaction logging enabled. Log file: {log_file_path}")
        return str(log_file_path)
    except Exception as e:
        logger.error(f"Failed to enable interaction logging: {e}")
        return None

# Function to preprocess files when uploaded
def preprocess_file(file_data: bytes, filename: str, use_advanced_extraction: bool = False):
    """
    Preprocesses a file by extracting chunks and computing their embeddings.
    Stores the results in session state for later use during prompt processing.

    Args:
        file_data: Raw bytes of the uploaded file
        filename: Name of the file
        use_advanced_extraction: Optional flag for advanced extraction features

    Returns:
        dict: Dictionary with preprocessing status and message
    """
    embedding_model = get_embedding_model()
    if embedding_model is None:
        logger.error(f"Skipping preprocessing for {filename}: Embedding model not loaded.")
        return {"status": "error", "message": "Embedding model not loaded"}

    # Get the current session ID for tracking temporary files
    session_id = get_session_id()
    update_session_access(session_id)

    # Create a temporary file to store the original file data if needed
    temp_files = []

    try:
        logger.info(f"Starting preprocessing for {filename}")
        file_extension = Path(filename).suffix.lower()

        # Extract chunks based on file type
        if file_extension == ".pdf":
            # Create a temporary file for the PDF
            temp_pdf_path = create_session_temp_file(prefix="pdf_", suffix=".pdf")
            temp_files.append(temp_pdf_path)

            with open(temp_pdf_path, 'wb') as f:
                f.write(file_data)

            processor = PDFProcessor(file_data)
            chunks, full_text = processor.extract_structured_text_and_chunks()
            original_pdf_bytes = file_data
        elif file_extension == ".docx":
            # Create a temporary file for the DOCX
            temp_docx_path = create_session_temp_file(prefix="docx_", suffix=".docx")
            temp_files.append(temp_docx_path)

            with open(temp_docx_path, 'wb') as f:
                f.write(file_data)

            word_processor = WordProcessor(file_data)
            pdf_bytes = word_processor.convert_to_pdf_bytes()
            if not pdf_bytes:
                raise ValueError("Failed to convert DOCX to PDF.")

            # Create a temporary file for the converted PDF
            temp_pdf_path = create_session_temp_file(prefix="converted_pdf_", suffix=".pdf")
            temp_files.append(temp_pdf_path)

            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_bytes)

            processor = PDFProcessor(pdf_bytes)
            chunks, full_text = processor.extract_structured_text_and_chunks()
            original_pdf_bytes = pdf_bytes
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if not chunks:
            logger.warning(f"No chunks extracted for {filename} during preprocessing.")
            return {
                "status": "error", 
                "message": "Error processing the PDF - possibly due to it being a scanned document. Please upload an OCR-converted version of the document."
            }

        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks from {filename}")
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        valid_chunk_indices = [i for i, text in enumerate(chunk_texts) if text.strip()]
        valid_chunk_texts = [chunk_texts[i] for i in valid_chunk_indices]

        if not valid_chunk_texts:
            logger.warning(f"No valid chunk texts found for {filename} during preprocessing.")
            return {
                "status": "error",
                "message": "Error processing the PDF - possibly due to it being a scanned document. Please upload an OCR-converted version of the document."
            }

        chunk_embeddings = embedding_model.encode(
            valid_chunk_texts, convert_to_tensor=True, show_progress_bar=False
        )

        # Store preprocessed data in session state
        if "preprocessed_data" not in st.session_state:
            st.session_state.preprocessed_data = {}

        st.session_state.preprocessed_data[filename] = {
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "valid_chunk_indices": valid_chunk_indices,
            "original_bytes": original_pdf_bytes,
            "timestamp": datetime.now().isoformat(),
            "temp_files": temp_files  # Store the list of temporary files for later cleanup
        }

        # Store the temporary files in session state for tracking
        if "temp_files" not in st.session_state:
            st.session_state.temp_files = []
        st.session_state.temp_files.extend(temp_files)

        logger.info(f"Successfully preprocessed {filename} with {len(chunks)} chunks and embeddings.")
        return {"status": "success", "message": f"Preprocessed {len(chunks)} chunks"}

    except Exception as e:
        # Clean up temporary files in case of error
        for temp_file in temp_files:
            try:
                remove_temp_file(temp_file)
            except Exception as cleanup_err:
                logger.error(f"Error cleaning up temporary file {temp_file}: {str(cleanup_err)}")

        logger.error(f"Error preprocessing {filename}: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Preprocessing failed: {str(e)}"}


def process_file_wrapper(args):
    """
    Wrapper for processing a single file: decompose prompt, run RAG & analysis per sub-prompt, aggregate results.
    Now uses precomputed embeddings when available.

    Supports explicit 'ask' and 'review' modes. Keyword mode is triggered automatically
    by the decomposition model when the prompt calls for deterministic keyword retrieval.
    """
    # Monitor memory usage before processing
    memory_before = get_memory_usage()
    logger.debug(f"Memory before processing: {format_bytes(memory_before['used'])} used, {memory_before['percent']}% of total")

    # Unpack args - now includes mode parameter
    if len(args) == 6:
        (
            uploaded_file_data,
            filename,
            user_prompt,
            use_advanced_extraction,
            preprocessed_data_for_file,
            mode  # 'ask', 'keyword', or 'review'
        ) = args
    else:
        # Backward compatibility: default to 'ask' mode if not specified
        (
            uploaded_file_data,
            filename,
            user_prompt,
            use_advanced_extraction,
            preprocessed_data_for_file
        ) = args
        mode = 'ask'

    ensure_interaction_logging()
    embedding_model = get_embedding_model()
    reranker_model = get_reranker_model()

    logger.info(f"Processing {filename} in '{mode}' mode")

    if mode == 'keyword':
        logger.warning("Explicit keyword mode flag is deprecated. Falling back to decomposition-driven flow.")
        mode = 'ask'

    if embedding_model is None:
        logger.error(f"Skipping processing for {filename}: Embedding model not loaded.")
        return {"filename": filename, "error": "Embedding model failed to load.", "annotated_pdf": None, "verification_results": {}, "phrase_locations": {}, "ai_analysis": json.dumps({"error": "Embedding model failed to load."})}

    try:
        current_session_id = get_session_id()
    except Exception as session_err:
        logger.debug("Unable to determine session id for Langfuse trace: %s", session_err)
        current_session_id = None

    if mode == 'review':
        logger.info(f"Review mode: Skipping preprocessing, RAG, and analysis for {filename}")
        return {
            "filename": filename,
            "mode": "review",
            "message": "Review mode: Validation handled separately",
            "annotated_pdf": None,
            "verification_results": {},
            "phrase_locations": {},
            "ai_analysis": json.dumps({"message": "Review mode: Validation handled separately"})
        }

    # Start analysis-level trace before preprocessing so chunking is captured
    trace_input = {
        "filename": filename,
        "prompt": user_prompt,
        "mode": mode,
    }
    trace_metadata = {
        "mode": mode,
        "use_advanced_extraction": bool(use_advanced_extraction),
        "preprocessed_available": bool(preprocessed_data_for_file),
        "memory_before_bytes": memory_before.get("used"),
        "num_chunks": 0,
    }
    trace_tags = [f"mode:{mode}", f"file:{filename}"]

    trace_context = start_trace(
        name="analysis.run",
        input_data=trace_input,
        metadata=trace_metadata,
        session_id=current_session_id,
        tags=trace_tags,
    )
    trace_entered = False
    try:
        if trace_context is not None:
            trace_context.__enter__()
            trace_entered = True
    except Exception as trace_err:
        logger.debug("Unable to start Langfuse analysis trace: %s", trace_err)
        trace_context = None

    exc_type = exc_value = exc_tb = None

    try:
        logger.info(f"Starting processing for {filename}")
        file_extension = Path(filename).suffix.lower()

        preprocessed_data: Optional[Dict[str, Any]] = None
        original_pdf_bytes_for_annotation: Optional[bytes] = None
        chunks: List[Dict[str, Any]] = []
        chunk_embeddings: Optional[Any] = None
        valid_chunk_indices: Optional[Sequence[int]] = None
        chunk_source: str = "unknown"

        if preprocessed_data_for_file and isinstance(preprocessed_data_for_file, dict):
            logger.info(f"Using preprocessed data for {filename}")
            chunks = preprocessed_data_for_file.get("chunks") or []
            chunk_embeddings = preprocessed_data_for_file.get("chunk_embeddings")
            valid_chunk_indices = preprocessed_data_for_file.get("valid_chunk_indices")
            original_pdf_bytes_for_annotation = preprocessed_data_for_file.get("original_bytes")

            if chunks and chunk_embeddings is not None and valid_chunk_indices is not None:
                preprocessed_data = {
                    "chunks": chunks,
                    "chunk_embeddings": chunk_embeddings,
                    "valid_chunk_indices": valid_chunk_indices,
                }
                chunk_source = "preprocessed_cache"
                logger.info(
                    "Successfully loaded preprocessed data for %s with %d chunks",
                    filename,
                    len(chunks),
                )
                _record_chunking_trace(
                    chunks=chunks,
                    source=chunk_source,
                    chunk_embeddings=chunk_embeddings,
                    valid_chunk_indices=valid_chunk_indices,
                )
            else:
                logger.warning(
                    "Preprocessed data for %s is incomplete, will reprocess",
                    filename,
                )
                preprocessed_data = None
                chunks = []
                chunk_embeddings = None
                valid_chunk_indices = None

        if not preprocessed_data:
            logger.info(f"No valid preprocessed data for {filename}, processing from scratch")
            processor: Optional[PDFProcessor] = None

            if file_extension == ".pdf":
                processor = PDFProcessor(uploaded_file_data)
                chunks, _ = processor.extract_structured_text_and_chunks()
                original_pdf_bytes_for_annotation = uploaded_file_data
            elif file_extension == ".docx":
                word_processor = WordProcessor(uploaded_file_data)
                pdf_bytes = word_processor.convert_to_pdf_bytes()
                if not pdf_bytes:
                    raise ValueError("Failed to convert DOCX to PDF.")
                processor = PDFProcessor(pdf_bytes)
                chunks, _ = processor.extract_structured_text_and_chunks()
                original_pdf_bytes_for_annotation = pdf_bytes
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            chunk_source = "fresh_extraction"
            _record_chunking_trace(
                chunks=chunks,
                source=chunk_source,
            )

        update_current_trace(
            metadata={
                "num_chunks": len(chunks),
                "chunk_source": chunk_source,
                "preprocessed_used": bool(preprocessed_data),
            },
            output={
                "preprocessing": {
                    "chunk_source": chunk_source,
                    "num_chunks": len(chunks),
                }
            },
        )

        analyzer = DocumentAnalyzer()
        from src.keyword_code.ai.decomposition import decompose_ask_mode_prompt

        active_chunks = preprocessed_data["chunks"] if preprocessed_data else chunks

        first_page_preview = ""
        document_index_preview = ""

        if active_chunks:
            first_page_number: Optional[int] = None
            for chunk_meta in active_chunks:
                page_num = chunk_meta.get("page_num")
                if isinstance(page_num, int):
                    if first_page_number is None or page_num < first_page_number:
                        first_page_number = page_num

            if first_page_number is not None:
                first_page_texts = [
                    chunk_meta.get("text", "")
                    for chunk_meta in active_chunks
                    if chunk_meta.get("page_num") == first_page_number and isinstance(chunk_meta.get("text"), str)
                ]
                combined_first_page = " ".join(first_page_texts).strip()
                if combined_first_page:
                    if len(combined_first_page) > 1800:
                        first_page_preview = combined_first_page[:1800].rstrip() + " ..."
                    else:
                        first_page_preview = combined_first_page

            for chunk_meta in active_chunks:
                metadata = chunk_meta.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("document_scope") != "table_of_contents":
                    continue
                toc_entries = metadata.get("toc_entries")
                if not isinstance(toc_entries, list):
                    continue
                formatted_entries: List[str] = []
                seen_entries: Set[str] = set()
                for entry in toc_entries:
                    entry_label: Optional[str] = None
                    entry_page: Optional[int] = None
                    if isinstance(entry, dict):
                        entry_label_raw = entry.get("entry") or entry.get("title") or entry.get("heading")
                        if isinstance(entry_label_raw, str):
                            entry_label = entry_label_raw.strip()
                        page_val = entry.get("page_number")
                        if isinstance(page_val, int):
                            entry_page = page_val
                    elif isinstance(entry, str):
                        entry_label = entry.strip()

                    if not entry_label:
                        continue

                    entry_text = entry_label
                    if entry_page is not None:
                        entry_text = f"{entry_text} (p.{entry_page})"

                    lowered = entry_text.lower()
                    if lowered in seen_entries:
                        continue
                    seen_entries.add(lowered)
                    formatted_entries.append(entry_text)

                    if len(formatted_entries) >= 20:
                        break

                if formatted_entries:
                    document_index_preview = "\n".join(formatted_entries)
                break

            if document_index_preview and len(document_index_preview) > 2000:
                document_index_preview = document_index_preview[:2000].rstrip() + "\n..."

        decomposition_output = run_async(
            decompose_ask_mode_prompt(
                analyzer,
                user_prompt,
                first_page_preview=first_page_preview,
                document_index_preview=document_index_preview,
            )
        )

        if isinstance(decomposition_output, dict):
            sub_prompts = decomposition_output.get("decomposition", []) or []
            user_request_context = decomposition_output.get("user_request_context", "")
            if not isinstance(user_request_context, str):
                user_request_context = ""
            else:
                user_request_context = user_request_context.strip()
        else:
            sub_prompts = decomposition_output or []
            user_request_context = ""

        # All sub-prompts use hybrid RAG with BM25 as one component
        rag_enabled_sub_prompts = [entry for entry in sub_prompts if isinstance(entry, dict)]

        # Keyword mode is disabled - all queries now use hybrid RAG workflow
        # BM25 keywords from decomposition are used within the RAG retrieval

        logger.info(f"Ask mode: Starting decomposition and RAG workflow for {filename}")
        logger.info(f"Decomposed prompt into {len(sub_prompts)} sub-prompts for {filename}")

        # --- Step 3: Process all sub-prompts with RAG using the unified context approach ---
        # This is an improved approach where we:
        # 1. Retrieve relevant chunks for each sub-prompt separately
        # 2. Send all sub-prompts and their contexts to the LLM in a single call
        # 3. The LLM can see the whole picture and provide more coherent answers
        # 4. This allows cross-referencing between sub-prompts while maintaining structure
        # First, collect all sub-prompts and their relevant chunks
        sub_prompts_with_contexts = []

        for sub_prompt_data in sub_prompts:
            sub_prompt = sub_prompt_data["sub_prompt"]
            sub_prompt_title = sub_prompt_data["title"]

            # Extract RAG parameters from decomposition (with defaults if missing)
            rag_params = sub_prompt_data.get("rag_params", {})
            bm25_weight = rag_params.get("bm25_weight", 0.5)
            semantic_weight = rag_params.get("semantic_weight", 0.5)
            rag_reasoning = rag_params.get("reasoning", "No reasoning provided")
            raw_bm25_terms = rag_params.get("bm25_terms", [])
            if isinstance(raw_bm25_terms, str):
                raw_bm25_terms = [raw_bm25_terms]
            elif isinstance(raw_bm25_terms, (tuple, set)):
                raw_bm25_terms = list(raw_bm25_terms)
            elif not isinstance(raw_bm25_terms, list):
                raw_bm25_terms = []

            bm25_terms: List[str] = []
            seen_terms_local: Set[str] = set()
            for term in raw_bm25_terms:
                if not isinstance(term, str):
                    continue
                cleaned = term.strip()
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if lowered in seen_terms_local:
                    continue
                seen_terms_local.add(lowered)
                bm25_terms.append(cleaned)

            rag_params["bm25_terms"] = bm25_terms

            hyde_queries: List[str] = []
            raw_hyde = sub_prompt_data.get("hyde")
            if isinstance(raw_hyde, str):
                raw_hyde = [raw_hyde]
            if isinstance(raw_hyde, (list, tuple)):
                for candidate in raw_hyde:
                    if not isinstance(candidate, str):
                        continue
                    cleaned_candidate = candidate.strip()
                    if cleaned_candidate:
                        hyde_queries.append(cleaned_candidate)
            rag_params["hyde"] = hyde_queries
            if hyde_queries:
                logger.info(f"HYDE semantic prompts for '{sub_prompt_title}': {hyde_queries}")

            logger.info(f"Retrieving relevant chunks for sub-prompt '{sub_prompt_title}' for {filename}")
            logger.info(f"Using RAG weights - BM25: {bm25_weight:.2f}, Semantic: {semantic_weight:.2f}")
            logger.info(f"RAG weight reasoning: {rag_reasoning}")
            if bm25_terms:
                logger.info(f"BM25 lexical terms: {bm25_terms}")

            # Log RAG parameters to interaction logger
            log_rag_parameters(
                sub_prompt_title=sub_prompt_title,
                sub_prompt=sub_prompt,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                bm25_terms=bm25_terms,
                reasoning=rag_reasoning,
                source="decomposition"
            )

            # Use retrieve_relevant_chunks with preprocessed embeddings if available
            # Use local reranker model instead of LLM ranking
            # Now using optimized weights from decomposition
            if preprocessed_data:
                relevant_chunks = retrieve_relevant_chunks(
                    prompt=sub_prompt,
                    chunks=preprocessed_data["chunks"],
                    model=embedding_model,
                    top_k=RAG_TOP_K,
                    precomputed_embeddings=preprocessed_data["chunk_embeddings"],
                    valid_chunk_indices=preprocessed_data["valid_chunk_indices"],
                    reranker_model=reranker_model,  # Use local reranker model
                    bm25_weight=bm25_weight,  # Use optimized weight from decomposition
                    semantic_weight=semantic_weight,  # Use optimized weight from decomposition
                    bm25_terms=bm25_terms,
                    alternate_hyde_queries=hyde_queries,
                )
            else:
                relevant_chunks = retrieve_relevant_chunks(
                    prompt=sub_prompt,
                    chunks=chunks,
                    model=embedding_model,
                    top_k=RAG_TOP_K,
                    reranker_model=reranker_model,  # Use local reranker model
                    bm25_weight=bm25_weight,  # Use optimized weight from decomposition
                    semantic_weight=semantic_weight,  # Use optimized weight from decomposition
                    bm25_terms=bm25_terms,
                    alternate_hyde_queries=hyde_queries,
                )

            # Store sub-prompt data with its relevant chunks and RAG params
            sub_prompts_with_contexts.append({
                "title": sub_prompt_title,
                "sub_prompt": sub_prompt,
                "relevant_chunks": relevant_chunks,
                "rag_params": rag_params,  # Store for potential retry use
                "hyde": hyde_queries,
            })

            logger.info(f"Retrieved {len(relevant_chunks)} chunks for sub-prompt '{sub_prompt_title}' in {filename} using optimized RAG weights")

        # Now analyze all sub-prompts in a single LLM call
        logger.info(f"Analyzing all {len(sub_prompts_with_contexts)} sub-prompts together for {filename} using unified context approach")

        start_time = datetime.now()
        all_sub_prompt_results = run_async(
            analyzer.analyze_document_with_all_contexts(
                filename=filename,
                main_prompt=user_prompt,
                sub_prompts_with_contexts=sub_prompts_with_contexts
            )
        )
        analysis_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Unified context analysis completed in {analysis_time:.2f} seconds for {len(sub_prompts_with_contexts)} sub-prompts")

        # Add relevant chunks to the results for verification and annotation
        for result in all_sub_prompt_results:
            # Find the matching sub-prompt context
            matching_context = next(
                (item for item in sub_prompts_with_contexts
                 if item["sub_prompt"] == result["sub_prompt"]),
                None
            )

            if matching_context:
                result["relevant_chunks"] = matching_context["relevant_chunks"]
                result["rag_params"] = matching_context.get("rag_params", {})
            else:
                # Fallback if no match found
                logger.warning(f"No matching context found for sub-prompt: {result['sub_prompt'][:50]}...")
                result["relevant_chunks"] = []
                result["rag_params"] = {}

        logger.info(f"Completed analysis for all sub-prompts in {filename}")

        # --- Step 4: Aggregate results from all sub-prompts ---
        aggregated_analysis = {
            "title": f"Analysis of {filename}",
            "analysis_sections": {}
        }

        for i, result in enumerate(all_sub_prompt_results):
            try:
                sub_analysis = json.loads(result["analysis_json"])
                section_key = f"section_{i+1}_{result['title'].replace(' ', '_').lower()}"

                # Track the section key on the sub-prompt payload so downstream UI can map prompts.
                result.setdefault("section_key", section_key)

                # Log the structure of the sub-analysis for debugging
                logger.info(f"Sub-analysis structure for '{result['title']}': {list(sub_analysis.keys())}")

                # Convert new format to old format for compatibility
                if "analysis_summary" in sub_analysis and "supporting_quotes" in sub_analysis:
                    # Ensure supporting_quotes is a list of strings
                    supporting_quotes = sub_analysis["supporting_quotes"]
                    if not isinstance(supporting_quotes, list):
                        logger.warning(f"supporting_quotes is not a list: {type(supporting_quotes)}")
                        supporting_quotes = [str(supporting_quotes)]

                    # Ensure each quote is a string
                    supporting_quotes = [str(quote) if not isinstance(quote, str) else quote for quote in supporting_quotes]

                    aggregated_analysis["analysis_sections"][section_key] = {
                        "Analysis": sub_analysis["analysis_summary"],
                        "Supporting_Phrases": supporting_quotes,
                        "Context": sub_analysis.get("analysis_context", f"From sub-prompt: {result['sub_prompt']}"),
                        "analysis_json": result.get("analysis_json", json.dumps(sub_analysis))
                    }
                else:
                    # Handle unexpected format
                    logger.warning(f"Unexpected sub-analysis format for '{result['title']}': {list(sub_analysis.keys())}")
                    aggregated_analysis["analysis_sections"][section_key] = {
                        "Analysis": f"Error parsing analysis for '{result['title']}'",
                        "Supporting_Phrases": ["No relevant phrase found."],
                        "Context": f"Error in sub-prompt: {result['sub_prompt']}",
                        "analysis_json": result.get("analysis_json", "{}")
                    }
            except Exception as e:
                logger.error(f"Error aggregating results for sub-prompt '{result['title']}' in {filename}: {e}")
                error_section_key = f"error_{i+1}_{result['title'].replace(' ', '_').lower()}"
                aggregated_analysis["analysis_sections"][error_section_key] = {
                    "Analysis": f"Error processing this section: {str(e)}",
                    "Supporting_Phrases": ["No relevant phrase found."],
                    "Context": f"Error in sub-prompt: {result['sub_prompt']}"
                }

        verification_input_json = json.dumps(aggregated_analysis, indent=2)
        logger.info(f"Aggregated analysis results for {filename} (pre-keyword merge)")

        # --- Step 4.5: Fact Extraction ---
        # Note: Fact extraction is now only performed on-demand when the user clicks "Generate Facts"
        # This improves performance by not running extraction during the main query processing
        extracted_facts = None

        # --- Step 5: Verification & Annotation (on aggregated results) ---
        # If we have a status_container from the main thread, update it

        # Create a processor if none exists (if preprocessed data was used)
        if preprocessed_data and 'processor' not in locals():
            processor = PDFProcessor(original_pdf_bytes_for_annotation)

        logger.info(
            "Aggregated analysis prepared %d section(s) before keyword merge for %s",
            len(aggregated_analysis.get("analysis_sections", {})),
            filename,
        )

        # Verification uses the *original* processor instance with all chunks
        logger.info(f"Verifying phrases from aggregated analysis for {filename}.")
        try:
            verification_results, phrase_locations = processor.verify_and_locate_phrases(
                verification_input_json
            )
            logger.info(f"Verification results: {len(verification_results)} phrases, {sum(1 for v in verification_results.values() if v)} verified")
        except Exception as verify_err:
            logger.error(f"Error during verification: {verify_err}", exc_info=True)
            verification_results, phrase_locations = {}, {}

        # Keyword mode is disabled - no separate keyword context to merge
        aggregated_ai_analysis_json_str = json.dumps(aggregated_analysis, indent=2)
        logger.info(
            "Final analysis for %s contains %d section(s)",
            filename,
            len(aggregated_analysis.get("analysis_sections", {})),
        )

        # Add annotations to the PDF
        logger.info(f"Adding annotations to PDF for {filename}.")
        try:
            annotated_pdf_bytes = processor.add_annotations(phrase_locations)
            logger.info(f"Successfully added annotations to PDF for {filename}.")
        except Exception as annot_err:
            logger.error(f"Error adding annotations to PDF for {filename}: {annot_err}", exc_info=True)
            annotated_pdf_bytes = original_pdf_bytes_for_annotation  # Use original PDF if annotation fails

        # Encode the annotated PDF for return
        annotated_pdf_base64 = base64.b64encode(annotated_pdf_bytes).decode('utf-8')

        # Monitor memory usage after processing
        memory_after = get_memory_usage()
        memory_used = memory_after['used'] - memory_before['used']
        logger.debug(f"Memory after processing: {format_bytes(memory_after['used'])} used, {memory_after['percent']}% of total")
        logger.debug(f"Memory used during processing: {format_bytes(memory_used)}")

        # Check if memory usage is high and perform cleanup if necessary
        if memory_after['percent'] > 75:
            logger.warning(f"High memory usage after processing {filename}: {memory_after['percent']}%")
            cleanup_result = cleanup_memory(force=True)
            logger.info(f"Memory cleanup performed: {cleanup_result.get('freed_formatted', '0 B')} freed")

        # --- Step 6: Return the results ---
        # Get the trace_id to store for future follow-up questions
        current_trace_id = None
        try:
            langfuse_client = get_langfuse_client_cached()
            if langfuse_client is not None:
                current_trace_id = langfuse_client.get_current_trace_id()
                logger.debug(f"Captured trace_id for {filename}: {current_trace_id}")
        except Exception as trace_id_err:
            logger.debug(f"Could not capture trace_id: {trace_id_err}")

        result_payload = {
            "filename": filename,
            "annotated_pdf": annotated_pdf_base64,
            "verification_results": verification_results,
            "phrase_locations": phrase_locations,
            "ai_analysis": aggregated_ai_analysis_json_str,
            "sub_prompt_results": all_sub_prompt_results,
            "extracted_facts": extracted_facts,
            "trace_id": current_trace_id,  # Store trace_id for follow-up questions
            "keyword_mode": False,  # Keyword mode is disabled
        }

        analysis_sections_count = len(aggregated_analysis.get("analysis_sections", {}))
        update_current_trace(output={
            "status": "success",
            "keyword_mode": False,
            "analysis_sections": analysis_sections_count,
            "sub_prompts_processed": len(all_sub_prompt_results),
        })

        return result_payload

    except Exception as e:
        exc_type, exc_value, exc_tb = type(e), e, e.__traceback__
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)

        update_current_span(level="ERROR", status_message=str(e))
        update_current_trace(output={
            "status": "error",
            "message": str(e),
        })

        # Perform memory cleanup on error
        cleanup_memory(force=True)

        return {
            "filename": filename,
            "error": f"Processing failed: {str(e)}",
            "annotated_pdf": None,
            "verification_results": {},
            "phrase_locations": {},
            "ai_analysis": json.dumps({"error": f"Processing failed: {str(e)}"})
        }
    finally:
        if trace_context is not None and trace_entered:
            try:
                trace_context.__exit__(exc_type, exc_value, exc_tb)
            except Exception as exit_err:
                logger.debug("Failed to close Langfuse trace: %s", exit_err)


def display_page():
    """Main function to display the Streamlit page with prompt decomposition."""
    # Apply UI styling first to ensure CSS is consistently applied on page load/reload
    apply_ui_styling()

    # Initialize session state variables
    initialize_session_state()

    embedding_model = get_embedding_model()

    # Check for incompatible embeddings and clear them automatically
    if embedding_model is not None:
        clear_incompatible_embeddings(embedding_model)

    # Monitor memory usage and perform cleanup if necessary
    memory_status = monitor_memory_usage(auto_cleanup=True)
    if memory_status.get("status") == "critical":
        logger.warning(f"Memory usage critical: {memory_status.get('message')}")
        if "cleanup" in memory_status:
            cleanup_info = memory_status["cleanup"]
            logger.info(f"Memory cleanup performed: {cleanup_info.get('freed_formatted', '0 B')} freed")

    # Render branding elements
    render_branding()

    # --- NEW: Initialize chat messages ---
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
    # --- NEW: Initialize follow-up Q&A ---
    if "followup_qa_by_doc" not in st.session_state: st.session_state.followup_qa_by_doc = {}
    # --- NEW: Flag for user file changes ---
    if "file_selection_changed_by_user" not in st.session_state: st.session_state.file_selection_changed_by_user = False
    # --- NEW: Store files temporarily during change handling ---
    if "current_file_objects_from_change" not in st.session_state: st.session_state.current_file_objects_from_change = None
    # --- NEW: Flag for auto-scroll ---
    if "results_just_generated" not in st.session_state: st.session_state.results_just_generated = False

    # Check if results exist
    if st.session_state.get("analysis_results"):
        # --- RESULTS VIEW ---
        try:
            logo_base64 = get_base64_encoded_image("src/keyword_code/assets/smartdocslogo.png")
            if logo_base64:
                st.markdown(
                    f"""
                    <div class="smartdocs-logo-container" style="text-align: center;">
                        <img src="data:image/png;base64,{logo_base64}" alt="CNT SmartDocs" style="max-width: 500px; width: 100%; height: auto;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Fallback to text if image fails to load
                st.markdown(
                    """
                    <div class="smartdocs-logo-container">
                        <h1><span style='color: #002345;'>CNT</span> <span style='color: #00ADE4;'>SmartDocs</span></h1>
                        <p>AI Powered Document Intelligence</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            logger.error(f"Failed to load SmartDocs logo: {e}")
            # Fallback to text if image fails to load
            st.markdown(
                """
                <div class="smartdocs-logo-container">
                    <h1><span style='color: #002345;'>CNT</span> <span style='color: #00ADE4;'>SmartDocs</span></h1>
                    <p>AI Powered Document Intelligence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🚀 Start New Analysis", key="new_analysis_button", use_container_width=True, type="primary"):
            # Use our new function to clear session state and temporary files
            clear_session_for_new_query()
            logger.info("Cleared state and temporary files for new analysis.")
            st.rerun()

        # Display Results Section (moved from outside)
        st.divider()

        results_to_display = st.session_state.get("analysis_results", [])
        errors = [r for r in results_to_display if isinstance(r, dict) and "error" in r]
        success_results = [r for r in results_to_display if isinstance(r, dict) and "error" not in r]

        # Status Summary
        total_processed = len(results_to_display)
        if errors:
            if not success_results: st.error(f"Processing failed for all {total_processed} file(s). See details below.")
            else: st.warning(f"Processing complete for {total_processed} file(s). {len(success_results)} succeeded, {len(errors)} failed.")
        # Removed the success message that was here

        # Error Details Expander
        if errors:
            with st.expander("⚠️ Processing Errors", expanded=True):
                for error_res in errors:
                    st.error(f"**{error_res.get('filename', 'Unknown File')}**: {error_res.get('error', 'Unknown error details.')}")

        # Display Successful Analysis
        if success_results:
            # Import lazily to avoid heavy initialization when the page isn't used
            from .display_utils.analysis_display import display_analysis_results

            # Call the display function from the display module
            display_analysis_results(success_results)
        elif not errors:
            st.warning("Processing finished, but no primary analysis content was generated.")

        # Auto-scroll logic (only runs when results are first shown)
        if st.session_state.get("results_just_generated", False):
            js = """
            <script>
                setTimeout(function() {
                    const anchor = document.getElementById('results-anchor');
                    if (anchor) {
                        console.log("Scrolling to results anchor...");
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 100);
            </script>
            """
            st.components.v1.html(js, height=0)
            st.session_state.results_just_generated = False # Reset flag after scroll

    else:
        # --- INPUT VIEW ---

        try:
            logo_base64 = get_base64_encoded_image("src/keyword_code/assets/smartdocslogo.png")
            if logo_base64:
                st.markdown(
                    f"""
                    <div class="smartdocs-logo-container" style="text-align: center;">
                        <img src="data:image/png;base64,{logo_base64}" alt="CNT SmartDocs" style="max-width: 500px; width: 100%; height: auto;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Fallback to text if image fails to load
                st.markdown(
                    """
                    <div class="smartdocs-logo-container">
                        <h1><span style='color: #002345;'>CNT</span> <span style='color: #00ADE4;'>SmartDocs</span></h1>
                        <p>AI Powered Document Intelligence</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            logger.error(f"Failed to load SmartDocs logo: {e}")
            # Fallback to text if image fails to load
            st.markdown(
                """
                <div class="smartdocs-logo-container">
                    <h1><span style='color: #002345;'>CNT</span> <span style='color: #00ADE4;'>SmartDocs</span></h1>
                    <p>AI Powered Document Intelligence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Check if embedding model loaded successfully (important for input view too)
        if embedding_model is None:
            st.error(
                "Embedding model failed to load. Document processing is disabled. "
                "Please check logs and ensure dependencies are installed correctly."
            )
            return # Stop further UI rendering

        # File Upload Callback
        def handle_file_change():
            current_files = st.session_state.get("file_uploader_decompose", [])
            st.session_state.current_file_objects_from_change = current_files
            st.session_state.file_selection_changed_by_user = True
            logger.debug(f"handle_file_change: Stored {len(current_files) if current_files else 0} files. Flag set.")

        # File Uploader - The variable isn't directly used but Streamlit needs it for the UI
        _ = st.file_uploader(
            "Upload PDF or Word files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader_decompose",
            on_change=handle_file_change,
        )

        # Create a placeholder for preprocessing status or features
        preprocessing_or_features_container = st.empty()

        # File Change Logic
        if st.session_state.file_selection_changed_by_user:
            logger.debug("Processing detected file change from user action.")
            st.session_state.file_selection_changed_by_user = False
            current_files = st.session_state.current_file_objects_from_change
            current_uploaded_filenames = set(f.name for f in current_files) if current_files else set()
            last_filenames = st.session_state.get('last_uploaded_filenames', set())

            if current_uploaded_filenames != last_filenames:
                logger.info(f"Actual file change confirmed: New={current_uploaded_filenames - last_filenames}, Removed={last_filenames - current_uploaded_filenames}")
                new_files = current_uploaded_filenames - last_filenames
                removed_files = last_filenames - current_uploaded_filenames
                st.session_state.uploaded_file_objects = current_files
                st.session_state.last_uploaded_filenames = current_uploaded_filenames

                for removed_file in removed_files:
                    if removed_file in st.session_state.preprocessed_data:
                        # Clean up any temporary files associated with this file
                        file_data = st.session_state.preprocessed_data[removed_file]
                        if "temp_files" in file_data and isinstance(file_data["temp_files"], list):
                            for temp_file in file_data["temp_files"]:
                                try:
                                    remove_temp_file(temp_file)
                                    logger.debug(f"Removed temporary file: {temp_file}")
                                except Exception as e:
                                    logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

                        # Remove from session state
                        del st.session_state.preprocessed_data[removed_file]
                        if removed_file in st.session_state.preprocessing_status:
                            del st.session_state.preprocessing_status[removed_file]
                        logger.info(f"Removed preprocessing data and temporary files for {removed_file}")

                st.session_state.analysis_results = [] # Clear any old results if files change
                st.session_state.chat_messages = [] # Clear chat too
                st.session_state.followup_qa_by_doc = {} # Clear per-document follow-up Q&A too
                logger.info("Cleared relevant state due to file change.")

                if new_files:
                    # Create a status indicator in place of the features container
                    with preprocessing_or_features_container.container():
                        with st.status(f"Preprocessing {len(new_files)} document(s)...", expanded=True) as status:
                            preprocessing_failed = False
                            success_count = 0

                            for i, filename in enumerate(sorted(list(new_files))):
                                file_obj = next((f for f in current_files if f.name == filename), None)
                                if file_obj:
                                    try:
                                        # Update status with current file
                                        status.update(label=f"Preprocessing file {i+1}/{len(new_files)}: {filename}")
                                        st.write(f"Processing {filename}...")

                                        file_data = file_obj.getvalue()
                                        result = preprocess_file(
                                            file_data,
                                            filename,
                                            st.session_state.get("use_advanced_extraction", False)
                                        )
                                        st.session_state.preprocessing_status[filename] = result
                                        logger.info(f"Preprocessed {filename}: {result['status']}")

                                        if result['status'] == 'success':
                                            success_count += 1
                                            st.write(f"✅ {filename} processed successfully.")
                                        elif result['status'] == 'warning':
                                            st.write(f"⚠️ {filename} processed with warnings: {result['message']}")
                                            preprocessing_failed = True
                                        else:
                                            st.write(f"❌ Error processing {filename}: {result['message']}")
                                            preprocessing_failed = True

                                    except Exception as e:
                                        logger.error(f"Failed to preprocess {filename}: {str(e)}", exc_info=True)
                                        st.session_state.preprocessing_status[filename] = {"status": "error", "message": f"Failed to preprocess: {str(e)}"}
                                        st.write(f"❌ Error processing {filename}: {str(e)}")
                                        preprocessing_failed = True

                            # Update status based on results
                            if preprocessing_failed:
                                if success_count > 0:
                                    status.update(label=f"Preprocessing complete with issues. {success_count}/{len(new_files)} files processed successfully.", state="warning", expanded=False)
                                else:
                                    status.update(label="Preprocessing failed. Please check the errors and try again.", state="error", expanded=False)
                            else:
                                status.update(label=f"Preprocessing complete! {success_count}/{len(new_files)} files processed successfully.", state="complete", expanded=False)
                else:
                    logger.debug("File change flag was True, but filename sets match. Ignoring spurious flag.")

            st.session_state.current_file_objects_from_change = None

        # Welcome Features Section - Only show before files are processed and when not preprocessing
        if not st.session_state.get("preprocessed_data"):
            # Display features grid only if not currently preprocessing files
            with preprocessing_or_features_container.container():
                display_welcome_features()

        # Analysis Inputs - Only show if preprocessed data exists
        if st.session_state.get("preprocessed_data"):
            # --- Prompt Suggestions (Ask mode from configuration) ---
            ask_config = SAVED_PROMPTS.get("Ask", {})
            prompt_suggestions = []
            for _cat, _items in ask_config.items():
                for _it in _items:
                    prompt_suggestions.append({"label": _it.get("label", "Unnamed"), "prompt": _it.get("prompt", "")})
            suggestion_labels = [s["label"] for s in prompt_suggestions]
            suggestion_prompts = [s["prompt"] for s in prompt_suggestions]
            # Show pills for suggestions
            selected_pill = stp.pills(
                "Prompt Suggestions:",
                suggestion_labels,
                clearable=True,
                index=None,
                label_visibility="visible"
            )
            if selected_pill:
                idx = suggestion_labels.index(selected_pill)
                st.session_state["user_prompt"] = suggestion_prompts[idx]

            with st.container(border=False):
                st.session_state.user_prompt = st.text_area(
                    "Analysis Prompt",
                    placeholder="Enter your analysis instructions...",
                    height=150,
                    key="prompt_input_decompose",
                    value=st.session_state.get("user_prompt", ""),
                )

            # Process Button
            process_button_disabled = (
                embedding_model is None
                or not st.session_state.get('uploaded_file_objects')
                or not st.session_state.get('user_prompt', '').strip()
            )
            if st.button("Process Documents", type="primary", use_container_width=True, disabled=process_button_disabled):
                    files_to_process = st.session_state.get("uploaded_file_objects", [])
                    current_user_prompt = st.session_state.get("user_prompt", "")
                    current_use_advanced = st.session_state.get("use_advanced_extraction", False)

                    if not files_to_process: st.warning("Please upload one or more documents.")
                    elif not current_user_prompt.strip(): st.error("Please enter an Analysis Prompt.")
                    else:
                        st.session_state.analysis_results = [] # Clear previous results before processing
                        st.session_state.followup_qa_by_doc = {} # Clear per-document follow-up Q&A before processing
                        st.session_state.show_pdf = False
                        st.session_state.pdf_bytes = None
                        st.session_state.current_pdf_name = None

                        total_files = len(files_to_process)
                        overall_start_time = datetime.now()
                        results_placeholder = [None] * total_files
                        file_map = {i: f.name for i, f in enumerate(files_to_process)}

                        process_args = []
                        files_read_ok = True
                        for i, uploaded_file in enumerate(files_to_process):
                            try:
                                file_data = uploaded_file.getvalue()
                                # Add the preprocessed data for this file to the args
                                preprocessed_file_data = st.session_state.get("preprocessed_data", {}).get(uploaded_file.name)
                                process_args.append(
                                    (file_data, uploaded_file.name, current_user_prompt, current_use_advanced, preprocessed_file_data)
                                )
                            except Exception as read_err:
                                logger.error(f"Failed to read file {uploaded_file.name}: {read_err}", exc_info=True)
                                st.error(f"Failed to read file {uploaded_file.name}. Please re-upload.")
                                results_placeholder[i] = {"filename": uploaded_file.name, "error": f"Failed to read file: {read_err}"}
                                files_read_ok = False

                        if files_read_ok and process_args:
                            files_to_run_count = len(process_args)

                            # Create expander for document processing status
                            with st.expander("Document Processing Status", expanded=True):
                                status_container = st.empty()
                                status_container.info(f"Starting to process {files_to_run_count} document(s)...")

                                with st.spinner("Analysing Query...", show_time=True):
                                    processed_indices = set()
                                    def run_task_with_index(item_index: int, args_tuple: tuple):
                                        filename = args_tuple[1]
                                        logger.info(f"Thread {threading.current_thread().name} starting task for index {item_index} ({filename})")
                                        try:
                                            result = process_file_wrapper(args_tuple)
                                            logger.info(f"Thread {threading.current_thread().name} finished task for index {item_index} ({filename})")
                                            return item_index, result
                                        except Exception as thread_err:
                                            logger.error(f"Unhandled error in thread task for index {item_index} ({filename}): {thread_err}", exc_info=True)
                                            return item_index, {"filename": filename, "error": f"Unhandled thread error: {thread_err}"}

                                try:
                                    if ENABLE_PARALLEL and len(process_args) > 1:
                                        logger.info(f"Executing {len(process_args)} tasks in parallel with max workers: {MAX_WORKERS}")
                                        # Process each file in parallel
                                        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                            future_to_index = {executor.submit(run_task_with_index, i, args): i for i, args in enumerate(process_args)}

                                            for future in concurrent.futures.as_completed(future_to_index):
                                                original_index = future_to_index[future]
                                                processed_indices.add(original_index)
                                                fname = file_map.get(original_index, f"File at index {original_index}")
                                                try:
                                                    _, result_data = future.result()
                                                    results_placeholder[original_index] = result_data
                                                except Exception as exc:
                                                    logger.error(f'Task for index {original_index} ({fname}) failed: {exc}', exc_info=True)
                                                    results_placeholder[original_index] = {"filename": fname, "error": f"Task execution failed: {exc}"}
                                    else:
                                        logger.info(f"Processing {files_to_run_count} task(s) sequentially.")
                                        for i, arg_tuple in enumerate(process_args):
                                            original_index = i
                                            processed_indices.add(original_index)
                                            try:
                                                _, result_data = run_task_with_index(original_index, arg_tuple)
                                                results_placeholder[original_index] = result_data
                                            except Exception as seq_exc:
                                                 fname = file_map.get(original_index, f"File at index {original_index}")
                                                 logger.error(f'Sequential task for index {original_index} ({fname}) failed: {seq_exc}', exc_info=True)
                                                 results_placeholder[original_index] = {"filename": fname, "error": f"Task execution failed: {seq_exc}"}

                                except Exception as pool_err:
                                     logger.error(f"Error during task execution setup/management: {pool_err}", exc_info=True)
                                     st.error(f"Error during processing: {pool_err}. Some files may not have been processed.")
                                     for i in range(total_files):
                                          if i not in processed_indices and results_placeholder[i] is None:
                                               fname = file_map.get(i, f"File at index {i}")
                                               results_placeholder[i] = {"filename": fname, "error": f"Processing cancelled due to execution error: {pool_err}"}

                                final_results = [r for r in results_placeholder if r is not None]
                                st.session_state.analysis_results = final_results
                                total_time = (datetime.now() - overall_start_time).total_seconds()
                                success_count = len([r for r in final_results if isinstance(r, dict) and "error" not in r])
                                logger.info(f"Processing batch complete. Processed {success_count}/{total_files} files successfully in {total_time:.2f}s.")
                                status_container.success(f"Processing complete! Processed {success_count}/{total_files} files successfully in {total_time:.2f} seconds.")

                        # Outside the expander, handle post-processing like PDF loading
                        first_success = next((r for r in final_results if isinstance(r, dict) and "error" not in r), None)
                        if first_success and first_success.get("annotated_pdf"):
                            try:
                                pdf_bytes = base64.b64decode(first_success["annotated_pdf"])
                                # Import here to avoid triggering heavy imports until needed
                                from .display_utils.pdf_utils import update_pdf_view

                                # Use the update_pdf_view function from the display module
                                update_pdf_view(
                                    pdf_bytes=pdf_bytes,
                                    page_num=1,
                                    filename=first_success.get("filename")
                                )
                                # Set flag to show PDF viewer
                                st.session_state.show_pdf = True
                            except Exception as decode_err:
                                logger.error(f"Failed to decode/set initial PDF: {decode_err}", exc_info=True)
                                st.error("Failed to load initial PDF view.")
                                st.session_state.show_pdf = False
                        elif first_success:
                             logger.warning("First successful result missing annotated PDF data.")
                             st.warning("Processing complete, but couldn't display the first annotated document.")
                             st.session_state.show_pdf = False
                        else:
                             logger.warning("No successful results found. No initial PDF view shown.")
                             st.session_state.show_pdf = False

                        # Set flag to trigger scroll on next rerun
                        if 'success_count' in locals() and success_count > 0:
                            st.session_state.results_just_generated = True

                        # Rerun to display results / updated PDF view state
                        st.rerun()


# Register cleanup functions to run when the application exits
from .utils.spacy_utils import cleanup_spacy_models
atexit.register(cleanup_spacy_models)
atexit.register(cleanup_all_temp_files)

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Check model again before displaying page, though initial check should handle most cases
    model = get_embedding_model()
    if model is not None:
        display_page()
    else:
        # If the model failed, display_page() will show an error,
        # but we can add a log here just in case.
        logger.critical("Application cannot start because the embedding model failed to load.")