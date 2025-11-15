"""
Retrieval functionality for RAG.
"""

import asyncio
import re
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set, Callable
from ..config import logger, RAG_TOP_K, RAG_WORKERS
from ..utils.async_utils import run_tasks_in_parallel, run_in_threadpool
from ..utils.interaction_logger import (
    log_bm25_results,
    log_semantic_search_results,
    log_reranker_results
)
from ..utils.langfuse_tracing import (
    optional_context,
    record_span_error,
    set_span_output,
    start_span,
)

# Add BM25 import for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logger.warning("rank_bm25 not available. BM25 retrieval will be disabled.")


def numpy_cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between two numpy arrays.
    Args:
        a: Query embedding array, shape (embedding_dim,) or (1, embedding_dim)
        b: Document embeddings array, shape (num_docs, embedding_dim)
    Returns:
        Cosine similarity scores, shape (num_docs,)
    """
    # Ensure a is 2D: (1, embedding_dim)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    # Ensure b is 2D: (num_docs, embedding_dim)
    if b.ndim == 1:
        b = b[np.newaxis, :]
    # Normalize
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    # Compute cosine similarity
    # a_norm: (1, embedding_dim), b_norm: (num_docs, embedding_dim)
    # Result: (num_docs,)
    return np.dot(b_norm, a_norm[0])


_SKIP_METADATA_KEYS = {"tokens", "embedding", "bbox", "bboxes"}


def _truncate_text(value: str, limit: int = 200) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _coerce_jsonable(value: Any, *, max_length: int = 200, max_items: int = 10) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + "..."
        return value
    if isinstance(value, list):
        sanitized_list: List[Any] = []
        for item in value[:max_items]:
            sanitized_list.append(_coerce_jsonable(item, max_length=max_length, max_items=max_items))
        if len(value) > max_items:
            sanitized_list.append("...")
        return sanitized_list
    if isinstance(value, dict):
        sanitized_dict: Dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_items:
                sanitized_dict["..."] = f"{len(value) - max_items} more keys"
                break
            sanitized_dict[str(key)] = _coerce_jsonable(val, max_length=max_length, max_items=max_items)
        return sanitized_dict
    if isinstance(value, set):
        truncated = list(value)[:max_items]
        result = [_coerce_jsonable(item, max_length=max_length, max_items=max_items) for item in truncated]
        if len(value) > max_items:
            result.append("...")
        return result
    value_str = str(value)
    if len(value_str) > max_length:
        return value_str[:max_length] + "..."
    return value_str


def _summarize_chunk(
    chunks: List[Dict[str, Any]],
    chunk_index: int,
    *,
    score: Optional[float] = None,
    score_label: str = "score",
    rank: Optional[int] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "chunk_index": chunk_index,
    }

    if rank is not None:
        summary["rank"] = rank

    if score is not None:
        try:
            summary[score_label] = float(score)
        except Exception:
            summary[score_label] = score

    if 0 <= chunk_index < len(chunks):
        chunk = chunks[chunk_index]
        summary["page_num"] = chunk.get("page_num")
        summary["page_label"] = chunk.get("page_label")
        summary["retrieval_method"] = chunk.get("retrieval_method")
        summary["char_count"] = len(chunk.get("text", ""))
        summary["text_preview"] = _truncate_text(chunk.get("text", ""))

        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            sanitized_metadata: Dict[str, Any] = {}
            for key, value in metadata.items():
                if key in _SKIP_METADATA_KEYS:
                    continue
                sanitized_metadata[key] = _coerce_jsonable(value)
            if sanitized_metadata:
                summary["metadata"] = sanitized_metadata

    return summary


def get_bm25_results(
    prompt: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
    *,
    bm25_terms: Optional[List[str]] = None,
) -> List[Tuple[int, float]]:
    """
    Retrieves the top_k most relevant chunks using BM25 ranking.
    Returns a list of (chunk_index, score) tuples.
    """
    if BM25Okapi is None:
        logger.warning("BM25Okapi not available. BM25 retrieval will be skipped.")
        return []
    
    term_list: List[str] = []
    if bm25_terms:
        for term in bm25_terms:
            if isinstance(term, str):
                cleaned = term.strip()
                if cleaned:
                    term_list.append(cleaned)

    query_text = " ".join(term_list) if term_list else prompt
    if not query_text:
        query_text = prompt

    with optional_context(
        start_span(
            name="rag.bm25_retrieval",
            input_data={"query": query_text[:200], "num_chunks": len(chunks), "top_k": top_k},
            metadata={"operation": "rag.bm25"},
        )
    ) as span:
        try:
            chunk_texts = [chunk.get("text", "").strip() for chunk in chunks]
            valid_indices = [i for i, text in enumerate(chunk_texts) if text]
            valid_texts = [chunk_texts[i] for i in valid_indices]
            if not valid_texts:
                logger.warning("No valid texts found for BM25 ranking.")
                set_span_output(
                    span,
                    output={"results": []},
                    metadata={"num_results": 0, "error": "no_valid_texts"},
                )
                return []
            
            tokenized_corpus = [text.lower().split() for text in valid_texts]
            normalized_query = re.sub(r"\bOR\b", " ", query_text, flags=re.IGNORECASE)
            normalized_query = re.sub(r"\bAND\b", " ", normalized_query, flags=re.IGNORECASE)
            normalized_query = normalized_query.replace('"', ' ')
            normalized_query = normalized_query.replace('(', ' ').replace(')', ' ')
            tokenized_query = normalized_query.lower().split()
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
            top_k_actual = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
            top_scores = scores[top_indices]
            sorted_pairs = sorted(zip(top_indices, top_scores), key=lambda x: x[1], reverse=True)
            results = [(valid_indices[idx], score) for idx, score in sorted_pairs]

            # Log BM25 results
            log_bm25_results(query_text, results, chunks)

            span_results = [
                _summarize_chunk(
                    chunks,
                    chunk_index=idx,
                    score=score,
                    score_label="bm25_score",
                    rank=position + 1,
                )
                for position, (idx, score) in enumerate(results)
            ]

            set_span_output(
                span,
                output={
                    "num_results": len(results),
                    "results": span_results,
                },
                metadata={
                    "num_results": len(results),
                    "top_score": float(results[0][1]) if results else 0.0,
                    "mean_score": float(np.mean([s for _, s in results])) if results else 0.0,
                },
            )
            return results
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}", exc_info=True)
            record_span_error(span, e, metadata={"stage": "rag.bm25_retrieval"})
            return []


async def get_semantic_search_results(
    prompt: str,
    chunks: List[Dict[str, Any]],
    model: Any,  # Embedding model (DatabricksEmbeddingModel or compatible)
    top_k: int,
    precomputed_embeddings=None,
    valid_chunk_indices=None
) -> set:
    """
    Performs semantic search to find the most relevant chunks.

    Returns:
        Tuple containing:
            - set of chunk indices corresponding to the top-k semantic matches
            - list of indices considered valid for semantic scoring (aligned with cosine scores)
            - numpy array of cosine similarity scores aligned with the valid indices
            - ordered list of (chunk_index, score) tuples for the semantic top-k results
    """
    with optional_context(
        start_span(
            name="rag.semantic_search",
            input_data={"query": prompt[:200], "num_chunks": len(chunks), "top_k": top_k},
            metadata={
                "operation": "rag.semantic_search",
                "using_precomputed": precomputed_embeddings is not None,
            },
        )
    ) as span:
        try:
            chunk_texts = [chunk.get("text", "") for chunk in chunks]
            use_precomputed = False
            if precomputed_embeddings is not None and valid_chunk_indices is not None:
                test_embedding = await run_in_threadpool(
                    model.encode,
                    "test",
                    convert_to_tensor=False,  # Ensure numpy output
                    show_progress_bar=False
                )
                if hasattr(test_embedding, 'shape'):
                    current_dim = test_embedding.shape[-1]
                else:
                    current_dim = len(test_embedding)
                if hasattr(precomputed_embeddings, 'shape'):
                    precomputed_dim = precomputed_embeddings.shape[-1]
                elif len(precomputed_embeddings) > 0:
                    first_embedding = precomputed_embeddings[0]
                    if hasattr(first_embedding, 'shape'):
                        precomputed_dim = first_embedding.shape[-1]
                    elif hasattr(first_embedding, '__len__'):
                        precomputed_dim = len(first_embedding)
                    else:
                        precomputed_dim = 0
                else:
                    precomputed_dim = 0
                if precomputed_dim == current_dim:
                    logger.info(f"Using precomputed embeddings for semantic search (dim={precomputed_dim})")
                    use_precomputed = True
                else:
                    logger.error(f"Precomputed embeddings dimension mismatch: stored={precomputed_dim}, current={current_dim}")
                    logger.error("Will regenerate embeddings with current model...")
                    use_precomputed = False
            prompt_embedding = await run_in_threadpool(
                model.encode,
                prompt,
                convert_to_tensor=False,  # Ensure numpy output
                show_progress_bar=False
            )
            if use_precomputed:
                chunk_embeddings = precomputed_embeddings
                semantic_valid_indices = list(valid_chunk_indices) if valid_chunk_indices is not None else []
                logger.info(f"Successfully using precomputed embeddings with shape: {getattr(chunk_embeddings, 'shape', 'unknown')}")
            else:
                logger.info(f"Generating embeddings for semantic search")
                semantic_valid_indices = [i for i, text in enumerate(chunk_texts) if text.strip()]
                valid_chunk_texts = [chunk_texts[i] for i in semantic_valid_indices]
                if not valid_chunk_texts:
                    logger.warning("No valid texts for semantic search.")
                    return set(), [], np.array([])
                chunk_embeddings = await run_in_threadpool(
                    model.encode,
                    valid_chunk_texts,
                    convert_to_tensor=False,  # Ensure numpy output
                    show_progress_bar=False
                )
            # Ensure numpy arrays
            prompt_embedding = np.asarray(prompt_embedding)
            chunk_embeddings = np.asarray(chunk_embeddings)
            # Debug: Check embedding dimensions
            logger.debug(f"Prompt embedding shape: {prompt_embedding.shape}")
            logger.debug(f"Chunk embeddings shape: {chunk_embeddings.shape}")
            # Ensure proper array shapes
            if prompt_embedding.ndim == 0:
                logger.error("Prompt embedding is a scalar, this should not happen")
                return set(), [], np.array([])
            if chunk_embeddings.ndim == 1:
                chunk_embeddings = chunk_embeddings[np.newaxis, :]
            elif chunk_embeddings.ndim == 0:
                logger.error("Chunk embeddings is a scalar, this should not happen")
                return set(), [], np.array([])
            prompt_dim = prompt_embedding.shape[-1]
            chunk_dim = chunk_embeddings.shape[-1]
            if prompt_dim != chunk_dim:
                logger.error(f"CRITICAL: Final dimension mismatch detected! prompt={prompt_dim}, chunks={chunk_dim}")
                logger.error(f"Prompt embedding shape: {prompt_embedding.shape}")
                logger.error(f"Chunk embeddings shape: {chunk_embeddings.shape}")
                logger.error("This should not happen after our checks. Returning empty results.")
                return set(), [], np.array([])
            # Calculate cosine similarity
            cosine_scores = numpy_cos_sim(prompt_embedding, chunk_embeddings)
            # Result is (num_chunks,)
            cosine_scores_np = np.asarray(cosine_scores)
            # Get top-k indices
            semantic_top_k = min(top_k, len(semantic_valid_indices))
            semantic_top_indices_relative = np.argpartition(cosine_scores_np, -semantic_top_k)[-semantic_top_k:]
            semantic_top_scores = cosine_scores_np[semantic_top_indices_relative]
            sorted_order = np.argsort(semantic_top_scores)[::-1]
            semantic_sorted_relative = semantic_top_indices_relative[sorted_order]
            semantic_sorted_scores = semantic_top_scores[sorted_order]

            semantic_indices = {semantic_valid_indices[i] for i in semantic_sorted_relative}

            span_results: List[Dict[str, Any]] = []
            semantic_ranked: List[Tuple[int, float]] = []
            for position, rel_idx in enumerate(semantic_sorted_relative):
                if rel_idx >= len(semantic_valid_indices):
                    continue
                absolute_index = semantic_valid_indices[rel_idx]
                semantic_ranked.append((absolute_index, float(semantic_sorted_scores[position])))
                span_results.append(
                    _summarize_chunk(
                        chunks,
                        chunk_index=absolute_index,
                        score=semantic_sorted_scores[position],
                        score_label="semantic_score",
                        rank=position + 1,
                    )
                )

            semantic_scores = [score for _, score in semantic_ranked]

            log_semantic_search_results(prompt, semantic_indices, chunks, semantic_scores)

            set_span_output(
                span,
                output={
                    "num_results": len(semantic_indices),
                    "results": span_results,
                },
                metadata={
                    "num_results": len(semantic_indices),
                    "top_score": max(semantic_scores) if semantic_scores else 0.0,
                    "mean_score": float(np.mean(semantic_scores)) if semantic_scores else 0.0,
                },
            )
            return semantic_indices, semantic_valid_indices, cosine_scores_np, semantic_ranked
        except Exception as e:
            logger.error(f"Error in semantic search: {e}", exc_info=True)
            record_span_error(span, e, metadata={"stage": "rag.semantic_search"})
            return set(), [], np.array([]), []


async def rerank_results(
    prompt: str,
    chunks: List[Dict[str, Any]],
    combined_indices: List[int],
    reranker_model,
    top_k: int,
    hyde_boosted_indices: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Reranks the combined results using a reranker model.
    Returns a list of chunk dictionaries with scores.

    This function works with both the local CrossEncoder reranker model
    and the Databricks reranker model, as both implement the predict method
    with the same interface.
    """
    with optional_context(
        start_span(
            name="rag.reranking",
            input_data={
                "query": prompt[:200],
                "num_candidates": len(combined_indices),
                "top_k": top_k,
            },
            metadata={"operation": "rag.reranking"},
        )
    ) as span:
        try:
            if not reranker_model or not combined_indices:
                set_span_output(
                    span,
                    output={"results": []},
                    metadata={"num_results": 0, "error": "no_model_or_indices"},
                )
                return []

            # Prepare pairs for reranking
            rerank_pairs = []
            for chunk_index in combined_indices:
                chunk_text = chunks[chunk_index].get("text", "")
                rerank_pairs.append([prompt, chunk_text])

            # Run reranking - works with both CrossEncoder and DatabricksRerankerModel
            # as they both implement the predict method with the same interface
            # NOTE: Run synchronously to preserve Langfuse trace context
            rerank_scores = reranker_model.predict(rerank_pairs)

            # Sort by score
            reranked_pairs = list(zip(combined_indices, rerank_scores))
            reranked_pairs.sort(key=lambda x: x[1], reverse=True)

            # Format results - IMPORTANT: Preserve ALL chunk data including bboxes for highlighting
            results = []
            for i in range(min(top_k, len(reranked_pairs))):
                chunk_index, score = reranked_pairs[i]
                chunk = chunks[chunk_index]
                # Create a copy of the original chunk to preserve all fields (especially bboxes)
                result_chunk = chunk.copy()
                # Update with reranking score and method
                result_chunk["score"] = float(score)
                result_chunk["retrieval_method"] = "hybrid"
                if hyde_boosted_indices and chunk_index in hyde_boosted_indices:
                    result_chunk["hyde_boosted"] = True
                results.append(result_chunk)

            # Log reranker results
            log_reranker_results(prompt, results)

            set_span_output(
                span,
                output={"num_results": len(results)},
                metadata={
                    "num_results": len(results),
                    "top_score": results[0]["score"] if results else 0.0,
                    "mean_score": float(np.mean([r["score"] for r in results])) if results else 0.0,
                },
            )
            return results

        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            record_span_error(span, e, metadata={"stage": "rag.reranking"})
            return []


async def retrieve_relevant_chunks_async(
    prompt: str,
    chunks: List[Dict[str, Any]],
    model: Any,  # Embedding model (DatabricksEmbeddingModel or compatible)
    top_k: int,
    precomputed_embeddings=None,
    valid_chunk_indices=None,
    reranker_model=None,
    disable_reranking=False,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    bm25_terms: Optional[List[str]] = None,
    alternate_hyde_queries: Optional[List[str]] = None,
    alternate_hyde_top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Asynchronously retrieves the top_k most relevant chunks using hybrid search (BM25 + semantic) and reranking.
    This version runs BM25 and semantic search in parallel for better performance.

    Args:
        prompt: The query text presented to the LLM and semantic search
        chunks: List of text chunks to search
        model: Embedding model (DatabricksEmbeddingModel or compatible)
        top_k: Number of top results to return
        precomputed_embeddings: Optional precomputed embeddings
        valid_chunk_indices: Optional list of valid chunk indices
        reranker_model: Optional reranker model
        disable_reranking: If True, skip reranking and return combined BM25 + semantic results
        bm25_weight: Weight for BM25 scores in hybrid combination (0-1)
        semantic_weight: Weight for semantic scores in hybrid combination (0-1)
        bm25_terms: Ordered list of lexical phrases that should be OR'ed together for BM25
        alternate_hyde_queries: Optional list of speculative HYDE phrases to seed additional semantic retrieval
        alternate_hyde_top_k: Number of HYDE-based semantic hits to request per speculative query
    """
    if not chunks or not prompt or model is None:
        logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': No chunks, prompt, or model.")
        return []

    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    if not any(chunk_texts):
        logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': All chunk texts are empty.")
        return []

    try:
        effective_bm25_terms = [term.strip() for term in (bm25_terms or []) if isinstance(term, str) and term.strip()]

        # --- Run BM25 and Semantic Search (sequential to preserve trace context) ---
        logger.info(
            "RAG: Running hybrid search with weights - BM25: %0.2f, Semantic: %0.2f",
            bm25_weight,
            semantic_weight,
        )

        if effective_bm25_terms:
            logger.info("RAG: Starting BM25 retrieval with terms: %s", effective_bm25_terms)
        else:
            logger.info("RAG: Starting BM25 retrieval with prompt fallback")

        bm25_results = get_bm25_results(
            prompt,
            chunks,
            top_k,
            bm25_terms=effective_bm25_terms or None,
        )
        logger.info("RAG: BM25 retrieval completed with %d results", len(bm25_results))

        logger.info("RAG: Starting semantic search...")
        semantic_indices, semantic_valid_indices, cosine_scores_np, semantic_ranked = await get_semantic_search_results(
            prompt=prompt,
            chunks=chunks,
            model=model,
            top_k=top_k,
            precomputed_embeddings=precomputed_embeddings,
            valid_chunk_indices=valid_chunk_indices,
        )
        logger.info("RAG: Semantic search completed with %d results", len(semantic_indices))

        # --- Step 3: Normalize scores and assemble candidate pool ---

        max_bm25_score = max([score for _, score in bm25_results]) if bm25_results else 1.0
        normalized_bm25_dict = {idx: score / max_bm25_score for idx, score in bm25_results}

        semantic_scores_dict: Dict[int, float] = {idx: score for idx, score in semantic_ranked}
        bm25_indices = {idx for idx, _ in bm25_results}

        hyde_queries_cleaned: List[str] = []
        if alternate_hyde_queries:
            for query in alternate_hyde_queries:
                if isinstance(query, str):
                    cleaned_query = query.strip()
                    if cleaned_query:
                        hyde_queries_cleaned.append(cleaned_query)

        hyde_scores_dict: Dict[int, float] = {}
        hyde_ranked_results: List[Tuple[int, float]] = []
        if hyde_queries_cleaned:
            hyde_top_k = max(1, min(alternate_hyde_top_k, top_k if top_k > 0 else alternate_hyde_top_k))
            hyde_debug_summary: List[Dict[str, Any]] = []
            for hyde_query in hyde_queries_cleaned:
                hyde_indices, hyde_valid_indices, hyde_cosine_scores, hyde_ranked = await get_semantic_search_results(
                    prompt=hyde_query,
                    chunks=chunks,
                    model=model,
                    top_k=hyde_top_k,
                    precomputed_embeddings=precomputed_embeddings,
                    valid_chunk_indices=valid_chunk_indices,
                )
                hyde_debug_summary.append({"query": hyde_query, "results": hyde_ranked})
                for idx, score in hyde_ranked:
                    existing = hyde_scores_dict.get(idx)
                    if existing is None or score > existing:
                        hyde_scores_dict[idx] = score

            if hyde_scores_dict:
                hyde_ranked_results = sorted(hyde_scores_dict.items(), key=lambda x: x[1], reverse=True)
                hyde_ranked_results = hyde_ranked_results[:hyde_top_k]
                for idx, score in hyde_ranked_results:
                    semantic_scores_dict[idx] = max(semantic_scores_dict.get(idx, 0.0), score)

            logger.info(
                "RAG: HYDE alternate semantic search produced %d candidates across %d queries",
                len(hyde_scores_dict),
                len(hyde_queries_cleaned),
            )
            logger.debug("HYDE retrieval detail: %s", hyde_debug_summary)

        combined_candidate_indices = set(bm25_indices) | set(semantic_scores_dict.keys())
        if not combined_candidate_indices:
            logger.warning("RAG: No candidates returned from BM25, semantic, or HYDE searches.")
            return []

        candidate_scores: Dict[int, float] = {}
        for idx in combined_candidate_indices:
            bm25_component = normalized_bm25_dict.get(idx, 0.0) * bm25_weight
            semantic_component = semantic_scores_dict.get(idx, 0.0) * semantic_weight
            candidate_scores[idx] = bm25_component + semantic_component

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        initial_candidates = [idx for idx, _ in sorted_candidates[:top_k]] if sorted_candidates else []

        hyde_ranked_indices = [idx for idx, _ in hyde_ranked_results]
        hyde_boosted_indices: Set[int] = set(hyde_ranked_indices)
        hyde_injected_indices: Set[int] = set()

        if initial_candidates and hyde_ranked_indices:
            hyde_new_candidates = [idx for idx in hyde_ranked_indices if idx not in initial_candidates]
            if hyde_new_candidates:
                replace_limit = max(1, int(round(top_k * 0.2))) if top_k > 0 else 1
                replace_count = min(len(initial_candidates), len(hyde_new_candidates), replace_limit)
                if replace_count > 0:
                    replaced_slice = initial_candidates[-replace_count:]
                    initial_candidates = initial_candidates[:-replace_count] + hyde_new_candidates[:replace_count]
                    hyde_injected_indices = set(hyde_new_candidates[:replace_count])
                    logger.info(
                        "RAG: Injected %d HYDE candidates replacing lowest-ranked originals for prompt '%s'",
                        replace_count,
                        prompt[:60],
                    )
                    logger.debug(
                        "HYDE injected indices %s replacing %s",
                        hyde_new_candidates[:replace_count],
                        replaced_slice,
                    )
        elif not initial_candidates and hyde_ranked_indices:
            initial_candidates = hyde_ranked_indices[:top_k] if top_k > 0 else hyde_ranked_indices
            hyde_injected_indices = set(initial_candidates)

        candidate_indices = list(dict.fromkeys(initial_candidates))

        if len(candidate_indices) < top_k:
            for idx, _ in sorted_candidates:
                if idx not in candidate_indices:
                    candidate_indices.append(idx)
                if len(candidate_indices) >= top_k:
                    break

        if len(candidate_indices) < top_k and hyde_ranked_indices:
            for idx in hyde_ranked_indices:
                if idx not in candidate_indices:
                    candidate_indices.append(idx)
                if len(candidate_indices) >= top_k:
                    break

        combined_indices = candidate_indices
        final_top_k = min(top_k, len(combined_indices))
        logger.info(
            "RAG: Candidate pool assembled with %d items (BM25=%d, semantic=%d, hyde=%d, injected=%d)",
            len(combined_indices),
            len(bm25_indices),
            len(semantic_scores_dict),
            len(hyde_ranked_indices),
            len(hyde_injected_indices),
        )

        # --- Step 4: Reranking of Combined Results ---

        if reranker_model is not None and not disable_reranking and combined_indices:
            logger.info(f"RAG: Reranking {len(combined_indices)} combined results...")
            results = await rerank_results(
                prompt=prompt,
                chunks=chunks,
                combined_indices=combined_indices,
                reranker_model=reranker_model,
                top_k=final_top_k,
                hyde_boosted_indices=hyde_boosted_indices,
            )
        else:
            logger.warning("RAG: Reranker not available, using combined scores without reranking.")

            # Format results without reranking, using normalized scores
            # IMPORTANT: Preserve ALL chunk data including bboxes for highlighting
            results = []
            for chunk_index in combined_indices[:final_top_k]:
                chunk = chunks[chunk_index]
                score = candidate_scores.get(chunk_index, 0.0)

                # Create a copy of the original chunk to preserve all fields (especially bboxes)
                result_chunk = chunk.copy()
                # Update with computed score and method
                result_chunk["score"] = score
                result_chunk["retrieval_method"] = "hybrid_no_rerank"
                if chunk_index in hyde_boosted_indices:
                    result_chunk["hyde_boosted"] = True
                results.append(result_chunk)

            # Sort results by score
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:final_top_k]

        logger.info(f"RAG: Retrieved and ranked {len(results)} chunks using parallel hybrid search.")
        return results

    except Exception as e:
        logger.error(f"Error during parallel hybrid RAG retrieval for prompt '{prompt[:50]}...': {e}", exc_info=True)
        return []


def retrieve_relevant_chunks(
    prompt: str,
    chunks: List[Dict[str, Any]],
    model: Any,  # Embedding model (DatabricksEmbeddingModel or compatible)
    top_k: int,
    precomputed_embeddings=None,
    valid_chunk_indices=None,
    reranker_model=None,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    bm25_terms: Optional[List[str]] = None,
    alternate_hyde_queries: Optional[List[str]] = None,
    alternate_hyde_top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant chunks using hybrid search (BM25 + semantic) and reranking.
    This is a synchronous wrapper around the async implementation.
    """
    from ..utils.async_utils import run_async

    return run_async(
        retrieve_relevant_chunks_async(
            prompt=prompt,
            chunks=chunks,
            model=model,
            top_k=top_k,
            precomputed_embeddings=precomputed_embeddings,
            valid_chunk_indices=valid_chunk_indices,
            reranker_model=reranker_model,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            bm25_terms=bm25_terms,
            alternate_hyde_queries=alternate_hyde_queries,
            alternate_hyde_top_k=alternate_hyde_top_k,
        )
    )


async def retrieve_relevant_chunks_for_chat_async(
    prompt: str,
    top_k_per_doc: int,
    embedding_model: Any,  # Embedding model (DatabricksEmbeddingModel or compatible)
    reranker_model=None,
    preprocessed_data=None
) -> List[Dict[str, Any]]:
    """
    Asynchronously retrieves the top_k most relevant chunks from ALL processed documents
    based on semantic similarity to the chat prompt.

    This version processes multiple documents in parallel for better performance.

    Args:
        prompt: The chat prompt/query text.
        top_k_per_doc: Number of top chunks to retrieve *per document*.
        embedding_model: The embedding model to use.
        reranker_model: Optional reranker model.
        preprocessed_data: Dictionary of preprocessed document data.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing 'filename',
                              'text', 'page_num', 'score', 'chunk_id' for the
                              most relevant chunks across all documents.
    """
    if embedding_model is None:
        logger.error("Chat RAG skipped: Embedding model not loaded.")
        return []

    if not preprocessed_data:
        logger.warning("Chat RAG skipped: No preprocessed documents found.")
        return []

    logger.info(f"Starting parallel chat RAG for prompt '{prompt[:50]}...' across {len(preprocessed_data)} documents.")

    # Create tasks for each document
    async def process_document(filename, data):
        if not data or 'chunks' not in data or 'chunk_embeddings' not in data:
            logger.warning(f"Skipping document {filename} for chat RAG: Missing required preprocessed data.")
            return []

        logger.debug(f"Running RAG for chat prompt on {filename}...")
        try:
            # Use the async version of retrieve_relevant_chunks for this document
            doc_relevant_chunks = await retrieve_relevant_chunks_async(
                prompt=prompt,
                chunks=data.get("chunks", []),
                model=embedding_model,
                top_k=top_k_per_doc,
                precomputed_embeddings=data.get("chunk_embeddings"),
                valid_chunk_indices=data.get("valid_chunk_indices"),
                reranker_model=reranker_model
            )

            # Add filename to each retrieved chunk
            for chunk in doc_relevant_chunks:
                chunk['filename'] = filename

            logger.debug(f"Retrieved {len(doc_relevant_chunks)} relevant chunks from {filename} for chat.")
            return doc_relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks for chat from {filename}: {e}", exc_info=True)
            return []

    # Process documents in parallel
    tasks = []
    for filename, data in preprocessed_data.items():
        tasks.append(process_document(filename, data))

    # Use asyncio.gather to run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Flatten the results
    all_relevant_chunks = []
    for doc_chunks in results:
        all_relevant_chunks.extend(doc_chunks)

    # Sort all combined chunks by score (highest first)
    all_relevant_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Optional: Limit the total number of chunks sent to the LLM
    # TOTAL_CHAT_CONTEXT_LIMIT = 20 # Example limit
    # all_relevant_chunks = all_relevant_chunks[:TOTAL_CHAT_CONTEXT_LIMIT]

    logger.info(f"Parallel chat RAG finished. Found {len(all_relevant_chunks)} potentially relevant chunks across all documents.")
    return all_relevant_chunks


def retrieve_relevant_chunks_for_chat(
    prompt: str,
    top_k_per_doc: int,
    embedding_model: Any,  # Embedding model (DatabricksEmbeddingModel or compatible)
    reranker_model=None,
    preprocessed_data=None
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant chunks from ALL processed documents
    based on semantic similarity to the chat prompt.

    This is a synchronous wrapper around the async implementation.

    Args:
        prompt: The chat prompt/query text.
        top_k_per_doc: Number of top chunks to retrieve *per document*.
        embedding_model: The embedding model to use.
        reranker_model: Optional reranker model.
        preprocessed_data: Dictionary of preprocessed document data.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing 'filename',
                              'text', 'page_num', 'score', 'chunk_id' for the
                              most relevant chunks across all documents.
    """
    from ..utils.async_utils import run_async

    return run_async(
        retrieve_relevant_chunks_for_chat_async(
            prompt=prompt,
            top_k_per_doc=top_k_per_doc,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            preprocessed_data=preprocessed_data
        )
    )
