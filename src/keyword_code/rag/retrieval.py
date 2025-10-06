"""
Retrieval functionality for RAG.
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set, Callable
from ..config import logger, RAG_TOP_K, RAG_WORKERS
from ..utils.async_utils import run_tasks_in_parallel, run_in_threadpool
from ..utils.interaction_logger import (
    log_bm25_results,
    log_semantic_search_results,
    log_reranker_results
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


def get_bm25_results(prompt: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Tuple[int, float]]:
    """
    Retrieves the top_k most relevant chunks using BM25 ranking.
    Returns a list of (chunk_index, score) tuples.
    """
    if BM25Okapi is None:
        logger.warning("BM25Okapi not available. BM25 retrieval will be skipped.")
        return []
    chunk_texts = [chunk.get("text", "").strip() for chunk in chunks]
    valid_indices = [i for i, text in enumerate(chunk_texts) if text]
    valid_texts = [chunk_texts[i] for i in valid_indices]
    if not valid_texts:
        logger.warning("No valid texts found for BM25 ranking.")
        return []
    try:
        tokenized_corpus = [text.lower().split() for text in valid_texts]
        tokenized_query = prompt.lower().split()
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        top_k_actual = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
        top_scores = scores[top_indices]
        sorted_pairs = sorted(zip(top_indices, top_scores), key=lambda x: x[1], reverse=True)
        results = [(valid_indices[idx], score) for idx, score in sorted_pairs]

        # Log BM25 results
        log_bm25_results(prompt, results, chunks)

        return results
    except Exception as e:
        logger.error(f"Error in BM25 retrieval: {e}", exc_info=True)
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
    Returns a set of chunk indices.
    """
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
            semantic_valid_indices = valid_chunk_indices
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
        semantic_sorted_indices = semantic_top_indices_relative[np.argsort(semantic_top_scores)[::-1]]
        semantic_indices = {semantic_valid_indices[i] for i in semantic_sorted_indices}
        semantic_scores = [float(semantic_top_scores[i]) for i in np.argsort(semantic_top_scores)[::-1]]
        log_semantic_search_results(prompt, semantic_indices, chunks, semantic_scores)
        return semantic_indices, semantic_valid_indices, cosine_scores_np
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        return set(), [], np.array([])


async def rerank_results(
    prompt: str,
    chunks: List[Dict[str, Any]],
    combined_indices: List[int],
    reranker_model,
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Reranks the combined results using a reranker model.
    Returns a list of chunk dictionaries with scores.

    This function works with both the local CrossEncoder reranker model
    and the Databricks reranker model, as both implement the predict method
    with the same interface.
    """
    try:
        if not reranker_model or not combined_indices:
            return []

        # Prepare pairs for reranking
        rerank_pairs = []
        for chunk_index in combined_indices:
            chunk_text = chunks[chunk_index].get("text", "")
            rerank_pairs.append([prompt, chunk_text])

        # Run reranking - works with both CrossEncoder and DatabricksRerankerModel
        # as they both implement the predict method with the same interface
        rerank_scores = await run_in_threadpool(reranker_model.predict, rerank_pairs)

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
            results.append(result_chunk)

        # Log reranker results
        log_reranker_results(prompt, results)

        return results

    except Exception as e:
        logger.error(f"Error in reranking: {e}", exc_info=True)
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
    semantic_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Asynchronously retrieves the top_k most relevant chunks using hybrid search (BM25 + semantic) and reranking.
    This version runs BM25 and semantic search in parallel for better performance.

    Args:
        prompt: The query text
        chunks: List of text chunks to search
        model: Embedding model (DatabricksEmbeddingModel or compatible)
        top_k: Number of top results to return
        precomputed_embeddings: Optional precomputed embeddings
        valid_chunk_indices: Optional list of valid chunk indices
        reranker_model: Optional reranker model
        disable_reranking: If True, skip reranking and return combined BM25 + semantic results
        bm25_weight: Weight for BM25 scores in hybrid combination (0-1)
        semantic_weight: Weight for semantic scores in hybrid combination (0-1)
    """
    if not chunks or not prompt or model is None:
        logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': No chunks, prompt, or model.")
        return []

    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    if not any(chunk_texts):
        logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': All chunk texts are empty.")
        return []

    try:
        # --- Run BM25 and Semantic Search in parallel ---
        logger.info(f"RAG: Running hybrid search with weights - BM25: {bm25_weight:.2f}, Semantic: {semantic_weight:.2f}")
        logger.info("RAG: Running BM25 and semantic search in parallel...")

        # Define tasks to run in parallel
        async def bm25_task():
            logger.info("RAG: Starting BM25 retrieval...")
            result = await run_in_threadpool(get_bm25_results, prompt, chunks, top_k)
            logger.info(f"RAG: BM25 retrieval completed with {len(result)} results")
            return result

        async def semantic_task():
            logger.info("RAG: Starting semantic search...")
            result, valid_indices, scores = await get_semantic_search_results(
                prompt=prompt,
                chunks=chunks,
                model=model,
                top_k=top_k,
                precomputed_embeddings=precomputed_embeddings,
                valid_chunk_indices=valid_chunk_indices
            )
            logger.info(f"RAG: Semantic search completed with {len(result)} results")
            return result, valid_indices, scores

        # Run both tasks in parallel
        bm25_results, (semantic_indices, semantic_valid_indices, cosine_scores_np) = await asyncio.gather(bm25_task(), semantic_task())

        # --- Step 3: Normalize and Combine Results ---

        # Normalize BM25 scores to [0-1] range
        max_bm25_score = max([score for _, score in bm25_results]) if bm25_results else 1.0
        normalized_bm25_dict = {idx: score/max_bm25_score for idx, score in bm25_results}

        # Create dict of semantic scores
        semantic_scores_dict = {}
        if len(semantic_indices) > 0 and len(semantic_valid_indices) > 0 and cosine_scores_np.size > 0:
            for idx in semantic_indices:
                if idx in semantic_valid_indices:
                    rel_idx = semantic_valid_indices.index(idx)
                    if rel_idx < len(cosine_scores_np):
                        semantic_scores_dict[idx] = float(cosine_scores_np[rel_idx])

        # Extract indices from both results
        bm25_indices = {idx for idx, _ in bm25_results}

        # Combine unique indices from both methods
        combined_indices = list(bm25_indices | semantic_indices)
        logger.info(f"RAG: Combined {len(bm25_indices)} BM25 results and {len(semantic_indices)} semantic results into {len(combined_indices)} unique indices")

        # --- Step 4: Reranking of Combined Results ---
        final_top_k = min(top_k, len(combined_indices))

        if reranker_model is not None and not disable_reranking and combined_indices:
            logger.info(f"RAG: Reranking {len(combined_indices)} combined results...")
            results = await rerank_results(
                prompt=prompt,
                chunks=chunks,
                combined_indices=combined_indices,
                reranker_model=reranker_model,
                top_k=final_top_k
            )
        else:
            logger.warning("RAG: Reranker not available, using combined scores without reranking.")

            # Format results without reranking, using normalized scores
            # IMPORTANT: Preserve ALL chunk data including bboxes for highlighting
            results = []
            for chunk_index in combined_indices[:final_top_k]:
                chunk = chunks[chunk_index]
                # Get normalized scores (default to 0.0 if not found)
                bm25_score = normalized_bm25_dict.get(chunk_index, 0.0)
                semantic_score = semantic_scores_dict.get(chunk_index, 0.0)

                # Weighted combination of available scores
                # If a chunk was found by only one method, still consider it
                if chunk_index in bm25_indices and chunk_index in semantic_indices:
                    score = (bm25_score * bm25_weight) + (semantic_score * semantic_weight)
                elif chunk_index in bm25_indices:
                    score = bm25_score * bm25_weight
                else:
                    score = semantic_score * semantic_weight

                # Create a copy of the original chunk to preserve all fields (especially bboxes)
                result_chunk = chunk.copy()
                # Update with computed score and method
                result_chunk["score"] = score
                result_chunk["retrieval_method"] = "hybrid_no_rerank"
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
    semantic_weight: float = 0.5
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
            semantic_weight=semantic_weight
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
