"""
Chunking strategies for text processing in RAG applications.

This module provides various chunking strategies for breaking down text into
semantically meaningful chunks for retrieval and analysis.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import spacy
from ..config import (
    logger,
    SENTENCES_PER_CHUNK,
    MIN_CHUNK_CHAR_LENGTH,
    USE_ADAPTIVE_SENTENCE_CHUNKER,
    ADAPTIVE_CHUNK_MIN_CHARS,
    ADAPTIVE_CHUNK_MAX_CHARS,
    ADAPTIVE_CHUNK_MIN_SENTENCES,
    ADAPTIVE_CHUNK_MAX_SENTENCES,
    ADAPTIVE_CHUNK_OVERLAP_SENTENCES,
    CHUNK_OVERLAP_RATIO,
)
from ..text_utils import normalize_text

class ChunkingStrategy:
    """Base class for all chunking strategies"""

    def __init__(self, name: str):
        self.name = name

    def create_chunks(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Create chunks from text and return them as a list of dictionaries"""
        raise NotImplementedError("Subclasses must implement this method")


class SentenceChunker(ChunkingStrategy):
    """
    Chunks text based on sentences using spaCy for sentence boundary detection.

    This chunker groups a specified number of sentences together to form chunks,
    which helps maintain semantic coherence better than fixed-size chunking.
    """

    def __init__(self, sentences_per_chunk: int = None, min_chunk_char_length: int = None, nlp=None):
        """
        Initialize the sentence chunker.

        Args:
            sentences_per_chunk: Number of sentences to include in each chunk (defaults to config value)
            min_chunk_char_length: Minimum character length for a chunk to be valid (defaults to config value)
            nlp: Optional pre-loaded spaCy model. If None, will attempt to load one.
        """
        # Use values from config if not provided
        self.sentences_per_chunk = sentences_per_chunk if sentences_per_chunk is not None else SENTENCES_PER_CHUNK
        self.min_chunk_char_length = min_chunk_char_length if min_chunk_char_length is not None else MIN_CHUNK_CHAR_LENGTH

        super().__init__(f"sentence_based_{self.sentences_per_chunk}sentences")
        self._nlp = nlp

        if self._nlp is None:
            try:
                from ..utils.spacy_utils import ensure_spacy_model
                self._nlp = ensure_spacy_model("en_core_web_sm")
                if self._nlp is None:
                    logger.error(
                        "Failed to load spaCy model 'en_core_web_sm'. "
                        "Text chunking will not work properly."
                    )
                else:
                    logger.info("spaCy model 'en_core_web_sm' loaded successfully for chunking.")
            except Exception as e:
                logger.error(f"Error loading spaCy model: {str(e)}")
                self._nlp = None

    def _split_sentences_on_headers(self, sentences: List[Any]) -> List[List[Any]]:
        """Split sentences into groups based on structural headers."""
        HEADER_PATTERNS = [
            re.compile(r"^ARTICLE\s+[IVXLCDM\d]+", re.IGNORECASE),
            re.compile(r"^Section\s+\d+", re.IGNORECASE),
            re.compile(r"^ANNEX\s+[A-Z0-9]+", re.IGNORECASE),
            re.compile(r"^SCHEDULE\s+[A-Z0-9]+", re.IGNORECASE),
        ]
        groups = []
        current_group = []
        for sent in sentences:
            # Check if sentence starts with a header
            text = sent.text.strip()
            if any(p.match(text) for p in HEADER_PATTERNS) and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(sent)
        if current_group:
            groups.append(current_group)
        return groups

    def create_chunks(self, text: str, page_num: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Create chunks from text by grouping sentences.

        Args:
            text: The text to chunk
            page_num: Optional page number for the text (for PDF documents)
            **kwargs: Additional arguments

        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []
        current_chunk_id_counter = kwargs.get("start_chunk_id", 0)

        if not self._nlp:
            logger.error("spaCy model not loaded. Cannot perform sentence-based chunking.")
            return chunks

        if not text or not text.strip():
            return chunks

        try:
            # Process the text with spaCy to get sentences
            doc = self._nlp(text)
            all_sentences = list(doc.sents)
            
            # Split sentences by headers to enforce structural boundaries
            sentence_groups = self._split_sentences_on_headers(all_sentences)

            # Support percentage-based overlap via `overlap_ratio` kwarg (0.0 - 1.0).
            # If not provided, fall back to non-overlapping sliding windows.
            overlap_ratio = None
            # Prefer explicit kwarg, then config value, then legacy integer overlap
            if "overlap_ratio" in kwargs:
                try:
                    overlap_ratio = float(kwargs.get("overlap_ratio"))
                except Exception:
                    overlap_ratio = None
            if overlap_ratio is None and CHUNK_OVERLAP_RATIO is not None:
                overlap_ratio = CHUNK_OVERLAP_RATIO

            if overlap_ratio is not None and 0.0 <= overlap_ratio < 1.0:
                overlap_sentences = int(round(self.sentences_per_chunk * overlap_ratio))
            else:
                # Fallback to explicit overlap_sentences if passed, else 0
                overlap_sentences = int(kwargs.get("overlap_sentences", 0) or 0)

            step = max(1, self.sentences_per_chunk - overlap_sentences)

            for group_sentences in sentence_groups:
                # Group sentences into chunks using sliding window with computed step
                for i in range(0, len(group_sentences), step):
                    current_sentence_group = group_sentences[i : i + self.sentences_per_chunk]
                    # Concatenate text of sentences in the current group
                    chunk_text_from_sentences = " ".join([sent.text for sent in current_sentence_group]).strip()

                    # Skip if chunk is too short or empty after normalization
                    if not chunk_text_from_sentences or \
                       len(normalize_text(chunk_text_from_sentences)) < self.min_chunk_char_length:
                        continue

                    chunk_id_str = f"chunk_{current_chunk_id_counter}"
                    chunk = {
                        "chunk_id": chunk_id_str,
                        "text": chunk_text_from_sentences,
                    }

                    # Add page number if provided
                    if page_num is not None:
                        chunk["page_num"] = page_num

                    chunks.append(chunk)
                    current_chunk_id_counter += 1

        except Exception as e:
            logger.error(f"Failed to create sentence-based chunks: {str(e)}", exc_info=True)

        return chunks


class AdaptiveSentenceChunker(SentenceChunker):
    """Sentence chunker that enforces character bounds and overlap to stabilize chunk sizes."""

    def __init__(
        self,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        min_sentences: Optional[int] = None,
        max_sentences: Optional[int] = None,
        overlap_sentences: Optional[int] = None,
        min_chunk_char_length: Optional[int] = None,
        nlp=None,
    ):
        super().__init__(nlp=nlp)
        self.target_min_chars = min_chars if min_chars is not None else ADAPTIVE_CHUNK_MIN_CHARS
        self.target_max_chars = max_chars if max_chars is not None else ADAPTIVE_CHUNK_MAX_CHARS
        self.min_sentences = min_sentences if min_sentences is not None else ADAPTIVE_CHUNK_MIN_SENTENCES
        self.max_sentences = max_sentences if max_sentences is not None else ADAPTIVE_CHUNK_MAX_SENTENCES
        self.overlap_sentences = overlap_sentences if overlap_sentences is not None else ADAPTIVE_CHUNK_OVERLAP_SENTENCES
        self.min_chunk_char_length = (
            min_chunk_char_length if min_chunk_char_length is not None else MIN_CHUNK_CHAR_LENGTH
        )
        self.merge_threshold_chars = max(self.min_chunk_char_length, int(self.target_min_chars * 0.5))
        self.name = "adaptive_sentence_chunker"

    def create_chunks(self, text: str, page_num: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        if not self._nlp:
            logger.error("spaCy model not loaded. Cannot perform adaptive sentence-based chunking.")
            return []

        if not text or not text.strip():
            return []

        try:
            doc = self._nlp(text)
        except Exception as exc:
            logger.error("Failed to process text with spaCy: %s", exc, exc_info=True)
            return []

        all_sentences = [sent for sent in doc.sents if sent.text and sent.text.strip()]
        if not all_sentences:
            return []

        # Split sentences by headers to enforce structural boundaries
        sentence_groups = self._split_sentences_on_headers(all_sentences)

        chunks: List[Dict[str, Any]] = []
        chunk_id_counter = kwargs.get("start_chunk_id", 0)

        for sentences in sentence_groups:
            cursor = 0
            total_sentences = len(sentences)

            while cursor < total_sentences:
                start_idx = cursor
                buffer: List[Any] = []
                char_count = 0

                # Always capture at least min_sentences unless text runs out.
                while cursor < total_sentences and (
                    len(buffer) < self.min_sentences or len(buffer) == 0
                ):
                    buffer.append(sentences[cursor])
                    char_count += len(sentences[cursor].text.strip())
                    cursor += 1

                # Keep adding sentences until reaching the lower char bound.
                while cursor < total_sentences and char_count < self.target_min_chars and len(buffer) < self.max_sentences:
                    buffer.append(sentences[cursor])
                    char_count += len(sentences[cursor].text.strip())
                    cursor += 1

                # Add more sentences only if they do not break the max thresholds too badly.
                while (
                    cursor < total_sentences
                    and len(buffer) < self.max_sentences
                    and (char_count + len(sentences[cursor].text.strip())) <= self.target_max_chars
                ):
                    buffer.append(sentences[cursor])
                    char_count += len(sentences[cursor].text.strip())
                    cursor += 1

                # If we overshot the max char limit while satisfying minimum constraints,
                # peel sentences off the end until we fall back under the ceiling.
                while buffer and char_count > self.target_max_chars and len(buffer) > self.min_sentences:
                    cursor -= 1
                    removed_sentence = buffer.pop()
                    char_count -= len(removed_sentence.text.strip())

                chunk_text = " ".join(sent.text.strip() for sent in buffer).strip()
                if not chunk_text:
                    continue

                normalized_len = len(normalize_text(chunk_text))
                if normalized_len < self.merge_threshold_chars and chunks:
                    # Merge undersized tail into previous chunk to avoid orphaned fragments.
                    # Only merge if it's NOT the first chunk of a new group (to preserve boundaries)
                    is_first_chunk_of_group = (start_idx == 0)
                    
                    if not is_first_chunk_of_group:
                        prev_chunk = chunks[-1]
                        prev_chunk["text"] = f"{prev_chunk['text'].rstrip()} {chunk_text}".strip()
                        prev_span = prev_chunk.get("sentence_span")
                        if prev_span:
                            prev_chunk["sentence_span"] = (prev_span[0], start_idx + len(buffer))
                        else:
                            prev_chunk["sentence_span"] = (0, start_idx + len(buffer))
                        continue

                chunk_payload: Dict[str, Any] = {
                    "chunk_id": f"chunk_{chunk_id_counter}",
                    "text": chunk_text,
                    "sentence_span": (start_idx, start_idx + len(buffer)),
                }

                if page_num is not None:
                    chunk_payload["page_num"] = page_num

                chunks.append(chunk_payload)
                chunk_id_counter += 1

                if self.overlap_sentences > 0:
                    overlap = min(self.overlap_sentences, len(buffer) - 1 if len(buffer) > 1 else 0)
                    cursor = max(start_idx + 1, cursor - overlap)

        return chunks


def build_default_chunker(nlp=None) -> ChunkingStrategy:
    """Return the configured default chunker instance."""
    if USE_ADAPTIVE_SENTENCE_CHUNKER:
        return AdaptiveSentenceChunker(nlp=nlp)
    return SentenceChunker(nlp=nlp)


def create_chunks_from_text(
    text: str,
    chunking_strategy: Optional[ChunkingStrategy] = None,
    page_num: Optional[int] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Create chunks from text using the specified chunking strategy.

    Args:
        text: The text to chunk
        chunking_strategy: The chunking strategy to use (defaults to SentenceChunker if None)
        page_num: Optional page number for the text (for PDF documents)
        **kwargs: Additional arguments to pass to the chunking strategy

    Returns:
        List of dictionaries containing chunk information
    """
    if chunking_strategy is None:
        chunking_strategy = build_default_chunker()

    return chunking_strategy.create_chunks(text, page_num=page_num, **kwargs)
