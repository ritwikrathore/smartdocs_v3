"""
Chunking strategies for text processing in RAG applications.

This module provides various chunking strategies for breaking down text into
semantically meaningful chunks for retrieval and analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import spacy
from ..config import (
    logger, SENTENCES_PER_CHUNK, MIN_CHUNK_CHAR_LENGTH
)
from ..utils.helpers import normalize_text

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
            page_sentences = list(doc.sents)

            # Group sentences into chunks
            for i in range(0, len(page_sentences), self.sentences_per_chunk):
                current_sentence_group = page_sentences[i : i + self.sentences_per_chunk]
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
        chunking_strategy = SentenceChunker()

    return chunking_strategy.create_chunks(text, page_num=page_num, **kwargs)
