from pathlib import Path
import statistics

import spacy

from src.keyword_code.config import MIN_CHUNK_CHAR_LENGTH
from src.keyword_code.rag.chunking import SentenceChunker, AdaptiveSentenceChunker


def _load_sample_text() -> str:
    sample_path = Path(__file__).parent / "data" / "chunking_sample.txt"
    return sample_path.read_text(encoding="utf-8")


def _build_test_nlp():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _lengths(chunks):
    return [len(chunk["text"]) for chunk in chunks]


def test_adaptive_chunker_stabilizes_chunk_lengths():
    text = _load_sample_text()
    nlp = _build_test_nlp()

    baseline_chunker = SentenceChunker(
        sentences_per_chunk=6,
        min_chunk_char_length=MIN_CHUNK_CHAR_LENGTH,
        nlp=nlp,
    )
    adaptive_chunker = AdaptiveSentenceChunker(nlp=nlp)

    baseline_chunks = baseline_chunker.create_chunks(text)
    adaptive_chunks = adaptive_chunker.create_chunks(text)

    assert baseline_chunks, "Baseline chunker failed to produce chunks"
    assert adaptive_chunks, "Adaptive chunker failed to produce chunks"

    baseline_lengths = _lengths(baseline_chunks)
    adaptive_lengths = _lengths(adaptive_chunks)

    # Baseline should show high variance: some tiny bullets and some huge paragraphs
    assert min(baseline_lengths) < 220, "Expected tiny baseline chunk to demonstrate current issue"
    assert max(baseline_lengths) > 1200, "Expected oversized baseline chunk for comparison"

    # Adaptive strategy should keep chunks within the desired character corridor
    assert min(adaptive_lengths) >= 400, "Adaptive chunker still producing undersized chunks"
    assert max(adaptive_lengths) <= 950, "Adaptive chunker still producing oversized chunks"

    # Overall dispersion should improve noticeably
    baseline_std = statistics.pstdev(baseline_lengths)
    adaptive_std = statistics.pstdev(adaptive_lengths)
    debug_summary = (
        f"Baseline lengths: {baseline_lengths} | "
        f"Adaptive lengths: {adaptive_lengths} | "
        f"σ baseline/adaptive: {baseline_std:.1f}/{adaptive_std:.1f}"
    )
    assert adaptive_std < baseline_std, debug_summary
