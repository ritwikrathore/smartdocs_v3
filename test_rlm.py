"""
Test script for the RLM (Recursive Language Model) engine.

Loads Sample_Moodys_Rating_Change.pdf, chunks it with metadata,
runs the RLM engine, and prints the results.
"""

import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from keyword_code.processors.pdf_processor import PDFProcessor
from keyword_code.ai.databricks_llm import DatabricksLLMClient
from keyword_code.rlm import RLMEngine


def main():
    pdf_path = Path(__file__).parent / "Sample_Moodys_Rating_Change.pdf"
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    # --- Step 1: Load and process PDF ---
    print("=" * 60)
    print("STEP 1: Loading and processing PDF")
    print("=" * 60)
    pdf_bytes = pdf_path.read_bytes()
    processor = PDFProcessor(pdf_bytes)
    chunks = processor.chunks
    full_text = processor.full_text

    print(f"  Chunks: {len(chunks)}")
    print(f"  Full text length: {len(full_text):,} chars")
    print(f"  Pages: {max((c.get('page_num', 0) for c in chunks), default=0) + 1}")

    if chunks:
        sample = chunks[0]
        print(f"\n  Sample chunk keys: {list(sample.keys())}")
        print(f"  Sample chunk text (first 150 chars): {sample['text'][:150]}...")
        if "metadata" in sample:
            print(f"  Sample metadata: {sample['metadata']}")

    # --- Step 2: Initialize RLM engine ---
    print("\n" + "=" * 60)
    print("STEP 2: Initializing RLM engine")
    print("=" * 60)
    llm_client = DatabricksLLMClient()
    engine = RLMEngine(llm_client=llm_client, verbose=True)
    print("  RLM engine ready")

    # --- Step 3: Run RLM ---
    question = (
        "What rating changes are described in this document? "
        "Summarize the key details including the entities, old ratings, "
        "new ratings, and reasons."
    )
    print("\n" + "=" * 60)
    print("STEP 3: Running RLM loop")
    print("=" * 60)
    print(f"  Question: {question}\n")

    start_time = time.time()
    result = engine.completion(chunks=chunks, full_text=full_text, question=question)
    elapsed = time.time() - start_time

    # --- Step 4: Print results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n--- Final Answer ---")
    print(result.answer)

    print(f"\n--- Citations ---")
    if result.citations:
        for i, cite in enumerate(result.citations, 1):
            print(f"  [{i}] chunk_id={cite.get('chunk_id', '?')}, "
                  f"page={cite.get('page_num', '?')}, "
                  f"section={cite.get('section', '?')}")
            snippet = cite.get("text_snippet", "")
            if snippet:
                print(f"      \"{snippet[:120]}...\"" if len(snippet) > 120 else f"      \"{snippet}\"")
    else:
        print("  No structured citations returned.")

    print(f"\n--- Execution Stats ---")
    print(f"  Iterations: {result.iterations}")
    print(f"  Total tokens: {result.total_tokens:,}")
    print(f"  Wall time: {elapsed:.1f}s")

    print(f"\n--- Execution Log ---")
    for entry in result.execution_log:
        it = entry["iteration"]
        n_blocks = entry["code_blocks_found"]
        errors = sum(1 for r in entry.get("results", []) if r.get("error"))
        finals = sum(1 for r in entry.get("results", []) if r.get("has_final"))
        print(f"  Iter {it}: {n_blocks} code blocks, {errors} errors, "
              f"{'FINAL found' if finals else 'continuing'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
