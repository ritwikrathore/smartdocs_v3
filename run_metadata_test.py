"""
Quick test runner for chunk metadata validation.
Run this with: python tmp/run_metadata_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress logging for cleaner output
import logging
logging.getLogger('src.keyword_code').setLevel(logging.ERROR)

from src.keyword_code.processors.pdf_processor import PDFProcessor

def main():
    pdf_path = Path("sample.pdf")
    
    print("Processing sample.pdf...")
    with pdf_path.open("rb") as f:
        processor = PDFProcessor(f.read())
    
    chunks, _ = processor.extract_structured_text_and_chunks()
    
    # Calculate stats
    total = len(chunks)
    with_meta = sum(1 for c in chunks if c.get("metadata"))
    with_article_section = sum(
        1 for c in chunks 
        if c.get("metadata", {}).get("article_number") and c.get("metadata", {}).get("section_number")
    )
    
    coverage = (with_article_section / total * 100) if total > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"  Total chunks: {total}")
    print(f"  With metadata: {with_meta}")
    print(f"  With Article+Section: {with_article_section}")
    print(f"  Coverage: {coverage:.1f}%")
    
    # Write report
    report_path = Path("tmp/chunk_metadata_test_report.txt")
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"total_chunks={total}\n")
        f.write(f"chunks_with_metadata={with_meta}\n")
        f.write(f"chunks_with_article_section={with_article_section}\n")
        f.write(f"coverage_pct={coverage:.2f}\n")
        f.write("chunks:\n")
        
        for chunk in chunks:
            f.write("---\n")
            f.write(f"chunk_id={chunk.get('chunk_id')}\n")
            f.write(f"page={chunk.get('page_num')}\n")
            metadata = chunk.get("metadata", {})
            f.write(f"metadata={metadata}\n")
            text_preview = (chunk.get("text", "")[:300]).replace("\n", " ")
            f.write(f"text_preview={text_preview}\n")
    
    print(f"\n✓ Report: {report_path}")
    
    # Show first 3 examples
    print("\nSAMPLE CHUNKS:")
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        meta = chunk.get("metadata", {})
        print(f"\n  [{chunk.get('chunk_id')}] Page {chunk.get('page_num')}")
        print(f"    Article: {meta.get('article_number')} - {meta.get('article_title')}")
        print(f"    Section: {meta.get('section_number')}")
        print(f"    Path: {' > '.join(meta.get('hierarchy_path', []))}")

if __name__ == "__main__":
    main()
