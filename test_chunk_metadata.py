"""
Test script to evaluate chunk metadata extraction on sample.pdf.

This script:
1. Extracts chunks from sample.pdf using PDFProcessor
2. Counts how many chunks have complete metadata (article_number + section_number)
3. Outputs detailed chunk-wise metadata to a report file for analysis
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyword_code.processors.pdf_processor import PDFProcessor


def main():
    pdf_path = Path("sample.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    print("Loading PDF and extracting chunks...")
    with pdf_path.open("rb") as f:
        processor = PDFProcessor(f.read())
    
    chunks, _ = processor.extract_structured_text_and_chunks()
    
    if not chunks:
        print("Error: No chunks extracted")
        return
    
    print(f"\nTotal chunks extracted: {len(chunks)}")
    
    # Count chunks with metadata
    chunks_with_metadata = sum(1 for c in chunks if c.get("metadata"))
    print(f"Chunks with metadata field: {chunks_with_metadata}")
    
    # Count chunks with complete article/section tagging
    chunks_with_article_section = 0
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("article_number") and metadata.get("section_number"):
            chunks_with_article_section += 1
    
    print(f"Chunks with both article_number and section_number: {chunks_with_article_section}")
    
    # Calculate coverage percentage
    coverage_pct = (chunks_with_article_section / len(chunks)) * 100 if chunks else 0
    print(f"Metadata coverage: {coverage_pct:.2f}%")
    
    # Write detailed report
    report_path = Path("tmp/chunk_metadata_test_report.txt")
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"total_chunks={len(chunks)}\n")
        f.write(f"chunks_with_metadata={chunks_with_metadata}\n")
        f.write(f"chunks_with_article_section={chunks_with_article_section}\n")
        f.write(f"coverage_pct={coverage_pct:.2f}\n")
        f.write("chunks:\n")
        
        for chunk in chunks:
            f.write("---\n")
            f.write(f"chunk_id={chunk.get('chunk_id')}\n")
            f.write(f"page={chunk.get('page_num')}\n")
            
            metadata = chunk.get("metadata", {})
            if metadata:
                # Write metadata as key=value pairs
                f.write(f"metadata={metadata}\n")
            else:
                f.write("metadata=None\n")
            
            # Write a text preview (first 300 chars)
            text_preview = (chunk.get("text", "")[:300]).replace("\n", " ")
            f.write(f"text_preview={text_preview}\n")
    
    print(f"\nDetailed report written to: {report_path}")
    
    # Print first 50 chunks as examples
    print("\n=== Sample Chunks ===")
    for i, chunk in enumerate(chunks[75:85], 1):
        print(f"\n--- Chunk {i} (ID: {chunk.get('chunk_id')}, Page: {chunk.get('page_num')}) ---")
        metadata = chunk.get("metadata", {})
        print(f"Article: {metadata.get('article_number')} - {metadata.get('article_title')}")
        print(f"Section: {metadata.get('section_number')} - {metadata.get('section_title')}")
        if metadata.get("subsection_label"):
            print(f"Subsection: {metadata.get('subsection_label')} - {metadata.get('subsection_title')}")
        print(f"Hierarchy: {' > '.join(metadata.get('hierarchy_path', []))}")
        print(f"Text preview: {chunk.get('text', '')[:150]}...")


if __name__ == "__main__":
    main()
