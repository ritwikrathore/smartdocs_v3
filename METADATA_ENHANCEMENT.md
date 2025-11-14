# Chunk Metadata Enhancement Summary

## What Was Done

### 1. **DocumentStructureTracker Class** (pdf_processor.py, lines 21-350+)
Created a sophisticated parser that tracks document structure as it processes chunks:

**Regex Patterns:**
- `_ARTICLE_PATTERN`: Matches "ARTICLE I", "ARTICLE 1", etc.
- `_SECTION_PATTERN`: Matches "Section 1.01", "Section 2.03(a)", etc.
- `_ANNEX_PATTERN`: Matches "ANNEX A", "ANNEX B", etc.
- `_SCHEDULE_PATTERN`: Matches "SCHEDULE 1", "SCHEDULE 2", etc.
- `_RECITAL_PATTERN`: Matches "RECITALS", "RECITAL (A)", etc.
- `_SUBSECTION_PATTERN`: Matches "(a)", "(i)", "(1)", etc.
- `_PAGE_MARKER_PATTERN`: Detects page numbers to skip
- `_HEADING_BOUNDARY_PATTERN`: Identifies where headings end
- `_TOC_ENTRY_PATTERN`: **NEW** - Extracts structured entries from table of contents

**State Tracking:**
- Maintains current article, section, and subsection context
- Handles pending titles that appear on the next line
- **Extracts and preserves** table of contents with page mappings
- Builds hierarchical paths like: ["Article I - Definitions", "Section 1.01 - Definitions"]

**Metadata Fields Added to Each Chunk:**
```python
{
    "document_scope": "article" | "annex" | "schedule" | "recital" | "table_of_contents" | "preamble",
    "article_type": "Article" | "Annex" | "Schedule" | "Recitals" | "Preamble" | "Table of Contents",
    "article_number": "I" | "1" | "A" | "Recitals",
    "article_title": "Definitions and Interpretation",
    "section_number": "Section 1.01",
    "section_title": "Definitions",
    "subsection_label": "a" | "i" | "1",
    "subsection_title": "specific subsection content",
    "hierarchy_path": ["Article I - Definitions", "Section 1.01 - Definitions"],
    "toc_entries": [  # NEW: Only for table_of_contents chunks
        {"entry": "ARTICLE I", "page_number": 6, "raw_text": "ARTICLE I ......6"},
        {"entry": "Section 1.01", "page_number": 6, "raw_text": "Section 1.01. Definitions ......6"}
    ]
}
```

### 2. **Integration with Chunking** (pdf_processor.py, line 603)
- Added `_assign_chunk_metadata()` method that runs after chunk extraction
- Processes each chunk's text through DocumentStructureTracker
- Gracefully handles errors with logging

### 3. **Circular Import Fix** (display_utils/pdf_utils.py, lines 1-9)
- Changed PDFProcessor import to use `TYPE_CHECKING` guard
- Made import lazy (only at function call time) to break circular dependency
- Maintains type hints for development

### 4. **Test Scripts Created**
- `tmp/test_chunk_metadata.py`: Full-featured test with detailed reporting
- `tmp/run_metadata_test.py`: Simplified runner with clean output
- `tmp/test_import.py`: Minimal import validation

### 5. **RAG Integration** (ai/analyzer.py, app.py)
**Chunk Metadata in Retrieval Context:**
- All retrieved chunks now include their hierarchical location (Article/Section/Subsection) in the context
- BM25, semantic search, and reranker all work with metadata-enriched chunks
- Analyzer receives formatted location info like: `"LOCATION: Article I - Definitions and Interpretation - Section 1.01 - Definitions"`
- This helps the LLM understand document structure and provide more accurate section references

**Table of Contents as Default Context:**
- TOC is automatically extracted from chunks with `document_scope: "table_of_contents"`
- Structured TOC entries (section name + page number) are included in every analysis request
- LLM receives document structure overview before analyzing sub-prompts
- Example TOC format provided to LLM:
  ```
  Document Structure (Table of Contents):
    - ARTICLE I: Page 6
    - Section 1.01. Definitions: Page 6
    - ARTICLE II: Page 22
    - ANNEX A: Page 66
  ```

**Benefits:**
- More accurate section references in analysis responses
- Better context awareness across document structure
- Improved cross-referencing between related sections
- Enables future navigation features using TOC page mappings

## How to Run the Test

```bash
# From project root:
python tmp/run_metadata_test.py
```

This will:
1. Load sample.pdf
2. Extract all chunks with metadata
3. Calculate coverage percentage (chunks with both article_number AND section_number)
4. Generate `tmp/chunk_metadata_test_report.txt` with detailed chunk-by-chunk analysis
5. Print summary and sample chunks to console

## Expected Output Format

**Console:**
```
RESULTS:
  Total chunks: 450
  With metadata: 450
  With Article+Section: 423
  Coverage: 94.0%

SAMPLE CHUNKS:
  [chunk_0] Page 0
    Article: Preamble - Introductory Statements
    Section: Preamble
    Path: Preamble > Preamble - Introductory Statements

  [chunk_5] Page 5
    Article: I - Definitions and Interpretation
    Section: Section 1.01
    Path: Article I - Definitions and Interpretation > Section 1.01 - Definitions
```

**Report File (tmp/chunk_metadata_test_report.txt):**
```
total_chunks=450
chunks_with_metadata=450
chunks_with_article_section=423
coverage_pct=94.00
chunks:
---
chunk_id=chunk_0
page=0
metadata={'document_scope': 'preamble', 'article_number': 'Preamble', ...}
text_preview=LOAN AGREEMENT dated as of...
---
chunk_id=chunk_1
page=1
metadata={'document_scope': 'article', 'article_number': 'I', ...}
text_preview=ARTICLE I Definitions and Interpretation...
```

## Key Design Decisions

1. **Stateful Tracking**: Maintains context across chunks to handle multi-page sections
2. **Pending Titles**: Handles cases where "ARTICLE I" appears on one line and the title on the next
3. **Preamble Handling**: Assigns meaningful metadata to introductory content before Article I
4. **Hierarchy Paths**: Builds readable breadcrumb trails for navigation
5. **Graceful Degradation**: If parsing fails, chunk still exists (just with incomplete metadata)

## What Wasn't Changed

- Existing chunking logic (sentence-based grouping) remains intact
- Bounding box extraction unchanged
- RAG retrieval algorithms (BM25/semantic/reranker) unchanged - only enhanced with metadata context
- All existing functionality preserved
- Backward compatible - metadata is purely additive

The metadata enhancement doesn't break any existing workflows and provides immediate value by giving the LLM better document structure awareness.
