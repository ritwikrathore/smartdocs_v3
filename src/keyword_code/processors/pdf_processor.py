"""
PDF processing functionality.
"""

import json
import fitz  # PyMuPDF
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from thefuzz import fuzz
from ..config import (
    logger, FUZZY_MATCH_THRESHOLD,
    SENTENCES_PER_CHUNK, MIN_CHUNK_CHAR_LENGTH,
    highlight_debug_logger, ENABLE_HIGHLIGHT_DEBUG_LOGGING
)
from ..utils.helpers import normalize_text, remove_markdown_formatting
from ..utils.spacy_utils import ensure_spacy_model
from ..rag.chunking import SentenceChunker, create_chunks_from_text


class DocumentStructureTracker:
    """Utility to derive hierarchical metadata (Article/Section/Subsection) for PDF chunks."""

    _ARTICLE_PATTERN = re.compile(r"^ARTICLE\s+(?P<number>[IVXLCDM]+|\d+)\b(?P<rest>.*)", re.IGNORECASE)
    _SECTION_PATTERN = re.compile(
        r"^Section\s+(?P<number>\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))?)(?P<rest>.*)",
        re.IGNORECASE,
    )
    _ANNEX_PATTERN = re.compile(r"^ANNEX\s+(?P<label>[A-Z0-9]+)\b(?P<rest>.*)", re.IGNORECASE)
    _SCHEDULE_PATTERN = re.compile(r"^SCHEDULE\s+(?P<label>[A-Z0-9]+)\b(?P<rest>.*)", re.IGNORECASE)
    _RECITAL_PATTERN = re.compile(r"^RECITALS?\s*(?:\((?P<label>[A-Z])\))?(?P<rest>.*)", re.IGNORECASE)
    _SUBSECTION_PATTERN = re.compile(r"^\((?P<label>[A-Za-z0-9ivxIVX]+)\)\s*(?P<rest>.*)")
    _PAGE_MARKER_PATTERN = re.compile(r"^\s*[-–—]?\s*(?P<marker>(?:[ivxlcdmIVXLCDM]+|\d+))\s*[-–—]?\s*$")
    _HEADING_BOUNDARY_PATTERN = re.compile(
        r"\b(?:Section\s+\d|ARTICLE\s+[IVXLCDM\d]+|ANNEX\s+[A-Z0-9]+|SCHEDULE\s+[A-Z0-9]+|RECITALS?\s*\(|RECITAL\s*\(|ANNEX\s+|SCHEDULE\s+)",
        re.IGNORECASE,
    )
    _TOC_ENTRY_PATTERN = re.compile(
        r"(?P<entry>(?:ARTICLE|Section|ANNEX|SCHEDULE)\s+[\w\.\(\)]+)(?P<dots>\.{3,}|\s{2,})(?P<page>\d+)\s*$",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.scope: str = "preamble"
        self.article_type: Optional[str] = None
        self.article_number: Optional[str] = None
        self.article_title: Optional[str] = None
        self.section_number: Optional[str] = None
        self.section_title: Optional[str] = None
        self.subsection_label: Optional[str] = None
        self.subsection_title: Optional[str] = None
        self.within_table_of_contents: bool = False
        self.pending_top_level_title: Optional[str] = None
        self.pending_section_title: bool = False
        self.toc_entries: List[Dict[str, Any]] = []

    def process_chunk(self, text: str, page_num: Optional[int]) -> Dict[str, Any]:
        """Analyze chunk text and return metadata dictionary"""
        working_text = (text or "").strip()

        if not working_text:
            return self._build_metadata()

        if self._is_page_marker(working_text):
            return self._build_metadata(scope_override="page_marker")

        if self._is_table_of_contents(working_text, page_num):
            self.within_table_of_contents = True
            self.scope = "table_of_contents"
            self.article_type = "Table of Contents"
            self.article_number = None
            self.article_title = None
            self.section_number = None
            self.section_title = None
            self.subsection_label = None
            self.subsection_title = None
            return self._build_metadata()

        if self.within_table_of_contents:
            # Extract TOC entries while in TOC region
            self._extract_toc_entries(working_text)
            # Continue processing - still in TOC
            return self._build_metadata()

        # Check if we're leaving TOC (detected non-TOC content)
        if self.within_table_of_contents and not self._looks_like_toc_content(working_text):
            self.within_table_of_contents = False
            self._reset_for_new_top_level()
            self.scope = "preamble"

        # Handle pending titles (article/annex/schedule) that may be on this chunk
        working_text = self._maybe_capture_pending_top_level_title(working_text)
        working_text = self._maybe_capture_pending_section_title(working_text)

        remaining = working_text
        while remaining:
            remaining = remaining.lstrip()
            if not remaining:
                break

            top_level_match = self._ANNEX_PATTERN.match(remaining) or self._SCHEDULE_PATTERN.match(remaining) or self._ARTICLE_PATTERN.match(remaining)
            if not top_level_match:
                top_level_match = self._RECITAL_PATTERN.match(remaining)

            if top_level_match:
                remainder = self._handle_top_level_match(top_level_match, remaining)
                if remainder is remaining:
                    break  # Safety guard to prevent infinite loop
                remaining = remainder
                continue

            section_match = self._SECTION_PATTERN.match(remaining)
            if section_match:
                remainder = self._handle_section_match(section_match, remaining)
                if remainder is remaining:
                    break
                remaining = remainder
                continue

            subsection_match = self._SUBSECTION_PATTERN.match(remaining)
            if subsection_match and self.section_number:
                remainder = self._handle_subsection_match(subsection_match, remaining)
                if remainder is remaining:
                    break
                remaining = remainder
                # Only consume the first subsection per chunk to avoid stripping substantive content
                break

            break  # No more structural markers at the start

        return self._build_metadata()

    def _handle_top_level_match(self, match: re.Match, original_text: str) -> str:
        pattern = match.re
        rest = match.group("rest") if "rest" in match.groupdict() else ""
        remainder = original_text[match.end():]

        if pattern is self._ANNEX_PATTERN:
            label = match.group("label").strip()
            title, trailing = self._split_heading_title(rest)
            self.scope = "annex"
            self.article_type = "Annex"
            self.article_number = label
            self.article_title = title or self.article_title
            self.pending_top_level_title = None if title else "Annex"
            self._reset_section()
            return trailing or remainder

        if pattern is self._SCHEDULE_PATTERN:
            label = match.group("label").strip()
            title, trailing = self._split_heading_title(rest)
            self.scope = "schedule"
            self.article_type = "Schedule"
            self.article_number = label
            self.article_title = title or self.article_title
            self.pending_top_level_title = None if title else "Schedule"
            self._reset_section()
            return trailing or remainder

        if pattern is self._ARTICLE_PATTERN:
            number = match.group("number").strip()
            title, trailing = self._split_heading_title(rest)
            self.scope = "article"
            self.article_type = "Article"
            self.article_number = number
            self.article_title = title or None
            self.pending_top_level_title = None if title else "Article"
            self._reset_section()
            return trailing or remainder

        if pattern is self._RECITAL_PATTERN:
            label = match.group("label")
            title, trailing = self._split_heading_title(rest)
            self.scope = "recital"
            self.article_type = "Recitals"
            self.article_number = "Recitals"
            if label:
                self.section_number = f"Recital ({label})"
                self.section_title = title or self.section_title
                self.pending_section_title = False if title else True
            else:
                # Bare RECITAL header – treat as introduction to recitals
                self.section_number = "Recitals"
                self.section_title = title or self.section_title or "Recitals"
            self.subsection_label = None
            self.subsection_title = None
            return trailing or remainder

        return remainder

    def _handle_section_match(self, match: re.Match, original_text: str) -> str:
        rest = match.group("rest") or ""
        remainder = original_text[match.end():]

        number = match.group("number").strip()
        cleaned_number = number
        if not cleaned_number.lower().startswith("section"):
            cleaned_number = f"Section {cleaned_number}"

        self.section_number = cleaned_number
        title, trailing = self._split_heading_title(rest)
        if title:
            self.section_title = title
            self.pending_section_title = False
        else:
            self.section_title = None
            self.pending_section_title = True

        self.subsection_label = None
        self.subsection_title = None

        return trailing or remainder

    def _handle_subsection_match(self, match: re.Match, original_text: str) -> str:
        label = match.group("label").strip()
        rest = match.group("rest") or ""
        remainder = original_text[match.end():]

        clean_label = label.strip().strip("().")
        title, trailing = self._split_heading_title(rest)
        self.subsection_label = clean_label
        self.subsection_title = title or self.subsection_title

        return trailing or remainder

    def _maybe_capture_pending_top_level_title(self, text: str) -> str:
        if not self.pending_top_level_title:
            return text

        title, remainder = self._split_heading_title(text)
        if title:
            self.article_title = title
            self.pending_top_level_title = None
            return remainder
        return text

    def _maybe_capture_pending_section_title(self, text: str) -> str:
        if not self.pending_section_title:
            return text

        title, remainder = self._split_heading_title(text)
        if title:
            self.section_title = title
            self.pending_section_title = False
            return remainder
        return text

    def _split_heading_title(self, text: str) -> Tuple[Optional[str], str]:
        if not text:
            return None, ""

        cleaned = text.lstrip(" .:-–—")
        if not cleaned:
            return None, ""

        boundary_match = self._HEADING_BOUNDARY_PATTERN.search(cleaned)
        if boundary_match:
            title = cleaned[: boundary_match.start()].strip(" .:-–—")
            remainder = cleaned[boundary_match.start():]
            return (title or None), remainder

        # If no explicit boundary, attempt to cut at first sentence break to avoid capturing entire chunk
        sentence_break = re.search(r"(?<=[.;:])\s{2,}|(?<=[.;:])\s(?=[A-Z])", cleaned)
        if sentence_break:
            title = cleaned[: sentence_break.start()].strip(" .:-–—")
            remainder = cleaned[sentence_break.start():]
            return (title or None), remainder

        # Fallback: if cleaned text is short, treat it as title; otherwise keep for body content
        if len(cleaned) <= 80:
            return cleaned.strip(" .:-–—") or None, ""

        return None, text

    def _reset_for_new_top_level(self) -> None:
        self.article_type = None
        self.article_number = None
        self.article_title = None
        self._reset_section()
        self.pending_top_level_title = None

    def _reset_section(self) -> None:
        self.section_number = None
        self.section_title = None
        self.subsection_label = None
        self.subsection_title = None
        self.pending_section_title = False

    def _is_table_of_contents(self, text: str, page_num: Optional[int]) -> bool:
        upper_text = text.upper()
        if "TABLE OF CONTENTS" in upper_text:
            return True
        if page_num is not None and page_num <= 5:
            if re.search(r"\.{5,}", text) and any(keyword in upper_text for keyword in ("SECTION", "ARTICLE", "ANNEX", "SCHEDULE")):
                return True
        return False

    def _is_page_marker(self, text: str) -> bool:
        return bool(self._PAGE_MARKER_PATTERN.match(text))

    def _looks_like_toc_content(self, text: str) -> bool:
        """Check if text appears to be TOC content (dots/page numbers/section refs)."""
        upper_text = text.upper()
        # Has dots connecting to page numbers
        if re.search(r"\.{5,}\s*\d+", text):
            return True
        # Has section/article references with page numbers
        if re.search(r"(?:SECTION|ARTICLE|ANNEX|SCHEDULE)\s+[\w\.]+.*\d+\s*$", text, re.IGNORECASE):
            return True
        # Mostly whitespace with scattered numbers (page refs)
        if re.search(r"^\s*\d+\s*$", text):
            return True
        return False

    def _extract_toc_entries(self, text: str) -> None:
        """Extract structured entries from table of contents text."""
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match TOC entry pattern
            match = self._TOC_ENTRY_PATTERN.search(line)
            if match:
                entry = match.group("entry").strip()
                page = match.group("page").strip()
                self.toc_entries.append({
                    "entry": entry,
                    "page_number": int(page),
                    "raw_text": line,
                })
            # Also capture entries without dots (sometimes formatted differently)
            elif re.search(r"(?:ARTICLE|Section|ANNEX|SCHEDULE)\s+[\w\.\(\)]+", line, re.IGNORECASE):
                # Extract page number from end if present
                page_match = re.search(r"(\d+)\s*$", line)
                if page_match:
                    entry_text = line[:page_match.start()].strip()
                    self.toc_entries.append({
                        "entry": entry_text,
                        "page_number": int(page_match.group(1)),
                        "raw_text": line,
                    })

    def _build_metadata(self, scope_override: Optional[str] = None) -> Dict[str, Any]:
        scope = scope_override or self.scope or "unknown"

        article_type = self.article_type
        article_number = self.article_number
        article_title = self.article_title

        if scope not in {"table_of_contents", "page_marker"}:
            if not article_type:
                article_type = "Preamble"
            if article_type == "Preamble" and not article_title:
                article_title = "Introductory Statements"
            if article_type == "Preamble" and not article_number:
                article_number = "Preamble"

        metadata: Dict[str, Any] = {
            "document_scope": scope,
            "article_type": article_type,
            "article_number": article_number,
            "article_title": article_title,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "subsection_label": self.subsection_label,
            "subsection_title": self.subsection_title,
        }

        # Add TOC entries if this is a table of contents chunk
        if scope == "table_of_contents" and self.toc_entries:
            metadata["toc_entries"] = self.toc_entries.copy()

        path: List[str] = []
        if article_type:
            if article_type == "Article" and article_number:
                article_entry = f"Article {article_number}"
            elif article_type in {"Annex", "Schedule"} and article_number:
                article_entry = f"{article_type} {article_number}"
            else:
                article_entry = article_type
            if article_title:
                article_entry = f"{article_entry} - {article_title}"
            path.append(article_entry)

        section_number = self.section_number
        section_title = self.section_title
        if not section_number and article_type == "Preamble" and scope == "preamble":
            section_number = "Preamble"
            if not section_title:
                section_title = "Introductory Statements"

        if section_number:
            section_entry = section_number
            if section_title:
                section_entry = f"{section_entry} - {section_title}"
            path.append(section_entry)

        if self.subsection_label:
            subsection_entry = f"Subsection {self.subsection_label}"
            if self.subsection_title:
                subsection_entry = f"{subsection_entry} - {self.subsection_title}"
            path.append(subsection_entry)

        metadata["hierarchy_path"] = path
        return metadata


class PDFProcessor:
    """Handles PDF processing, chunking, verification, and annotation."""

    _HIGHLIGHT_METHOD_PRIORITY: Dict[str, int] = {
        "exact_cleaned_search": 0,
        "exact_original_search": 0,
        "exact_normalized_search": 0,
        "exact_quote_stripped_search": 0,
        "exact_with_section_prefix": 0,
        "exact_normalized_with_section_prefix": 0,
        "exact_original_search_expanded": 0,
        "exact_normalized_search_expanded": 0,
        "exact_quote_stripped_search_expanded": 0,
        "exact_with_section_prefix_expanded": 0,
        "exact_normalized_with_section_prefix_expanded": 0,
        "fuzzy_span_match": 1,
        "cross_page_fuzzy_match_part1": 2,
        "cross_page_fuzzy_match_part2": 2,
        "special_case_quotes_handling": 2,
        "fuzzy_chunk_fallback_individual": 3,
        "fuzzy_chunk_fallback": 4,
    }

    _SEARCH_CHAR_TRANSLATION = {
        ord("\u2010"): "-",
        ord("\u2011"): "-",
        ord("\u2012"): "-",
        ord("\u2013"): "-",
        ord("\u2014"): "-",
        ord("\u2015"): "-",
        ord("\u2212"): "-",
        ord("\u00ad"): "",
        ord("\ufb01"): "fi",
        ord("\ufb02"): "fl",
    }

    _QUOTE_TRANSLATION = {
        ord("\u201c"): '"',
        ord("\u201d"): '"',
        ord("\u201e"): '"',
        ord("\u201f"): '"',
        ord("\u2019"): "'",
        ord("\u2018"): "'",
        ord("\u2032"): "'",
        ord("\u2035"): "'",
        ord("\u0060"): "'",
    }

    def __init__(self, pdf_bytes: bytes):
        if not isinstance(pdf_bytes, bytes):
            raise ValueError("pdf_bytes must be of type bytes")
        self.pdf_bytes = pdf_bytes
        self._chunks: List[Dict[str, Any]] = []
        self._full_text: Optional[str] = None
        self._processed = False  # Flag to track if extraction ran

        # Use the utility function to ensure the spaCy model is available locally
        self._nlp = ensure_spacy_model("en_core_web_sm")

        if self._nlp is None:
            logger.error(
                "Failed to load spaCy model 'en_core_web_sm'. "
                "Text extraction and chunking will not work properly."
            )
        else:
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")

        logger.info(f"PDFProcessor initialized with {len(pdf_bytes)} bytes.")

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        if not self._processed:
            self.extract_structured_text_and_chunks()  # Lazy extraction
        return self._chunks

    # Keep full_text property in case it's needed elsewhere
    @property
    def full_text(self) -> str:
        if not self._processed:
            self.extract_structured_text_and_chunks()  # Lazy extraction
        return self._full_text if self._full_text is not None else ""

    def extract_structured_text_and_chunks(self) -> Tuple[List[Dict[str, Any]], str]:
        """Extracts text using PyMuPDF blocks, segments into sentences with spaCy, and groups them into chunks."""
        if self._processed:  # Already processed
            return self._chunks, self._full_text if self._full_text is not None else ""

        self._chunks = []
        all_text_parts = [] # Used to build self._full_text
        current_chunk_id_counter = 0
        doc = None

        # Create a sentence chunker with configuration from config.py
        chunker = SentenceChunker(
            sentences_per_chunk=SENTENCES_PER_CHUNK,
            min_chunk_char_length=MIN_CHUNK_CHAR_LENGTH,
            nlp=self._nlp
        )

        if not self._nlp:
            logger.error("spaCy model not loaded. Cannot perform sentence-based chunking.")
            self._full_text = ""
            self._chunks = []
            self._processed = True
            return self._chunks, self._full_text

        try:
            logger.info("Starting sentence-based text extraction and chunking...")
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            logger.info(f"PDF opened with {doc.page_count} pages.")

            for page_num, page in enumerate(doc):
                page_text_content = ""
                # Stores tuples: (char_start_idx_in_page_text, char_end_idx_in_page_text, fitz.Rect_of_block)
                char_pos_to_bbox_map: List[Tuple[int, int, fitz.Rect]] = []
                current_char_offset = 0

                blocks = page.get_text("blocks", sort=True) # sort=True sorts blocks by y-coordinate, then x

                for b_idx, b in enumerate(blocks):
                    # x0, y0, x1, y1, text, block_no, block_type = b # PyMuPDF block structure
                    block_text_original = b[4] # The text content of the block

                    # Normalize block text: replace multiple spaces/newlines with a single space, then strip.
                    # This helps create a cleaner text string for spaCy and for char mapping.
                    # Fixed: Use \s instead of \\s for proper regex matching
                    block_text_cleaned = re.sub(r'\s+', ' ', block_text_original).strip()

                    if not block_text_cleaned:
                        continue

                    # Append block text to the running page_text_content
                    if page_text_content:  # Add a space separator if not the first piece of text
                        page_text_content += " "
                        current_char_offset += 1 # Account for the space

                    start_offset_for_block = current_char_offset
                    page_text_content += block_text_cleaned
                    current_char_offset += len(block_text_cleaned)
                    end_offset_for_block = current_char_offset # end_offset is exclusive

                    char_pos_to_bbox_map.append(
                        (start_offset_for_block, end_offset_for_block, fitz.Rect(b[0], b[1], b[2], b[3]))
                    )

                if not page_text_content.strip(): # If page is blank or only whitespace
                    continue

                # Process the concatenated text of the page with spaCy
                spacy_page_doc = self._nlp(page_text_content)
                page_sentences = list(spacy_page_doc.sents) # list of spaCy Span objects (sentences)

                # Create chunks using our chunker
                page_chunks = chunker.create_chunks(
                    text=page_text_content,
                    page_num=page_num,
                    start_chunk_id=current_chunk_id_counter
                )

                # Process each chunk to add bounding box information
                for chunk in page_chunks:
                    chunk_text = chunk["text"]

                    # Find the sentences that make up this chunk
                    # Improved: Use more robust matching with normalized text comparison
                    chunk_sentences = []
                    chunk_text_normalized = chunk_text.strip()

                    for i in range(0, len(page_sentences)):
                        # Try exact match first
                        if chunk_text_normalized.startswith(page_sentences[i].text.strip()):
                            # Found the first sentence of the chunk
                            end_idx = min(i + SENTENCES_PER_CHUNK, len(page_sentences))
                            chunk_sentences = page_sentences[i:end_idx]
                            break

                    # Fallback: If no match found, try fuzzy matching on first 50 chars
                    if not chunk_sentences and page_sentences:
                        from fuzzywuzzy import fuzz
                        chunk_start = chunk_text_normalized[:50]
                        best_match_idx = -1
                        best_score = 0
                        for i in range(0, len(page_sentences)):
                            sent_start = page_sentences[i].text.strip()[:50]
                            score = fuzz.ratio(chunk_start, sent_start)
                            if score > best_score and score > 80:  # 80% similarity threshold
                                best_score = score
                                best_match_idx = i

                        if best_match_idx >= 0:
                            end_idx = min(best_match_idx + SENTENCES_PER_CHUNK, len(page_sentences))
                            chunk_sentences = page_sentences[best_match_idx:end_idx]
                            logger.debug(f"Used fuzzy matching (score: {best_score}) to find sentences for chunk {chunk.get('chunk_id', 'unknown')}")

                    # Determine bounding boxes for this sentence-based chunk
                    chunk_associated_bboxes: List[fitz.Rect] = []
                    if chunk_sentences:
                        # Get character start of the first sentence and character end of the last sentence in this group
                        chunk_start_char_offset = chunk_sentences[0].start_char
                        chunk_end_char_offset = chunk_sentences[-1].end_char

                        # Find all original block bboxes that overlap with this chunk's character span
                        overlapping_blocks = []
                        for block_map_start, block_map_end, block_bbox in char_pos_to_bbox_map:
                            # Check for overlap between [block_map_start, block_map_end)
                            # and [chunk_start_char_offset, chunk_end_char_offset)
                            if max(block_map_start, chunk_start_char_offset) < min(block_map_end, chunk_end_char_offset):
                                # Calculate overlap percentage to prioritize blocks with significant overlap
                                overlap_start = max(block_map_start, chunk_start_char_offset)
                                overlap_end = min(block_map_end, chunk_end_char_offset)
                                overlap_length = overlap_end - overlap_start
                                block_length = block_map_end - block_map_start

                                # Improved: Lower threshold from 10% to 5% to capture more relevant blocks
                                # This helps with chunks that span multiple small blocks
                                if overlap_length > 0.05 * block_length:
                                    overlapping_blocks.append((block_bbox, overlap_length))

                        # Improved: Increase from top 3 to top 5 blocks to capture more complete text areas
                        # This helps with multi-line chunks and complex layouts
                        if overlapping_blocks:
                            overlapping_blocks.sort(key=lambda x: x[1], reverse=True)
                            max_blocks = min(5, len(overlapping_blocks))  # Take up to 5 blocks
                            for block_bbox, _ in overlapping_blocks[:max_blocks]:
                                chunk_associated_bboxes.append(block_bbox)

                        if not chunk_associated_bboxes and char_pos_to_bbox_map:
                            # Fallback: if no direct overlap found (e.g. due to spacing differences),
                            # try to associate with the block containing the start of the first sentence.
                            # This is a heuristic.
                            first_sent_start_char = chunk_sentences[0].start_char
                            for block_map_start, block_map_end, block_bbox in char_pos_to_bbox_map:
                                if block_map_start <= first_sent_start_char < block_map_end:
                                    chunk_associated_bboxes.append(block_bbox)
                                    logger.debug(f"Used fallback bbox matching for chunk {chunk.get('chunk_id', 'unknown')}")
                                    break # Found one, that's enough for this fallback

                    # Add bounding boxes to the chunk
                    chunk["bboxes"] = chunk_associated_bboxes

                    # Log warning if no bboxes found for this chunk
                    if not chunk_associated_bboxes:
                        logger.warning(f"No bounding boxes found for chunk {chunk.get('chunk_id', 'unknown')} on page {page_num}. Highlighting may fail for this chunk.")

                    # Add to our chunks list
                    self._chunks.append(chunk)
                    all_text_parts.append(chunk_text)  # For building self._full_text

                # Update the chunk counter for the next page
                if page_chunks:
                    current_chunk_id_counter += len(page_chunks)

            self._assign_chunk_metadata()
            self._full_text = "\n\n".join(all_text_parts) # Join chunks for full text
            self._processed = True
            logger.info(
                f"Sentence-based extraction complete. Generated {len(self._chunks)} chunks. "
                f"Total text length: {len(self._full_text or '')} chars."
            )

        except Exception as e:
            logger.error(f"Failed to extract sentence-based chunks: {str(e)}", exc_info=True)
            self._full_text = ""    # Reset on error
            self._chunks = []       # Reset on error
            self._processed = True  # Mark as processed even on failure to prevent re-runs

        finally:
            if doc:
                doc.close()
        return self._chunks, self._full_text if self._full_text is not None else ""

    def _assign_chunk_metadata(self) -> None:
        """Attach hierarchical metadata to each extracted chunk."""
        if not self._chunks:
            return

        tracker = DocumentStructureTracker()
        for chunk in self._chunks:
            try:
                chunk_text = chunk.get("text", "")
                page_num = chunk.get("page_num")
                chunk["metadata"] = tracker.process_chunk(chunk_text, page_num)
            except Exception as meta_err:
                logger.warning(
                    "Failed to derive metadata for chunk %s on page %s: %s",
                    chunk.get("chunk_id"),
                    chunk.get("page_num"),
                    meta_err,
                )

    def verify_and_locate_phrases(
        self, ai_analysis_json_str: str  # Expects the *aggregated* JSON string
    ) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
        """Verifies AI phrases from the aggregated analysis against chunks and locates them."""
        verification_results = {}
        phrase_locations = {}
        method_stats: Dict[str, int] = {}

        chunks_data = self.chunks
        if not chunks_data:
            logger.warning("No chunks available for verification.")
            return {}, {}

        try:
            # Parse the *aggregated* AI analysis
            ai_analysis = json.loads(ai_analysis_json_str)

            # Check if the entire analysis was just an error placeholder
            if not ai_analysis.get("analysis_sections") or \
               all(k.startswith("error_") for k in ai_analysis.get("analysis_sections", {})):
                logger.warning("AI analysis contains only errors or is empty, skipping phrase verification.")
                return {}, {}

            phrases_to_verify = set()
            # Extract all supporting phrases from *all* sections in the aggregated analysis

            # Log the structure of the AI analysis for debugging
            logger.info(f"AI analysis structure: {list(ai_analysis.keys())}")
            if "analysis_sections" in ai_analysis:
                logger.info(f"Analysis sections: {list(ai_analysis.get('analysis_sections', {}).keys())}")

            # Handle both old and new JSON structures
            for section_key, section_data in ai_analysis.get("analysis_sections", {}).items():
                # Skip sections indicating skipped RAG or errors generated during analysis
                if section_key.startswith("info_skipped_") or section_key.startswith("error_"):
                    continue

                logger.info(f"Processing section: {section_key}, type: {type(section_data)}")
                if isinstance(section_data, dict):
                    # Log the keys in this section for debugging
                    logger.info(f"Section keys: {list(section_data.keys())}")

                    # Check for both old and new field names for supporting phrases
                    phrases = section_data.get("Supporting_Phrases", section_data.get("supporting_quotes", []))

                    # Handle case where phrases might be a string instead of a list
                    if isinstance(phrases, str):
                        logger.warning(f"Found phrases as string instead of list: {phrases}")
                        phrases = [phrases]

                    # Log the phrases found
                    logger.info(f"Found phrases: {phrases}")

                    if isinstance(phrases, list):
                        for phrase in phrases:
                            p_text = ""
                            if isinstance(phrase, str):
                                p_text = phrase
                            elif phrase is not None:
                                # Convert non-string phrases to string
                                try:
                                    p_text = str(phrase)
                                    logger.warning(f"Converted non-string phrase to string: {p_text}")
                                except Exception as e:
                                    logger.error(f"Failed to convert phrase to string: {e}")
                                    continue

                            p_text = p_text.strip()
                            # Exclude the "No relevant phrase found." placeholder
                            if p_text and p_text != "No relevant phrase found.":
                                phrases_to_verify.add(p_text)

            if not phrases_to_verify:
                logger.info("No supporting phrases found in aggregated AI analysis to verify.")
                return {}, {}

            logger.info(
                f"Starting verification for {len(phrases_to_verify)} unique phrases "
                f"(from aggregated analysis) against {len(chunks_data)} original chunks."
            )

            normalized_chunks = [
                (chunk, normalize_text(chunk["text"])) for chunk in chunks_data if chunk.get("text")
            ]

            doc = None
            try:
                doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")

                for original_phrase in phrases_to_verify:
                    verification_results[original_phrase] = False
                    phrase_locations[original_phrase] = []
                    normalized_phrase = normalize_text(remove_markdown_formatting(original_phrase))
                    if not normalized_phrase: continue

                    found_match_for_phrase = False
                    phrase_matches: List[Dict[str, Any]] = []
                    phrase_best_priority: Optional[int] = None

                    # Initialize highlight debug tracking for this phrase
                    if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                        highlight_debug_logger.info("="*100)
                        highlight_debug_logger.info(f"PHRASE: {original_phrase}")
                        highlight_debug_logger.info(f"  Pre-normalization: '{original_phrase}'")
                        highlight_debug_logger.info(f"  Post-normalization (for fuzzy): '{normalized_phrase}'")
                        highlight_debug_logger.info("-"*100)

                    def add_match(method: str, match_data: Dict[str, Any]) -> None:
                        """Record a match while retaining only the highest-priority methods per phrase."""
                        nonlocal phrase_matches, phrase_best_priority
                        priority = self._get_method_priority(method)

                        if phrase_best_priority is None or priority < phrase_best_priority:
                            if phrase_matches and priority < (phrase_best_priority or 99):
                                logger.debug(
                                    "Dropping %d lower-priority highlight(s) for phrase '%s' in favor of method '%s'.",
                                    len(phrase_matches),
                                    original_phrase[:60],
                                    method,
                                )
                            phrase_matches = [{**match_data, "method": method}]
                            phrase_best_priority = priority
                            return

                        if priority == phrase_best_priority:
                            phrase_matches.append({**match_data, "method": method})
                            return

                        logger.debug(
                            "Skipping lower-priority highlight method '%s' for phrase '%s' because better match already exists.",
                            method,
                            original_phrase[:60],
                        )

                    # Log the normalized phrase for debugging
                    logger.debug(f"Normalized phrase for verification: '{normalized_phrase}'")

                    # Verify against ALL original chunks
                    for chunk, norm_chunk_text in normalized_chunks:
                        if not norm_chunk_text: continue

                        # Log the normalized chunk text for debugging
                        logger.debug(f"Comparing with normalized chunk text: '{norm_chunk_text[:100]}...'")

                        # Try multiple fuzzy matching methods for better accuracy
                        partial_score = fuzz.partial_ratio(normalized_phrase, norm_chunk_text)
                        token_set_score = fuzz.token_set_ratio(normalized_phrase, norm_chunk_text)

                        # Use the better of the two scores
                        score = max(partial_score, token_set_score)

                        if score >= FUZZY_MATCH_THRESHOLD:
                            if not found_match_for_phrase:
                                logger.info(f"Verified (Score: {score}) '{original_phrase[:60]}...' potentially in chunk {chunk['chunk_id']}")
                            found_match_for_phrase = True
                            verification_results[original_phrase] = True
                            # best_score_for_phrase = max(best_score_for_phrase, score) # This variable was not used

                            # --- Precise Location Search ---
                            page_num = chunk["page_num"]
                            if 0 <= page_num < doc.page_count:
                                page = doc[page_num]
                                clip_rect = fitz.Rect()
                                for bbox in chunk.get('bboxes', []):
                                    try:
                                        if isinstance(bbox, fitz.Rect): clip_rect.include_rect(bbox)
                                        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4: clip_rect.include_rect(fitz.Rect(bbox))
                                    except Exception as bbox_err: logger.warning(f"Skipping invalid bbox {bbox} in chunk {chunk['chunk_id']}: {bbox_err}")

                                if not clip_rect.is_empty:
                                    try:
                                        exact_hits = self._find_exact_matches(
                                            page=page,
                                            clip_rect=clip_rect,
                                            original_phrase=original_phrase,
                                            chunk_text=chunk.get("text"),
                                        )

                                        if exact_hits:
                                            logger.debug(
                                                "Found %d exact match instance(s) in chunk %s for phrase '%s'",
                                                len(exact_hits),
                                                chunk.get("chunk_id"),
                                                original_phrase[:60],
                                            )
                                            if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                                                highlight_debug_logger.info(f"  ✓ EXACT MATCH in chunk {chunk.get('chunk_id')}")
                                                for rect, method in exact_hits:
                                                    highlight_debug_logger.info(f"    Method: {method}")
                                            for rect, method in exact_hits:
                                                if isinstance(rect, fitz.Rect) and not rect.is_empty:
                                                    add_match(
                                                        method,
                                                        {
                                                            "page_num": page_num,
                                                            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                                                            "chunk_id": chunk.get("chunk_id"),
                                                            "match_score": score,
                                                        },
                                                    )
                                        else:
                                            # Try fuzzy span matching within the chunk before falling back to full chunk
                                            logger.debug(
                                                "Exact search failed for phrase '%s' in verified chunk %s (score: %s). Trying fuzzy span matching.",
                                                original_phrase[:60],
                                                chunk.get("chunk_id"),
                                                score,
                                            )
                                            
                                            fuzzy_span_match = self._find_fuzzy_span_in_chunk(
                                                original_phrase=original_phrase,
                                                chunk=chunk,
                                                page=page,
                                                doc=doc,
                                                fuzzy_score=score,
                                            )
                                            
                                            if fuzzy_span_match:
                                                if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                                                    highlight_debug_logger.info(f"  ✓ FUZZY SPAN MATCH in chunk {chunk.get('chunk_id')} (score: {score})")
                                                    highlight_debug_logger.info(f"    Matched span: '{fuzzy_span_match.get('matched_text', '')[:100]}...'")
                                                
                                                add_match(
                                                    "fuzzy_span_match",
                                                    {
                                                        "page_num": page_num,
                                                        "rect": fuzzy_span_match["rect"],
                                                        "chunk_id": chunk.get("chunk_id"),
                                                        "match_score": score,
                                                    },
                                                )
                                            else:
                                                # Final fallback to chunk bounding box
                                                if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                                                    highlight_debug_logger.info(f"  ✗ FALLBACK in chunk {chunk.get('chunk_id')} (fuzzy score: {score})")
                                                    chunk_text_sample = chunk.get("text", "")[:200].replace("\n", " ")
                                                    highlight_debug_logger.info(f"    Chunk text sample: '{chunk_text_sample}...'")

                                                individual_bboxes = chunk.get('bboxes', [])
                                                if individual_bboxes and len(individual_bboxes) <= 3:
                                                    for bbox in individual_bboxes:
                                                        if isinstance(bbox, fitz.Rect) and not bbox.is_empty:
                                                            add_match(
                                                                "fuzzy_chunk_fallback_individual",
                                                                {
                                                                    "page_num": page_num,
                                                                    "rect": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                                                                    "chunk_id": chunk.get("chunk_id"),
                                                                    "match_score": score,
                                                                },
                                                            )
                                                elif not clip_rect.is_empty:
                                                    add_match(
                                                        "fuzzy_chunk_fallback",
                                                        {
                                                            "page_num": page_num,
                                                            "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                                            "chunk_id": chunk.get("chunk_id"),
                                                            "match_score": score,
                                                        },
                                                    )
                                    except Exception as search_err: logger.error(f"Error during search_for/fallback in chunk {chunk['chunk_id']}: {search_err}")
                            # else: logger.warning(f"Invalid page number {page_num} for chunk {chunk['chunk_id']}")

                    # --- Second Pass: If not found in a single chunk, try cross-page concatenation ---
                    if not found_match_for_phrase:
                        for i in range(len(chunks_data) - 1):
                            chunk_A = chunks_data[i]
                            chunk_B = chunks_data[i+1]

                            # Condition: chunk_A is on page N, chunk_B is on page N+1
                            # (Assumes chunks_data is sorted by page, then by position on page)
                            if chunk_A.get("page_num") is not None and \
                               chunk_B.get("page_num") is not None and \
                               chunk_A.get("page_num") == chunk_B.get("page_num") - 1:

                                text_A = chunk_A.get("text", "")
                                text_B = chunk_B.get("text", "")

                                if not text_A.strip() or not text_B.strip():  # Ensure there's text to combine
                                    continue

                                # Combine text (simple concatenation with a space)
                                combined_text = text_A + " " + text_B
                                normalized_combined_text = normalize_text(combined_text)

                                # Log the combined text for debugging
                                logger.debug(f"Cross-page combined text (normalized): '{normalized_combined_text[:100]}...'")

                                # Try multiple fuzzy matching methods for better accuracy
                                partial_score = fuzz.partial_ratio(normalized_phrase, normalized_combined_text)
                                token_set_score = fuzz.token_set_ratio(normalized_phrase, normalized_combined_text)

                                # Use the better of the two scores
                                score = max(partial_score, token_set_score)

                                if score >= FUZZY_MATCH_THRESHOLD:
                                    logger.info(f"Verified (Score: {score}, Cross-Page) '{original_phrase[:60]}...' by combining chunk {chunk_A['chunk_id']} (pg {chunk_A['page_num']}) and chunk {chunk_B['chunk_id']} (pg {chunk_B['page_num']})")
                                    verification_results[original_phrase] = True
                                    found_match_for_phrase = True  # Mark as found to prevent "NOT Verified" log and stop further cross-page checks for this phrase

                                    # Add locations for both involved chunks
                                    # Location for chunk_A part (page N)
                                    page_A_num = chunk_A["page_num"]
                                    if 0 <= page_A_num < doc.page_count:
                                        clip_rect_A = fitz.Rect()
                                        for bbox in chunk_A.get('bboxes', []):
                                            try:
                                                if isinstance(bbox, fitz.Rect): clip_rect_A.include_rect(bbox)
                                                elif isinstance(bbox, (list, tuple)) and len(bbox) == 4: clip_rect_A.include_rect(fitz.Rect(bbox))
                                            except Exception as bbox_err: logger.warning(f"Skipping invalid bbox {bbox} in chunk {chunk_A['chunk_id']}: {bbox_err}")
                                        if not clip_rect_A.is_empty:
                                            add_match(
                                                "cross_page_fuzzy_match_part1",
                                                {
                                                    "page_num": page_A_num,
                                                    "rect": [clip_rect_A.x0, clip_rect_A.y0, clip_rect_A.x1, clip_rect_A.y1],
                                                    "chunk_id": chunk_A.get("chunk_id"),
                                                    "match_score": score,
                                                },
                                            )

                                    # Location for chunk_B part (page N+1)
                                    page_B_num = chunk_B["page_num"]
                                    if 0 <= page_B_num < doc.page_count:
                                        clip_rect_B = fitz.Rect()
                                        for bbox in chunk_B.get('bboxes', []):
                                            try:
                                                if isinstance(bbox, fitz.Rect): clip_rect_B.include_rect(bbox)
                                                elif isinstance(bbox, (list, tuple)) and len(bbox) == 4: clip_rect_B.include_rect(fitz.Rect(bbox))
                                            except Exception as bbox_err: logger.warning(f"Skipping invalid bbox {bbox} in chunk {chunk_B['chunk_id']}: {bbox_err}")
                                        if not clip_rect_B.is_empty:
                                            add_match(
                                                "cross_page_fuzzy_match_part2",
                                                {
                                                    "page_num": page_B_num,
                                                    "rect": [clip_rect_B.x0, clip_rect_B.y0, clip_rect_B.x1, clip_rect_B.y1],
                                                    "chunk_id": chunk_B.get("chunk_id"),
                                                    "match_score": score,
                                                },
                                            )
                                    break  # Found a cross-page match for this phrase, move to the next phrase

                    # Special case handling for phrases with quotation marks
                    if not found_match_for_phrase and '"' in original_phrase:
                        logger.info(f"Attempting special case handling for phrase with quotes: '{original_phrase[:60]}...'")

                        # Create an alternative version with quotes removed completely (not just replaced with spaces)
                        alt_phrase = re.sub(r'[\'"""'']', '', original_phrase)
                        normalized_alt_phrase = normalize_text(alt_phrase)

                        # Try again with all chunks
                        for chunk, norm_chunk_text in normalized_chunks:
                            if not norm_chunk_text: continue

                            # Try multiple fuzzy matching methods for better accuracy
                            partial_score = fuzz.partial_ratio(normalized_alt_phrase, norm_chunk_text)
                            token_set_score = fuzz.token_set_ratio(normalized_alt_phrase, norm_chunk_text)

                            # Use the better of the two scores with a slightly lower threshold
                            score = max(partial_score, token_set_score)
                            special_case_threshold = FUZZY_MATCH_THRESHOLD - 5  # More lenient for special cases

                            if score >= special_case_threshold:
                                logger.info(f"Verified via special case handling (Score: {score}) '{original_phrase[:60]}...' in chunk {chunk['chunk_id']}")
                                found_match_for_phrase = True
                                verification_results[original_phrase] = True

                                # Add location information
                                page_num = chunk["page_num"]
                                if 0 <= page_num < doc.page_count:
                                    clip_rect = fitz.Rect()
                                    for bbox in chunk.get('bboxes', []):
                                        try:
                                            if isinstance(bbox, fitz.Rect): clip_rect.include_rect(bbox)
                                            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4: clip_rect.include_rect(fitz.Rect(bbox))
                                        except Exception as bbox_err: logger.warning(f"Skipping invalid bbox {bbox} in chunk {chunk['chunk_id']}: {bbox_err}")

                                    if not clip_rect.is_empty:
                                        add_match(
                                            "special_case_quotes_handling",
                                            {
                                                "page_num": page_num,
                                                "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                                "chunk_id": chunk.get("chunk_id"),
                                                "match_score": score,
                                            },
                                        )
                                break  # Found a match, no need to check other chunks

                    if not found_match_for_phrase:
                        logger.warning(f"NOT Verified: '{original_phrase[:60]}...' did not meet fuzzy threshold ({FUZZY_MATCH_THRESHOLD}) in any chunk or cross-page combination.")

                    phrase_locations[original_phrase] = phrase_matches
                    if phrase_matches:
                        for match in phrase_matches:
                            method = match.get("method")
                            method_stats[method] = method_stats.get(method, 0) + 1
            finally:
                if doc: doc.close()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse aggregated AI analysis JSON for verification: {e}")
            logger.debug(f"Problematic JSON string: {ai_analysis_json_str[:500]}...")  # Log start of bad JSON
        except Exception as e:
            logger.error(f"Error during phrase verification and location: {str(e)}", exc_info=True)

        total_highlights = sum(method_stats.values())
        if total_highlights:
            breakdown = []
            for method, count in sorted(method_stats.items(), key=lambda item: (self._get_method_priority(item[0]), item[0])):
                percentage = (count / total_highlights) * 100
                breakdown.append(f"{method}: {count} ({percentage:.1f}%)")
            logger.info(
                "Highlight method breakdown (total %d): %s",
                total_highlights,
                "; ".join(breakdown),
            )
        else:
            logger.info("Highlight method breakdown: no highlights recorded.")

        return verification_results, phrase_locations

    def add_annotations(
        self, phrase_locations: Dict[str, List[Dict[str, Any]]]
    ) -> bytes:
        """Adds highlights to the PDF based on found phrase locations (from aggregated results)."""
        if not phrase_locations:
            logger.warning("No phrase locations provided for annotation. Returning original PDF bytes.")
            return self.pdf_bytes

        doc = None
        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            annotated_count = 0
            highlight_color = [1, 0.9, 0.3]  # Yellow
            fallback_color = [0.5, 0.7, 1.0]  # Light Blue for fallback

            # Flatten all locations from the dict for easier processing
            all_locs = []
            for phrase, locations in phrase_locations.items():
                for loc in locations:
                    # Add the phrase back into the location dict for context in annotation info
                    loc['phrase_text'] = phrase
                    all_locs.append(loc)

            # Optional: Sort annotations to potentially process page by page
            # all_locs.sort(key=lambda x: (x.get('page_num', -1), x.get('rect', [0,0,0,0])[1]))

            for loc in all_locs:
                try:
                    page_num = loc.get("page_num")
                    rect_coords = loc.get("rect")
                    method = loc.get("method", "unknown")
                    phrase = loc.get("phrase_text", "Unknown Phrase")

                    if page_num is None or rect_coords is None:
                        logger.warning(f"Skipping annotation due to missing page_num/rect for phrase '{phrase[:50]}...': {loc}")
                        continue

                    if 0 <= page_num < doc.page_count:
                        page = doc[page_num]
                        rect = fitz.Rect(rect_coords)
                        if not rect.is_empty:
                            color = fallback_color if "fallback" in method else highlight_color
                            highlight = page.add_highlight_annot(rect)
                            highlight.set_colors(stroke=color)
                            highlight.set_info(
                                content=(f"Verified ({method}, Score: {loc.get('match_score', 'N/A'):.0f}): {phrase[:100]}...")
                            )
                            highlight.update(opacity=0.4)
                            annotated_count += 1
                        # else: logger.debug(f"Skipping annotation for empty rect: {rect}")
                    # else: logger.warning(f"Skipping annotation due to invalid page num {page_num} from location data.")
                except Exception as annot_err:
                    logger.error(f"Error adding annotation for phrase '{phrase[:50]}...' at {loc}: {annot_err}")

            if annotated_count > 0:
                logger.info(f"Added {annotated_count} highlight annotations.")
                annotated_bytes = doc.tobytes(garbage=4, deflate=True)
            else:
                logger.warning("No annotations were successfully added. Returning original PDF bytes.")
                annotated_bytes = self.pdf_bytes

            return annotated_bytes

        except Exception as e:
            logger.error(f"Failed to add annotations: {str(e)}", exc_info=True)
            return self.pdf_bytes  # Return original on error
        finally:
            if doc: doc.close()

    @classmethod
    def _get_method_priority(cls, method: Optional[str]) -> int:
        if not method:
            return 99
        if method in cls._HIGHLIGHT_METHOD_PRIORITY:
            return cls._HIGHLIGHT_METHOD_PRIORITY[method]
        if method.endswith("_expanded"):
            base = method.rsplit("_expanded", 1)[0]
            if base in cls._HIGHLIGHT_METHOD_PRIORITY:
                return cls._HIGHLIGHT_METHOD_PRIORITY[base]
        if method.startswith("exact_"):
            return 0
        return 99

    @classmethod
    def _get_search_flags(cls, *, ignore_case: bool = True) -> int:
        flags = 0
        for attr in ("TEXT_DEHYPHENATE", "TEXT_PRESERVE_WHITESPACE", "TEXT_PRESERVE_LIGATURES"):
            flags |= getattr(fitz, attr, 0)
        if ignore_case:
            flags |= getattr(fitz, "TEXT_IGNORECASE", 0)
        return flags

    @classmethod
    def _normalize_for_search(
        cls,
        text: str,
        *,
        convert_quotes: bool = False,
        strip_quotes: bool = False,
    ) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\u00A0", " ")
        normalized = normalized.translate(cls._SEARCH_CHAR_TRANSLATION)
        if convert_quotes:
            normalized = normalized.translate(cls._QUOTE_TRANSLATION)
        if strip_quotes:
            normalized = normalized.replace('"', "").replace("'", "")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _build_search_candidates(
        cls,
        original_phrase: str,
        chunk_text: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        candidates: List[Tuple[str, str]] = []
        if not original_phrase:
            return candidates

        seen: Dict[str, bool] = {}

        def add_candidate(label: str, value: str) -> None:
            candidate = value.strip()
            if candidate and candidate not in seen:
                seen[candidate] = True
                candidates.append((label, candidate))

        # Base normalized versions
        normalized_original = cls._normalize_for_search(original_phrase, convert_quotes=False, strip_quotes=False)
        add_candidate("exact_original_search", normalized_original)
        add_candidate(
            "exact_normalized_search",
            cls._normalize_for_search(original_phrase, convert_quotes=True, strip_quotes=False),
        )
        add_candidate(
            "exact_quote_stripped_search",
            cls._normalize_for_search(original_phrase, convert_quotes=True, strip_quotes=True),
        )

        # DISABLED: Extract section header prefixes from chunk text if available
        # This caused regression - needs more careful implementation
        # if chunk_text and normalized_original:
        #     prefixes = cls._extract_section_prefixes(chunk_text, normalized_original)
        #     for prefix in prefixes:
        #         # Try adding the prefix to each base candidate
        #         add_candidate(
        #             "exact_with_section_prefix",
        #             prefix + normalized_original
        #         )
        #         # Also try normalized version with prefix
        #         normalized_with_quotes = cls._normalize_for_search(original_phrase, convert_quotes=True, strip_quotes=False)
        #         if normalized_with_quotes != normalized_original:
        #             add_candidate(
        #                 "exact_normalized_with_section_prefix",
        #                 prefix + normalized_with_quotes
        #             )

        return candidates

    @classmethod
    def _extract_section_prefixes(cls, chunk_text: str, normalized_phrase: str) -> List[str]:
        """Extract potential section header prefixes from chunk text that precede the phrase."""
        prefixes: List[str] = []
        if not chunk_text or not normalized_phrase:
            return prefixes

        # Normalize chunk text for comparison
        normalized_chunk = cls._normalize_for_search(chunk_text, convert_quotes=False, strip_quotes=False)
        
        # Find where the phrase might appear in the chunk
        phrase_start_pos = normalized_chunk.find(normalized_phrase[:50])  # Use first 50 chars for matching
        if phrase_start_pos == -1:
            return prefixes

        # Extract text before the phrase (up to 100 chars)
        prefix_text = normalized_chunk[max(0, phrase_start_pos - 100):phrase_start_pos]
        
        # Common legal document section header patterns
        patterns = [
            r'([A-Z][a-z]+\.\s*\([a-z]\)\s*)$',  # "Fees. (a) "
            r'(\([a-z]+\)\s*)$',                   # "(a) " or "(iv) "
            r'(\d+\.\d+\s+[A-Z][a-z]+\.\s*)$',    # "2.07 Fees. "
            r'(\d+\.\d+\s*)$',                     # "2.07 "
            r'([A-Z][a-z]+\s+\d+\.\d+\s*)$',      # "Section 2.07 "
            r'([A-Z][A-Z\s]+\.\s*)$',              # "DEFAULT RATE INTEREST. "
            r'(\([a-z]\)\s+[A-Z][a-z]+\.\s*)$',   # "(a) Fees. "
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prefix_text)
            if match:
                prefix = match.group(1)
                # Verify this prefix + phrase exists in original chunk
                if (prefix + normalized_phrase[:30]) in normalized_chunk:
                    prefixes.append(prefix)
                    break  # Use first matching pattern

        return prefixes

    @staticmethod
    def _expand_rect(rect: fitz.Rect, expand_by: float, page: fitz.Page) -> fitz.Rect:
        if rect.is_empty or expand_by <= 0:
            return rect
        page_rect = page.rect
        expanded = fitz.Rect(
            max(rect.x0 - expand_by, page_rect.x0),
            max(rect.y0 - expand_by, page_rect.y0),
            min(rect.x1 + expand_by, page_rect.x1),
            min(rect.y1 + expand_by, page_rect.y1),
        )
        return expanded

    def _find_exact_matches(
        self,
        *,
        page: fitz.Page,
        clip_rect: fitz.Rect,
        original_phrase: str,
        chunk_text: Optional[str] = None,
    ) -> List[Tuple[fitz.Rect, str]]:
        if not original_phrase:
            return []

        candidates = self._build_search_candidates(original_phrase, chunk_text)
        if not candidates:
            return []

        if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
            highlight_debug_logger.info(f"  Search candidates generated: {len(candidates)}")
            for idx, (label, candidate) in enumerate(candidates, 1):
                highlight_debug_logger.info(f"    Candidate {idx} ({label}): '{candidate}'")

        flags = self._get_search_flags(ignore_case=True)

        results: List[Tuple[fitz.Rect, str]] = []
        target_rect = fitz.Rect(clip_rect) if clip_rect and not clip_rect.is_empty else None

        for label, candidate in candidates:
            try:
                if target_rect:
                    search_hits = page.search_for(candidate, clip=target_rect, flags=flags, quads=False)
                else:
                    search_hits = page.search_for(candidate, flags=flags, quads=False)
            except Exception as search_error:
                logger.debug("Exact search error for candidate '%s': %s", candidate[:80], search_error)
                if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                    highlight_debug_logger.info(f"    Search error for '{label}': {search_error}")
                continue

            if search_hits:
                if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                    highlight_debug_logger.info(f"    ✓ Match found with '{label}' ({len(search_hits)} instance(s))")
                for rect in search_hits:
                    results.append((rect, label))
                return results
            else:
                if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                    highlight_debug_logger.info(f"    ✗ No match with '{label}'")

        if target_rect:
            expanded = self._expand_rect(target_rect, 2.0, page)
            if expanded and expanded != target_rect and not expanded.is_empty:
                if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                    highlight_debug_logger.info(f"  Trying expanded clip rectangle...")
                for label, candidate in candidates:
                    try:
                        search_hits = page.search_for(candidate, clip=expanded, flags=flags, quads=False)
                    except Exception as search_error:
                        logger.debug("Expanded search error for candidate '%s': %s", candidate[:80], search_error)
                        continue

                    if search_hits:
                        if ENABLE_HIGHLIGHT_DEBUG_LOGGING and highlight_debug_logger:
                            highlight_debug_logger.info(f"    ✓ Match found with expanded '{label}' ({len(search_hits)} instance(s))")
                        for rect in search_hits:
                            results.append((rect, f"{label}_expanded"))
                        return results

        return results

    def _find_fuzzy_span_in_chunk(
        self,
        *,
        original_phrase: str,
        chunk: Dict[str, Any],
        page: fitz.Page,
        doc: fitz.Document,
        fuzzy_score: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best fuzzy matching span within a chunk's text when exact search fails.
        Returns bounding box for the matched span to enable precise highlighting.
        """
        chunk_text = chunk.get("text", "")
        if not chunk_text or not original_phrase:
            return None

        # Normalize both for comparison
        normalized_phrase = normalize_text(remove_markdown_formatting(original_phrase))
        normalized_chunk = normalize_text(chunk_text)

        # Use sliding window to find best matching span
        phrase_words = normalized_phrase.split()
        chunk_words = normalized_chunk.split()
        
        if len(phrase_words) < 3:  # Too short, not reliable
            return None

        best_score = 0
        best_start_word = -1
        best_end_word = -1
        window_size = len(phrase_words)

        # Slide through chunk looking for best match
        for i in range(len(chunk_words) - window_size + 1):
            window = " ".join(chunk_words[i:i + window_size])
            # Use token_set_ratio for better flexibility with word order
            score = fuzz.token_set_ratio(normalized_phrase, window)
            
            if score > best_score:
                best_score = score
                best_start_word = i
                best_end_word = i + window_size

        # Require high similarity for fuzzy span match
        if best_score < 85 or best_start_word == -1:
            return None

        # Get character positions in original chunk text
        # Reconstruct from word positions
        current_word_idx = 0
        char_start = -1
        char_end = -1
        current_pos = 0

        for char in chunk_text:
            if current_word_idx == best_start_word and char_start == -1:
                char_start = current_pos
            if current_word_idx == best_end_word and char_end == -1:
                char_end = current_pos
                break
            if char.isspace():
                if current_pos > 0 and not chunk_text[current_pos - 1].isspace():
                    current_word_idx += 1
            current_pos += 1

        if char_start == -1 or char_end == -1:
            return None

        matched_text = chunk_text[char_start:char_end]

        # Try to find this span on the page
        page_num = chunk.get("page_num", -1)
        if not (0 <= page_num < doc.page_count):
            return None

        # Use page.search_for to find the matched span
        flags = self._get_search_flags(ignore_case=True)
        try:
            # Clean the matched text
            search_text = self._normalize_for_search(matched_text, convert_quotes=False, strip_quotes=False)
            rects = page.search_for(search_text, flags=flags, quads=False)
            
            if rects and len(rects) > 0:
                # Combine all rectangles (handles multi-line text)
                combined_rect = fitz.Rect(rects[0])
                for rect in rects[1:]:
                    combined_rect.include_rect(rect)
                
                return {
                    "rect": [combined_rect.x0, combined_rect.y0, combined_rect.x1, combined_rect.y1],
                    "matched_text": matched_text,
                    "fuzzy_score": best_score,
                }
        except Exception as e:
            logger.debug(f"Error during fuzzy span search: {e}")

        return None
