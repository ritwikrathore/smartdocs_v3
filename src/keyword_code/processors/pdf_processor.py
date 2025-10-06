"""
PDF processing functionality.
"""

import json
import fitz  # PyMuPDF
import re
from typing import Any, Dict, List, Optional, Tuple
from thefuzz import fuzz
from ..config import (
    logger, FUZZY_MATCH_THRESHOLD,
    SENTENCES_PER_CHUNK, MIN_CHUNK_CHAR_LENGTH
)
from ..utils.helpers import normalize_text, remove_markdown_formatting
from ..utils.spacy_utils import ensure_spacy_model
from ..rag.chunking import SentenceChunker, create_chunks_from_text


class PDFProcessor:
    """Handles PDF processing, chunking, verification, and annotation."""

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

            self._full_text = "\\n\\n".join(all_text_parts) # Join chunks for full text
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

    def verify_and_locate_phrases(
        self, ai_analysis_json_str: str  # Expects the *aggregated* JSON string
    ) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
        """Verifies AI phrases from the aggregated analysis against chunks and locates them."""
        verification_results = {}
        phrase_locations = {}

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
                                        cleaned_search_phrase = remove_markdown_formatting(original_phrase)
                                        cleaned_search_phrase = re.sub(r"\s+", " ", cleaned_search_phrase).strip()
                                        instances = page.search_for(cleaned_search_phrase, clip=clip_rect, quads=False)

                                        if instances:
                                            logger.debug(f"Found {len(instances)} instance(s) via search_for in chunk {chunk['chunk_id']} area for '{cleaned_search_phrase[:60]}...'")
                                            for rect in instances:
                                                if isinstance(rect, fitz.Rect) and not rect.is_empty:
                                                    phrase_locations[original_phrase].append({
                                                        "page_num": page_num,
                                                        "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                                                        "chunk_id": chunk["chunk_id"],
                                                        "match_score": score,
                                                        "method": "exact_cleaned_search",
                                                    })
                                        else:
                                            # Fallback to chunk bounding box if exact search fails within the verified chunk
                                            logger.debug(f"Exact search failed for '{cleaned_search_phrase[:60]}...' in verified chunk {chunk['chunk_id']} (score: {score}). Falling back to chunk bbox.")

                                            # Try to find a more precise area to highlight by looking at individual bboxes
                                            # rather than the combined clip_rect which can be very large
                                            individual_bboxes = chunk.get('bboxes', [])
                                            if individual_bboxes and len(individual_bboxes) <= 3:  # Only use individual boxes if there aren't too many
                                                for bbox in individual_bboxes:
                                                    if isinstance(bbox, fitz.Rect) and not bbox.is_empty:
                                                        phrase_locations[original_phrase].append({
                                                            "page_num": page_num,
                                                            "rect": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                                                            "chunk_id": chunk["chunk_id"],
                                                            "match_score": score,
                                                            "method": "fuzzy_chunk_fallback_individual",
                                                        })
                                            else:
                                                # If there are too many individual boxes or none, fall back to the combined area
                                                phrase_locations[original_phrase].append({
                                                    "page_num": page_num,
                                                    "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                                    "chunk_id": chunk["chunk_id"],
                                                    "match_score": score,
                                                    "method": "fuzzy_chunk_fallback",
                                                })
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
                                            phrase_locations[original_phrase].append({
                                                "page_num": page_A_num,
                                                "rect": [clip_rect_A.x0, clip_rect_A.y0, clip_rect_A.x1, clip_rect_A.y1],
                                                "chunk_id": chunk_A["chunk_id"],
                                                "match_score": score,  # Use combined score
                                                "method": "cross_page_fuzzy_match_part1",
                                            })

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
                                            phrase_locations[original_phrase].append({
                                                "page_num": page_B_num,
                                                "rect": [clip_rect_B.x0, clip_rect_B.y0, clip_rect_B.x1, clip_rect_B.y1],
                                                "chunk_id": chunk_B["chunk_id"],
                                                "match_score": score,  # Use combined score
                                                "method": "cross_page_fuzzy_match_part2",
                                            })
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
                                        phrase_locations[original_phrase].append({
                                            "page_num": page_num,
                                            "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                            "chunk_id": chunk["chunk_id"],
                                            "match_score": score,
                                            "method": "special_case_quotes_handling",
                                        })
                                break  # Found a match, no need to check other chunks

                    if not found_match_for_phrase:
                        logger.warning(f"NOT Verified: '{original_phrase[:60]}...' did not meet fuzzy threshold ({FUZZY_MATCH_THRESHOLD}) in any chunk or cross-page combination.")
            finally:
                if doc: doc.close()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse aggregated AI analysis JSON for verification: {e}")
            logger.debug(f"Problematic JSON string: {ai_analysis_json_str[:500]}...")  # Log start of bad JSON
        except Exception as e:
            logger.error(f"Error during phrase verification and location: {str(e)}", exc_info=True)

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
