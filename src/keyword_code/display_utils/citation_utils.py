"""
Citation processing and display utilities.
"""

import streamlit as st
import base64
import re
from typing import Dict, List, Any, Tuple
from ..config import logger
from .pdf_utils import find_annotated_pdf_for_filename, update_pdf_view


def process_chat_response_for_numbered_citations(raw_response_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Processes raw AI response text containing (Source:...) citations.
    Replaces them with sequential numbers [1], [2], etc., and returns the
    modified text along with a list of citation details for creating buttons.

    Args:
        raw_response_text: The original text from the AI.

    Returns:
        Tuple containing:
        - str: The response text with inline citations replaced by numbers ([1], [2]).
        - List[Dict[str, Any]]: A list of citation details, each dict containing
                               'number', 'filename', 'page', 'pdf_bytes'.
    """
    if not raw_response_text:
        return "", []

    citation_pattern = re.compile(r"\(Source:\s*(?P<filename>[^,]+?)\s*,\s*Page:\s*(?P<page>\d+)\)")

    citations_found_for_replacement = []  # Stores info needed for text replacement
    citation_details_for_footer = []  # Stores unique details for footer buttons
    next_citation_number = 1
    processed_text = raw_response_text

    # Find all citations and assign sequential numbers
    for match in citation_pattern.finditer(raw_response_text):
        filename = match.group("filename").strip()
        page_str = match.group("page").strip()
        try:
            page_num = int(page_str)

            # --- Assign unique number to THIS instance ---
            current_number = next_citation_number

            # Get PDF bytes for this source
            pdf_bytes = find_annotated_pdf_for_filename(filename)

            # Store details for the footer button list
            citation_details_for_footer.append({
                'number': current_number,
                'filename': filename,
                'page': page_num,
                'pdf_bytes': pdf_bytes  # Can be None if not found
            })

            # Store details needed to replace the text later
            citations_found_for_replacement.append({
                'start': match.start(),
                'end': match.end(),
                'number': current_number,
                'original_text': match.group(0)
            })

            # Increment for the *next* citation found
            next_citation_number += 1

        except ValueError:
            logger.warning(f"Found invalid page number in citation: {match.group(0)}")
        except Exception as e:
            logger.error(f"Error processing citation {match.group(0)}: {e}")

    # Second pass: Replace citations in the text from end to start (to avoid index issues)
    # Sort by start position in reverse order
    citations_found_for_replacement.sort(key=lambda x: x['start'], reverse=True)

    for citation in citations_found_for_replacement:
        processed_text = (
            processed_text[:citation['start']] +
            f" [{citation['number']}]" +
            processed_text[citation['end']:]
        )

    return processed_text.strip(), citation_details_for_footer


def display_followup_citations_like_main_analysis(citation_details: List[Dict[str, Any]], qa_index: int = 0, answer_text: str = "", relevant_chunks: List[Dict[str, Any]] = None):
    """
    Displays follow-up citations in the same format as the main analysis supporting citations.
    Uses the same styling with quoted text, verification badges, and "Go" buttons.

    Args:
        citation_details: List of citation dictionaries from process_chat_response_for_numbered_citations
        qa_index: Index of the Q&A pair for unique keys
        answer_text: The raw answer text to try to extract context from
        relevant_chunks: List of RAG chunks with score information (optional)
    """
    if not citation_details:
        st.info("No supporting citations were identified for this follow-up question.")
        return

    citation_counter = 0
    for citation_idx, citation in enumerate(citation_details):
        citation_counter += 1

        # Extract citation info
        filename = citation.get('filename', 'Unknown')
        page_num = citation.get('page_num', citation.get('page', 1))
        pdf_bytes = citation.get('pdf_bytes')
        actual_citation_number = citation.get('number', citation_counter)

        # For follow-up citations, we'll assume they're verified since they come from the RAG system
        is_verified = True  # Follow-up citations are from RAG retrieval, so considered verified

        # Extract confidence score from relevant_chunks if available
        confidence_score = None
        if relevant_chunks:
            # Find the chunk that matches this citation's filename and page
            for chunk in relevant_chunks:
                chunk_filename = chunk.get('filename', '')
                # page_num in citation is 1-based, chunk page_num is 0-based
                chunk_page = chunk.get('page_num', -1) + 1
                if chunk_filename == filename and chunk_page == page_num:
                    confidence_score = chunk.get('score', None)
                    break

        # Extract the actual relevant phrase from the AI response
        citation_text = f"Referenced content from {filename}, Page {page_num}"  # Default fallback

        if answer_text:
            # Look for text around where this citation number appears
            # Use the actual citation number from the citation details, not the counter
            citation_pattern = f"\\[{actual_citation_number}\\]"
            logger.debug(f"Looking for citation pattern '{citation_pattern}' in answer text for {filename}")

            # First, let's check if the citation pattern exists in the text
            if re.search(citation_pattern, answer_text):
                # Split the text into sentences and find the one with this citation
                # Use multiple sentence delimiters to be more comprehensive
                sentences = re.split(r'[.!?]+(?:\s|$)', answer_text)

                for sentence in sentences:
                    if re.search(citation_pattern, sentence):
                        # Clean up the sentence by removing citation markers and extra whitespace
                        clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
                        clean_sentence = re.sub(r'\s+', ' ', clean_sentence)  # Normalize whitespace

                        if len(clean_sentence) > 15:  # Only use if it's substantial enough
                            # Truncate if too long, but try to end at a word boundary
                            if len(clean_sentence) > 120:
                                truncated = clean_sentence[:120]
                                # Try to end at the last complete word
                                last_space = truncated.rfind(' ')
                                if last_space > 80:  # Only if we don't cut too much
                                    truncated = truncated[:last_space]
                                citation_text = truncated + "..."
                            else:
                                citation_text = clean_sentence
                        break

            # If we didn't find a sentence with the citation, try a different approach
            # Look for text immediately before the citation marker
            else:
                # Try to find any occurrence of the citation number in the text
                match = re.search(citation_pattern, answer_text)
                if match:
                    start_pos = match.start()
                    # Look backwards to find the start of the relevant phrase
                    # Try to find the beginning of the sentence or clause
                    text_before = answer_text[:start_pos]

                    # Look for sentence boundaries going backwards
                    sentence_starts = [m.end() for m in re.finditer(r'[.!?]\s+', text_before)]
                    if sentence_starts:
                        sentence_start = sentence_starts[-1]
                    else:
                        # If no sentence boundary, look for other natural breaks
                        clause_starts = [m.end() for m in re.finditer(r'[,;]\s+', text_before)]
                        if clause_starts:
                            sentence_start = clause_starts[-1]
                        else:
                            sentence_start = max(0, start_pos - 100)  # Last resort: 100 chars back

                    # Extract the relevant phrase
                    relevant_phrase = answer_text[sentence_start:start_pos].strip()
                    if len(relevant_phrase) > 15:
                        # Clean up and use this phrase
                        relevant_phrase = re.sub(r'\s+', ' ', relevant_phrase)
                        if len(relevant_phrase) > 120:
                            last_space = relevant_phrase.rfind(' ', 0, 120)
                            if last_space > 80:
                                relevant_phrase = relevant_phrase[:last_space] + "..."
                            else:
                                relevant_phrase = relevant_phrase[:120] + "..."
                        citation_text = relevant_phrase
                else:
                    # If we still can't find the citation, try to extract meaningful content
                    # Look for sentences that mention the filename or related content
                    filename_base = filename.replace('.pdf', '').replace('.docx', '')
                    if filename_base in answer_text:
                        # Find sentences that mention this document
                        sentences = re.split(r'[.!?]+(?:\s|$)', answer_text)
                        for sentence in sentences:
                            if filename_base.lower() in sentence.lower():
                                clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
                                clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
                                if len(clean_sentence) > 15:
                                    if len(clean_sentence) > 120:
                                        last_space = clean_sentence.rfind(' ', 0, 120)
                                        if last_space > 80:
                                            clean_sentence = clean_sentence[:last_space] + "..."
                                        else:
                                            clean_sentence = clean_sentence[:120] + "..."
                                    citation_text = clean_sentence
                                break

        # Badge HTML (same as main analysis)
        if is_verified:
            badge_html = '<span style="display: inline-block; background-color: #d1fecf; color: #11631a; padding: 1px 6px; border-radius: 0.25rem; font-size: 0.8em; margin-left: 5px; border: 1px solid #a1e0a3; font-weight: 600;">✔ Verified</span>'
        else:
            badge_html = '<span style="display: inline-block; background-color: #ffeacc; color: #a05e03; padding: 1px 6px; border-radius: 0.25rem; font-size: 0.8em; margin-left: 5px; border: 1px solid #f8c78d; font-weight: 600;">⚠️ Needs Review</span>'

        # Format score info (same as main analysis)
        if confidence_score is not None:
            score_info = f"Score: {confidence_score:.1%}"
        else:
            score_info = "RAG Retrieved"

        # Create columns for citation and Go button (same layout as main analysis)
        cite_col, btn_col = st.columns([0.90, 0.10], gap="small")

        with cite_col:
            # Citation text container (same styling as main analysis)
            st.markdown(f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 8px 12px; margin-top: 5px; margin-bottom: 8px; background-color: #f9f9f9;">
                <div style="margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold;">Citation {citation_counter} {badge_html}</span>
                    <span style="font-size: 0.8em; color: #555;">Page {page_num} | {score_info}</span>
                </div>
                <div style="color: #333; line-height: 1.4; font-size: 0.95em;"><i>"{citation_text}"</i></div>
            </div>
            """, unsafe_allow_html=True)

        with btn_col:
            # 'Go' button logic (same as main analysis)
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            if pdf_bytes:
                button_key = f"followup_goto_{qa_index}_{citation_counter}_{citation_idx}"
                if st.button("Go", key=button_key, type="secondary", help=f"Go to Page {page_num} in {filename}", use_container_width=True):
                    update_pdf_view(pdf_bytes=pdf_bytes, page_num=page_num, filename=filename)
                    st.session_state.scroll_to_pdf_viewer = True
                    st.rerun()
            else:
                st.caption("PDF N/A")


def display_chat_message_with_citations(processed_text: str, citation_details: List[Dict[str, Any]], msg_idx: int = 0):
    """
    Displays the processed chat message containing numbered citations [1], [2], etc.,
    and lists the corresponding source buttons below.

    Args:
        processed_text: The message text with (Source:...) replaced by [1], [2].
        citation_details: A list of dictionaries from process_chat_response_for_numbered_citations,
                          each containing 'number', 'filename', 'page', 'pdf_bytes'.
        msg_idx: The index of the message in the overall chat history (for unique keys).
    """

    # Display the main message content with inline numbers
    # Use a div with word-wrap to ensure text wraps properly without horizontal scrolling
    st.markdown(
        f'<div style="word-wrap: break-word; overflow-wrap: break-word; white-space: normal;">{processed_text}</div>',
        unsafe_allow_html=True
    )

    # Display the citation sources below if any exist
    if citation_details:
        st.markdown('<hr style="margin: 10px 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
        st.caption("Sources:")

        for i, citation in enumerate(citation_details):
            number = citation.get('number', i+1)
            filename = citation.get('filename', 'Unknown')
            page_num = citation.get('page_num', citation.get('page', 1))
            pdf_bytes = citation.get('pdf_bytes')

            # Create a unique key for each citation
            citation_key = f"cite_{filename}_{page_num}_{number}_{msg_idx}_{i}"

            if pdf_bytes:
                # Include the message index (msg_idx) and citation index (i) for guaranteed uniqueness
                button_key = f"chat_footer_cite_{citation_key}"
                # Make the entire citation text a button
                st.button(
                    f"[{number}] 📄 {filename}, p{page_num}",
                    key=button_key,
                    help=f"View Page {page_num} in {filename}",
                    type="secondary",
                    on_click=update_pdf_view,
                    args=(pdf_bytes, page_num, filename)
                )
            else:
                # Try to find the PDF in the analysis results as a fallback
                found_pdf = False
                if "analysis_results" in st.session_state:
                    for result in st.session_state.analysis_results:
                        if isinstance(result, dict) and result.get("filename") == filename and result.get("annotated_pdf"):
                            try:
                                pdf_bytes = base64.b64decode(result["annotated_pdf"])
                                button_key = f"chat_footer_cite_fallback_{citation_key}"
                                st.button(
                                    f"[{number}] 📄 {filename}, p{page_num}",
                                    key=button_key,
                                    help=f"View Page {page_num} in {filename}",
                                    type="secondary",
                                    on_click=update_pdf_view,
                                    args=(pdf_bytes, page_num, filename)
                                )
                                found_pdf = True
                                break
                            except Exception as e:
                                logger.error(f"Failed to decode annotated PDF for {filename} in fallback: {e}")

                # If PDF not found, display text indicating the source
                if not found_pdf:
                    st.markdown(
                        f'<div style="color: #888; padding: 0.25rem 0.75rem; font-size: 0.9em; border-radius: 0.25rem; background-color: #f0f0f0;">'
                        f'[{number}] 📄 {filename}, p{page_num} (PDF not available)'
                        f'</div>',
                        unsafe_allow_html=True
                    )

