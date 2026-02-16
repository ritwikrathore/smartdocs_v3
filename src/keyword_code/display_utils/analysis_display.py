"""
Analysis results display functions.
"""

import streamlit as st
import base64
import json
import re
import pandas as pd
import fitz
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Any, Optional
from ..config import logger, RAG_TOP_K
from ..models.embedding import load_embedding_model, load_reranker_model
from ..rag.retrieval import retrieve_relevant_chunks_for_chat
from ..utils.async_utils import run_async
from ..ai.analyzer import DocumentAnalyzer
from ..ai.chat import generate_chat_response
from .pdf_utils import update_pdf_view, regenerate_annotated_pdfs_from_chat_chunks
from .citation_utils import (
    process_chat_response_for_numbered_citations,
    display_followup_citations_like_main_analysis,
    display_chat_message_with_citations
)
from .export_utils import export_to_word
from ..utils.helpers import render_limited_markdown
from ..utils.langfuse_tracing import (
    start_span,
    set_span_output,
    record_span_error,
    get_langfuse_client_cached,
)
# from ..app import process_followup_question  # Moved to local import to avoid circular dependency

def _get_embedding_model():
    """Lazily retrieve the shared embedding model."""
    return load_embedding_model()


def _get_reranker_model():
    """Lazily retrieve the reranker model."""
    return load_reranker_model()


def display_analysis_results(results: List[Dict[str, Any]]):
    """
    Displays the analysis results in a structured format with a two-column layout.
    Left column shows AI analysis, right column shows tools including PDF viewer, chat, and export options.

    Args:
        results: A list of result dictionaries, each containing analysis data for a file
    """
    # Initialize followup_qa if it doesn't exist
    if "followup_qa" not in st.session_state:
        st.session_state.followup_qa = []

    if not results:
        st.warning("No analysis results to display.")
        return

    embedding_model = _get_embedding_model()
    reranker_model = _get_reranker_model()

    if "guided_prompt_defaults" not in st.session_state:
        st.session_state.guided_prompt_defaults = {}

    def _make_guided_prompt_key(filename: str, section_key: str) -> str:
        raw = f"guided_prompt_{filename}_{section_key}"
        return re.sub(r"[^0-9a-zA-Z_]+", "_", raw)

    # Define CSS styles based on app.py.bak
    st.markdown("""
    <style>
    .header-title {
        font-weight: 700;
        font-size: 1.5rem;
        color: #333; /* From app.py.bak's .header-title */
        margin: 0;
        padding: 0;
    }
    .sleek-container {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 0 0 16px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #e0e0e0;
    }
    .file-name {
        font-weight: 600;
        color: #424242; /* From app.py.bak's .file-name */
        font-size: 1rem;
        display: flex;
        align-items: center;
        margin: 0;
        padding: 0;
    }
    .file-icon {
        color: #1976d2; /* From app.py.bak's .file-icon */
        margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a two-column layout for the analysis results
    analysis_col, tools_col = st.columns([2.5, 1.5], gap="small")

    # Add an anchor for auto-scrolling
    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

    # Left Column: AI Analysis Display
    with analysis_col:
        # Header style from app.py.bak
        st.markdown('<div class="header-title">AI Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<hr style="margin: 12px 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)

        # Create a scrollable container for the analysis
        with st.container(height=1300, border=True):
            # Process results to extract only those with real analysis
            results_with_real_analysis = []
            for result in results:
                try:
                    filename = result.get("filename", "Unknown File")
                    ai_analysis_str = result.get("ai_analysis", "{}")

                    try:
                        ai_analysis = json.loads(ai_analysis_str)
                        # Only include results with actual analysis sections
                        if ai_analysis.get("analysis_sections", {}):
                            results_with_real_analysis.append((result, ai_analysis))
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse analysis JSON for {filename}")
                        continue
                except Exception as e:
                    logger.error(f"Error processing result for {result.get('filename', 'unknown')}: {e}")

            # If we have results with analysis, create tabs for each document
            if results_with_real_analysis:
                tab_titles = [res[0].get("filename", f"Result {i+1}") for i, res in enumerate(results_with_real_analysis)]
                tabs = st.tabs(tab_titles)

                for i, (result, ai_analysis) in enumerate(results_with_real_analysis):
                    with tabs[i]:
                        filename = result.get("filename", "Unknown File")
                        annotated_pdf_b64 = result.get("annotated_pdf")
                        annotated_pdf_bytes = base64.b64decode(annotated_pdf_b64) if annotated_pdf_b64 else None

                        # File info and download button row (app.py.bak style)
                        file_col1, file_col2 = st.columns([0.8, 0.2])
                        with file_col1:
                            st.markdown(f"""
                            <div class="sleek-container">
                                <div class="file-name">
                                    <span class="file-icon">📄</span> {filename}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with file_col2:
                            if annotated_pdf_bytes:
                                # Simpler label from app.py.bak
                                download_label = "💾 PDF"
                                st.download_button(
                                    label=download_label,
                                    data=annotated_pdf_bytes,
                                    file_name=f"{filename.replace('.pdf', '').replace('.docx', '')}_annotated.pdf",
                                    mime="application/pdf",
                                    key=f"download_pdf_{i}_{filename}",  # Ensure unique key
                                    use_container_width=True,  # Consistent with app.py.bak button style
                                    help=f"Download annotated PDF for {filename}"  # Added help text
                                )
                            else:
                                st.caption("No PDF")

                        # Map section keys to their originating sub-prompts for guided retries
                        section_prompt_map: Dict[str, str] = {}
                        sub_prompt_results = result.get("sub_prompt_results") or []
                        for idx, sub_prompt_entry in enumerate(sub_prompt_results, start=1):
                            if not isinstance(sub_prompt_entry, dict):
                                continue
                            section_identifier = sub_prompt_entry.get("section_key")
                            if not isinstance(section_identifier, str) or not section_identifier:
                                title_candidate = sub_prompt_entry.get("title", f"section_{idx}")
                                if isinstance(title_candidate, str) and title_candidate:
                                    section_identifier = f"section_{idx}_{title_candidate.replace(' ', '_').lower()}"
                                else:
                                    section_identifier = f"section_{idx}_sub_prompt"
                            prompt_text = sub_prompt_entry.get("sub_prompt", "")
                            if isinstance(prompt_text, str):
                                section_prompt_map[section_identifier] = prompt_text

                        # Display analysis sections
                        analysis_sections = ai_analysis.get("analysis_sections", {})
                        citation_counter = 0  # For numbering citations within a tab
                        is_keyword_mode_result = bool(result.get("keyword_mode"))
                        keyword_mode_sections = result.get("keyword_mode_sections", {}) or {}

                        for section_key, section_data in analysis_sections.items():
                            # Extract the actual title from the section key (removing "section_n_" prefix)
                            # Example: "section_1_investment_amount" -> "investment amount"
                            section_title = section_key
                            # Check if it follows the pattern section_N_title
                            if re.match(r'^section_\d+_', section_key):
                                # Extract just the title part after section_N_
                                section_title = re.sub(r'^section_\d+_', '', section_key)
                            elif re.match(r'^followup_\d+_\d+_', section_key):
                                # Extract just the title part after followup_TIMESTAMP_N_
                                section_title = re.sub(r'^followup_\d+_\d+_', '', section_key)

                            # Format section name for display
                            display_section_name = section_title.replace("_", " ").title()
                            keyword_section_details = None
                            if is_keyword_mode_result and isinstance(keyword_mode_sections, dict):
                                keyword_section_details = keyword_mode_sections.get(section_key)
                                if isinstance(keyword_section_details, dict):
                                    keyword_label = keyword_section_details.get("keyword")
                                    if isinstance(keyword_label, str) and keyword_label.strip():
                                        display_section_name = keyword_label.strip()

                            default_prompt_for_section = section_prompt_map.get(section_key)
                            allow_guided_prompt = bool(default_prompt_for_section) and not keyword_section_details and not section_key.startswith("error_")
                            prompt_state_key = None
                            user_prompt_value: str | None = None
                            guided_prompt_changed = False

                            # Create a container for the section title with improved styling and RAG retry button
                            with st.container(border=False):
                                # Create columns for title and RAG retry button
                                title_col, rag_col = st.columns([0.9, 0.1], gap="small")

                                with title_col:
                                    if allow_guided_prompt:
                                        prompt_state_key = _make_guided_prompt_key(filename, section_key)
                                        default_prompt_text = default_prompt_for_section or ""
                                        guided_defaults = st.session_state.guided_prompt_defaults
                                        stored_default = guided_defaults.get(prompt_state_key)
                                        current_session_value = st.session_state.get(prompt_state_key)

                                        # Initialize session state only if not present
                                        if stored_default is None:
                                            guided_defaults[prompt_state_key] = default_prompt_text
                                            if current_session_value is None:
                                                st.session_state[prompt_state_key] = default_prompt_text
                                        elif stored_default != default_prompt_text:
                                            if current_session_value is None or (current_session_value or "").strip() == stored_default.strip():
                                                st.session_state[prompt_state_key] = default_prompt_text
                                            guided_defaults[prompt_state_key] = default_prompt_text

                                        trimmed_default = (default_prompt_text or "").strip()
                                        if trimmed_default and not (st.session_state.get(prompt_state_key) or "").strip():
                                            # Ensure the guided prompt field displays the original question when empty
                                            st.session_state[prompt_state_key] = trimmed_default

                                        placeholder_text = trimmed_default or "Refine the retrieval prompt for this section..."

                                        # Use text_input for single-line field and rely solely on session state
                                        user_prompt_value = st.text_input(
                                            label="Guided prompt",
                                            key=prompt_state_key,
                                            label_visibility="collapsed",
                                            placeholder=placeholder_text,
                                            help="Edit the sub-prompt used for RAG retries. Leave unchanged to reuse the automatic version."
                                        )

                                        trimmed_current = (user_prompt_value or "").strip()
                                        guided_prompt_changed = bool(trimmed_current) and trimmed_current != trimmed_default
                                    else:
                                        st.markdown(f"""
                                            <div style='background-color: #f5f5f5; padding: 0px 16px; border-radius: 8px;
                                                    margin: 16px 0 8px 0; border-left: 4px solid #1976d2;'>
                                                <h4 style='color: #333; font-size: 1.2rem; margin: 0; font-weight: 600;'>
                                                    {display_section_name}
                                                </h4>
                                            </div>
                                        """, unsafe_allow_html=True)

                                with rag_col:
                                    # Check if this section has a context request
                                    try:
                                        analysis_json_str = section_data.get("analysis_json", "{}")
                                        if isinstance(analysis_json_str, str):
                                            analysis_obj = json.loads(analysis_json_str)
                                            context_request = analysis_obj.get("context_request")
                                            if isinstance(context_request, dict) and context_request.get("needs_more_context"):
                                                # Show context request approval button
                                                display_context_request_button(
                                                    section_key,
                                                    result,
                                                    section_data,
                                                    context_request,
                                                )
                                            else:
                                                # Show regular RAG retry button
                                                display_rag_retry_button_header(
                                                    section_key,
                                                    result,
                                                    section_data,
                                                    guided_prompt=user_prompt_value,
                                                    default_prompt=default_prompt_for_section,
                                                    prompt_changed=guided_prompt_changed,
                                                )
                                        else:
                                            # Fallback to regular retry button
                                            display_rag_retry_button_header(
                                                section_key,
                                                result,
                                                section_data,
                                                guided_prompt=user_prompt_value,
                                                default_prompt=default_prompt_for_section,
                                                prompt_changed=guided_prompt_changed,
                                            )
                                    except Exception as e:
                                        logger.error(f"Error checking for context request in section {section_key}: {e}")
                                        # Fallback to regular retry button
                                        display_rag_retry_button_header(
                                            section_key,
                                            result,
                                            section_data,
                                            guided_prompt=user_prompt_value,
                                            default_prompt=default_prompt_for_section,
                                            prompt_changed=guided_prompt_changed,
                                        )

                            # Display RAG analysis and retry results if available (below the header)
                            # Retry results are integrated into the main view; no separate section

                            # Section content in a bordered container
                            with st.container(border=True):
                                analysis_content = section_data.get("Analysis")
                                context_content = section_data.get("Context")

                                if analysis_content:
                                    # Render limited Markdown in analysis content
                                    rendered_analysis = render_limited_markdown(analysis_content)

                                    analysis_html_parts = [
                                        f"<div style='background-color: #f8f9fa; padding: .5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>",
                                        f"<h4 style='color: #1e88e5; font-size: 1.1rem;'>{display_section_name}</h4>",
                                        f"<div style='color: #424242; line-height: 1.6;'>{rendered_analysis}"
                                    ]
                                    
                                    # Check if this section has a context request and display it
                                    try:
                                        analysis_json_str = section_data.get("analysis_json", "{}")
                                        if isinstance(analysis_json_str, str):
                                            analysis_obj = json.loads(analysis_json_str)
                                            context_request = analysis_obj.get("context_request")
                                            if isinstance(context_request, dict) and context_request.get("needs_more_context"):
                                                request_reason = context_request.get("reason", "")
                                                chunk_indices = context_request.get("chunk_indices", [])
                                                
                                                # Build a description of what's being requested
                                                request_details = []
                                                if chunk_indices:
                                                    if len(chunk_indices) > 3:
                                                        request_details.append(f"Chunks {chunk_indices[0]}-{chunk_indices[-1]} ({len(chunk_indices)} chunks)")
                                                    else:
                                                        request_details.append(f"Chunks {', '.join(map(str, chunk_indices))}")
                                                
                                                articles = context_request.get("article_numbers", [])
                                                if articles:
                                                    request_details.append(f"Articles: {', '.join(articles[:3])}")
                                                
                                                sections = context_request.get("section_numbers", [])
                                                if sections:
                                                    request_details.append(f"Sections: {', '.join(sections[:3])}")
                                                
                                                titles = context_request.get("section_titles", [])
                                                if titles:
                                                    request_details.append(f"'{titles[0]}'")
                                                
                                                request_summary = " | ".join(request_details) if request_details else "Additional context"
                                                
                                                if request_reason:
                                                    analysis_html_parts.extend([
                                                        f"<div style='margin-top: 0.8rem; border-top: 1px solid #ffa726; padding-top: 0.8rem; background-color: #fff3e0; padding: 0.6rem; border-radius: 0.3rem;'>",
                                                        f"<span style='color: #e65100; font-size: 0.85rem; font-weight: 600;'>🔍 Context Request ({request_summary}):</span> ",
                                                        f"<span style='color: #424242; font-size: 0.9rem; line-height: 1.4;'>{request_reason}</span>",
                                                        f"</div>"
                                                    ])
                                    except Exception as e:
                                        logger.debug(f"Could not extract context request for section {section_key}: {e}")
                                    
                                    if context_content:
                                        # Context is NOT rendered with Markdown - keep as plain text
                                        analysis_html_parts.extend([
                                            f"<div style='margin-top: 0.8rem; border-top: 1px solid #e0e0e0; padding-top: 0.8rem;'>",
                                            f"<span style='color: #1b5e20; font-size: 0.9rem; line-height: 1.4;'>{context_content}</span>",
                                            f"</div>"
                                        ])
                                    analysis_html_parts.extend(["</div></div>"])
                                    st.markdown("".join(analysis_html_parts), unsafe_allow_html=True)
                                elif context_content:  # Display context even if analysis is missing
                                    # Context is NOT rendered with Markdown - keep as plain text
                                    st.markdown(f"""
                                        <div style='background-color: #f8f9fa; padding: .5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                                            <h4 style='color: #1e88e5; font-size: 1.1rem;'>Context</h4>
                                            <div style='color: #424242; line-height: 1.6;'>
                                                <span style='color: #1b5e20; font-size: 0.9rem; line-height: 1.4;'>{context_content}</span>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                            # Supporting Citations in an expander
                            supporting_phrases = section_data.get("Supporting_Phrases", [])
                            verification_results = result.get("verification_results", {})
                            phrase_locations = result.get("phrase_locations", {})
                            keyword_section_details = keyword_section_details if isinstance(keyword_section_details, dict) else (keyword_mode_sections.get(section_key) if isinstance(keyword_mode_sections, dict) else None)

                            any_needs_review = False
                            if supporting_phrases and supporting_phrases != ["No relevant phrase found."]:
                                for phrase in supporting_phrases:
                                    phrase_verification = verification_results.get(phrase, {})
                                    if isinstance(phrase_verification, dict):
                                        is_verified = phrase_verification.get("verified", False)
                                    elif isinstance(phrase_verification, bool):
                                        is_verified = phrase_verification
                                    else:
                                        is_verified = bool(phrase_verification)  # Fallback
                                    if not is_verified:
                                        any_needs_review = True
                                        break

                            total_keyword_matches = None
                            if keyword_section_details and isinstance(keyword_section_details, dict):
                                total_keyword_matches = keyword_section_details.get("total_occurrences")
                                if total_keyword_matches is None:
                                    total_keyword_matches = keyword_section_details.get("count")

                            expand_state = any_needs_review or bool(total_keyword_matches)

                            with st.expander("Supporting Citations", expanded=expand_state):
                                # If optimized RAG results exist for this section, use them to REPLACE the citations display
                                # Disabled displaying optimized RAG results separately; use verified citations from analysis
                                if keyword_section_details and total_keyword_matches:
                                    keyword_label = keyword_section_details.get("keyword")
                                    summary_parts = []
                                    if keyword_label:
                                        summary_parts.append(f"**Keyword:** `{keyword_label}`")
                                    summary_parts.append(f"**Total Matches:** {total_keyword_matches}")
                                    st.markdown(" • ".join(summary_parts))

                                new_results = []

                                if new_results:
                                    # This section handles optimized RAG results (currently disabled)
                                    pass
                                else:
                                    # Fallback to original citations if no optimized results
                                    if not supporting_phrases or supporting_phrases == ["No relevant phrase found."]:
                                        st.info("No supporting citations were identified for this section.")
                                    else:
                                        has_citations_to_show = False
                                        for phrase_idx, phrase in enumerate(supporting_phrases):
                                            if not isinstance(phrase, str) or phrase == "No relevant phrase found.":
                                                continue
                                            has_citations_to_show = True
                                            citation_counter += 1  # Increment citation counter

                                            phrase_verification = verification_results.get(phrase, {})
                                            phrase_location_data = phrase_locations.get(phrase, {})

                                            is_verified = False
                                            score = 0
                                            best_location_dict = {}

                                            if isinstance(phrase_verification, bool):
                                                is_verified = phrase_verification
                                            elif isinstance(phrase_verification, dict):
                                                is_verified = phrase_verification.get("verified", False)
                                                score = phrase_verification.get("score", 0)
                                            else:
                                                try:
                                                    is_verified = bool(phrase_verification)
                                                except Exception:
                                                    is_verified = False

                                            current_page_num_info = "Page unknown"
                                            current_score_info = "N/A"

                                            candidate_locs = []
                                            if isinstance(phrase_location_data, list):
                                                candidate_locs = [loc for loc in phrase_location_data if isinstance(loc, dict)]
                                            elif isinstance(phrase_location_data, dict):
                                                if 'best_match' in phrase_location_data and isinstance(phrase_location_data['best_match'], dict):
                                                    candidate_locs = [phrase_location_data['best_match']]
                                                else:
                                                    candidate_locs = [phrase_location_data]

                                            if candidate_locs:
                                                method_priority = {
                                                    'exact': 5,
                                                    'exact_cleaned_search': 5,
                                                    'special_case_quotes_handling': 3,
                                                    'cross_page_fuzzy_match_part1': 2,
                                                    'cross_page_fuzzy_match_part2': 2,
                                                    'fuzzy': 2,
                                                    'fuzzy_chunk_fallback_individual': 1,
                                                    'fuzzy_chunk_fallback': 0
                                                }

                                                def loc_key(loc):
                                                    method = loc.get('method', '')
                                                    score_val = loc.get('match_score', 0) or 0
                                                    try:
                                                        score_val = float(score_val)
                                                    except Exception:
                                                        score_val = 0.0
                                                    return (method_priority.get(method, -1), score_val)

                                                best_location_dict = max(candidate_locs, key=loc_key)
                                                page_num_val = best_location_dict.get("page_num")
                                                if isinstance(page_num_val, int):
                                                    current_page_num_info = f"Page {page_num_val + 1}"
                                                else:
                                                    current_page_num_info = f"Page {page_num_val}" if page_num_val is not None else "Page unknown"

                                                score_val = best_location_dict.get("match_score", score)
                                                if score_val:
                                                    try:
                                                        current_score_info = f"{float(score_val):.1f}"
                                                    except Exception:
                                                        current_score_info = str(score_val)
                                                elif score:
                                                    try:
                                                        current_score_info = f"{float(score):.1f}"
                                                    except Exception:
                                                        current_score_info = str(score)

                                                try:
                                                    cand_summaries = []
                                                    for loc in candidate_locs:
                                                        p = loc.get('page_num')
                                                        p_disp = (p + 1) if isinstance(p, int) else p
                                                        m = loc.get('method', '')
                                                        s = loc.get('match_score', 0)
                                                        try:
                                                            s = float(s) if s is not None else 0.0
                                                        except Exception:
                                                            s = 0.0
                                                        cand_summaries.append(f"p={p_disp}, m={m}, s={s:.2f}")
                                                    logger.debug(f"Candidates for phrase '{phrase[:50]}...': [" + "; ".join(cand_summaries) + "]")
                                                except Exception as _e:
                                                    logger.debug("Error building candidate summaries for logging")
                                                logger.debug(f"Selected best location out of {len(candidate_locs)} candidates for phrase '{phrase[:50]}...': page={page_num_val}")
                                            else:
                                                logger.debug(f"No candidate locations available for phrase '{phrase[:50]}...' to determine page")

                                            if is_verified:
                                                badge_html = '<span style="display: inline-block; background-color: #d1fecf; color: #11631a; padding: 1px 6px; border-radius: 0.25rem; font-size: 0.8em; margin-left: 5px; border: 1px solid #a1e0a3; font-weight: 600;">✔ Verified</span>'
                                            else:
                                                badge_html = '<span style="display: inline-block; background-color: #ffeacc; color: #a05e03; padding: 1px 6px; border-radius: 0.25rem; font-size: 0.8em; margin-left: 5px; border: 1px solid #f8c78d; font-weight: 600;">⚠️ Needs Review</span>'

                                            cite_col, btn_col = st.columns([0.90, 0.10], gap="small")
                                            with cite_col:
                                                st.markdown(f"""
                                                <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 8px 12px; margin-top: 5px; margin-bottom: 8px; background-color: #f9f9f9;">
                                                    <div style="margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                                                        <span style="font-weight: bold;">Citation {citation_counter} {badge_html}</span>
                                                        <span style="font-size: 0.8em; color: #555;">{current_page_num_info} | Score: {current_score_info}</span>
                                                    </div>
                                                    <div style="color: #333; line-height: 1.4; font-size: 0.95em;"><i>"{phrase}"</i></div>
                                                </div>
                                                """, unsafe_allow_html=True)

                                            with btn_col:
                                                st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                                                if is_verified and best_location_dict and isinstance(best_location_dict, dict) and "page_num" in best_location_dict and annotated_pdf_b64:
                                                    try:
                                                        page_num_to_go = best_location_dict["page_num"]
                                                        page_num_1_indexed = page_num_to_go + 1 if isinstance(page_num_to_go, int) else int(page_num_to_go) + 1
                                                        button_key = f"goto_{i}_{section_key}_{citation_counter}_{phrase_idx}"
                                                        if st.button("Go", key=button_key, type="secondary", help=f"Go to Page {page_num_1_indexed} in {filename}", use_container_width=True):
                                                            pdf_bytes_for_view = base64.b64decode(annotated_pdf_b64)
                                                            update_pdf_view(pdf_bytes=pdf_bytes_for_view, page_num=page_num_1_indexed, filename=filename)
                                                            st.session_state.scroll_to_pdf_viewer = True
                                                            st.rerun()
                                                    except Exception as e_go:
                                                        logger.error(f"Error setting up 'Go' button for citation: {e_go}")
                                                elif is_verified:
                                                    st.caption("Loc N/A")

                                        if not has_citations_to_show:
                                            st.caption("No supporting citations provided or found for this section.")

                            # Facts display removed per request (use Export Results > Export Facts)

                        # --- Follow-up Question UI (Per Document) ---
                        st.markdown('<hr style="margin: 20px 0; border: 0; border-top: 2px solid #e0e0e0;">', unsafe_allow_html=True)
                        st.markdown('<div class="header-title" style="font-size: 1.3rem;">Follow-up Questions</div>', unsafe_allow_html=True)
                        
                        # Input for follow-up
                        followup_key = f"followup_input_{i}_{filename}"
                        followup_btn_key = f"followup_btn_{i}_{filename}"
                        
                        input_col, button_col = st.columns([0.90, 0.10], gap="small")
                        with input_col:
                            followup_q = st.text_input(
                                "Ask a follow-up question for this document:",
                                placeholder="e.g., Can you provide more details about the investment timeline?",
                                key=followup_key
                            )
                        with button_col:
                            st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
                            ask_btn = st.button("➤", key=followup_btn_key, type="primary", disabled=not followup_q.strip(), use_container_width=True)
                            
                        if ask_btn and followup_q.strip():
                            with st.spinner("Processing follow-up question..."):
                                try:
                                    # Import here to avoid circular dependency
                                    from ..app import process_followup_question

                                    # Get preprocessed data
                                    preprocessed_data = st.session_state.get("preprocessed_data", {}).get(filename)
                                    if not preprocessed_data:
                                        st.error("Preprocessing data not found for this file.")
                                    else:
                                        # Call the new pipeline
                                        new_results = process_followup_question(filename, followup_q, preprocessed_data)
                                        
                                        # Merge results
                                        # 1. Merge analysis sections
                                        current_analysis_json = result.get("ai_analysis", "{}")
                                        try:
                                            current_analysis = json.loads(current_analysis_json)
                                        except json.JSONDecodeError:
                                            current_analysis = {}
                                            
                                        if "analysis_sections" not in current_analysis:
                                            current_analysis["analysis_sections"] = {}
                                        
                                        current_analysis["analysis_sections"].update(new_results["analysis_sections"])
                                        result["ai_analysis"] = json.dumps(current_analysis)
                                        
                                        # 2. Merge verification results and phrase locations
                                        if "verification_results" not in result:
                                            result["verification_results"] = {}
                                        result["verification_results"].update(new_results["verification_results"])
                                        
                                        if "phrase_locations" not in result:
                                            result["phrase_locations"] = {}
                                        result["phrase_locations"].update(new_results["phrase_locations"])
                                        
                                        # 3. Merge sub_prompt_results
                                        if "sub_prompt_results" not in result:
                                            result["sub_prompt_results"] = []
                                        # Append new ones
                                        result["sub_prompt_results"].extend(new_results.get("sub_prompt_results", []))
                                        
                                        # 4. Regenerate PDF
                                        # We need to regenerate the PDF with ALL annotations
                                        from ..processors.pdf_processor import PDFProcessor
                                        original_bytes = preprocessed_data.get("original_bytes")
                                        # Coerce common representations to raw bytes (base64 str, memoryview, bytearray)
                                        if isinstance(original_bytes, str):
                                            try:
                                                original_bytes = base64.b64decode(original_bytes)
                                            except Exception:
                                                logger.warning("original_bytes for %s is a string but not valid base64; skipping annotation", filename)
                                        elif isinstance(original_bytes, memoryview):
                                            original_bytes = bytes(original_bytes)
                                        elif isinstance(original_bytes, bytearray):
                                            original_bytes = bytes(original_bytes)

                                        # Try to restore if missing
                                        if not isinstance(original_bytes, (bytes, bytearray)):
                                            try:
                                                from .pdf_utils import restore_original_bytes_if_needed

                                                original_bytes = restore_original_bytes_if_needed(filename)
                                            except Exception:
                                                original_bytes = None

                                        if original_bytes:
                                            processor = PDFProcessor(original_bytes)
                                            annotated_pdf_bytes = processor.add_annotations(result["phrase_locations"])
                                            result["annotated_pdf"] = base64.b64encode(annotated_pdf_bytes).decode('utf-8')
                                            
                                            # Update PDF view if this file is currently being viewed
                                            if st.session_state.get("current_pdf_name") == filename:
                                                update_pdf_view(annotated_pdf_bytes, filename=filename)
                                        
                                        st.success("Follow-up analysis added!")
                                        st.rerun()
                                        
                                except Exception as e:
                                    logger.error(f"Error processing follow-up: {e}", exc_info=True)
                                    st.error(f"Error: {str(e)}")
            else:  # No results_with_real_analysis
                st.info("Processing complete, but no analysis sections were generated or found.")

            # Old follow-up section removed


    # Right Column: Tools & PDF Viewer
    from .tools_column import display_tools_column
    display_tools_column(results_with_real_analysis, tools_col)


def display_rag_retry_button_header(
    section_key: str,
    result: Dict[str, Any],
    section_data: Dict[str, Any],
    *,
    guided_prompt: Optional[str] = None,
    default_prompt: Optional[str] = None,
    prompt_changed: bool = False,
):
    """
    Display RAG retry button in the section header.

    Args:
        section_key: The section identifier
        result: Result dictionary containing analysis data
        section_data: The specific section data
    """
    if result.get("keyword_mode"):
        return

    # Create a unique key for this section's retry button
    retry_key = f"retry_rag_{section_key}_{result.get('filename', 'unknown')}"

    # Since this function is called within a column context, we cannot create nested columns
    # Instead, we'll create buttons directly without columns, stacked vertically
    # Analyze button removed per new agent/tool design

    trimmed_prompt = (guided_prompt or "").strip()
    trimmed_default = (default_prompt or "").strip()
    use_custom_prompt = prompt_changed and bool(trimmed_prompt)
    if not use_custom_prompt and trimmed_prompt and trimmed_prompt != trimmed_default:
        use_custom_prompt = True

    if st.button("↻", key=f"retry_{retry_key}", help="Retry Retrieval (beta)", use_container_width=True):
        # Store the retry request in session state
        if "rag_retry_requests" not in st.session_state:
            st.session_state.rag_retry_requests = {}

        request_payload: Dict[str, Any] = {
            "status": "requested",
            "section_data": section_data,
            "result": result,
            "default_prompt": default_prompt,
        }

        if use_custom_prompt:
            request_payload["custom_prompt"] = trimmed_prompt
            request_payload["guided_prompt_changed"] = True
        else:
            request_payload["guided_prompt_changed"] = False

        st.session_state.rag_retry_requests[section_key] = request_payload
        st.rerun()


def display_context_request_button(
    section_key: str,
    result: Dict[str, Any],
    section_data: Dict[str, Any],
    context_request: Dict[str, Any],
):
    """
    Display button for user to approve context request from LLM.

    Args:
        section_key: The section identifier
        result: Result dictionary containing analysis data
        section_data: The specific section data
        context_request: The context request details from the LLM
    """
    # Create a unique key for this section's context approval button
    context_key = f"context_req_{section_key}_{result.get('filename', 'unknown')}"

    if st.button("✔️", key=f"approve_{context_key}", help="Approve Context Request", use_container_width=True):
        # Store the context request approval in session state
        if "context_request_approvals" not in st.session_state:
            st.session_state.context_request_approvals = {}

        st.session_state.context_request_approvals[section_key] = {
            "status": "approved",
            "context_request": context_request,
            "section_data": section_data,
            "result": result,
        }
        st.rerun()


def display_rag_results_section(section_key: str):
    """
    Display RAG analysis and retry results for a section.

    Args:
        section_key: The section identifier
    """
    # Display analysis results if available
    if hasattr(st.session_state, 'rag_analysis_results') and section_key in st.session_state.rag_analysis_results:
        analysis = st.session_state.rag_analysis_results[section_key]

        with st.expander("📊 RAG Analysis Results", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Query Type", analysis.get("query_type", "Unknown"))
                st.metric("Quality Score", f"{analysis.get('current_quality_score', 0):.2f}")

            with col2:
                st.metric("Recommended BM25 Weight", f"{analysis.get('recommended_bm25_weight', 0.5):.2f}")
                st.metric("Recommended Semantic Weight", f"{analysis.get('recommended_semantic_weight', 0.5):.2f}")

            if analysis.get("issues_identified"):
                st.markdown("**Issues Identified:**")
                for issue in analysis["issues_identified"]:
                    st.markdown(f"• {issue}")

            if analysis.get("reasoning"):
                st.markdown("**Reasoning:**")
                st.markdown(analysis["reasoning"])

    # Display retry results if available
    if hasattr(st.session_state, 'rag_retry_results') and section_key in st.session_state.rag_retry_results:
        retry_data = st.session_state.rag_retry_results[section_key]

        with st.expander("🔄 RAG Retry Results", expanded=False):
            st.markdown("**New Retrieval Results:**")

            new_results = retry_data.get("new_results", [])
            if new_results:
                for i, chunk in enumerate(new_results[:3]):  # Show top 3 results
                    st.markdown(f"**Result {i+1}** (Score: {chunk.get('score', 0):.3f})")
                    # Convert from 0-based to 1-based page numbering for display
                    page_num = chunk.get('page_num', 'Unknown')
                    if isinstance(page_num, int):
                        page_display = page_num + 1
                    else:
                        page_display = page_num
                    st.markdown(f"*Page {page_display}*")
                    st.markdown(chunk.get('text', '')[:200] + '...')
                    st.markdown("---")
            else:
                st.info("No new results retrieved")

            # Show comparison with original results
            original_results = retry_data.get("original_results", [])
            original_count = len(original_results) if isinstance(original_results, list) else 0
            new_count = len(new_results)
            st.metric("Results Comparison", f"{new_count} new vs {original_count} original")

        # If we produced a new AI analysis for this retry, show it with validation and facts
        ai_section = retry_data.get("ai_section")
        if ai_section:
            with st.expander("🧠 AI Response (Retry) + Validation", expanded=True):
                guided_prompt_text = retry_data.get("guided_prompt")
                if isinstance(guided_prompt_text, str) and guided_prompt_text.strip():
                    prompt_label = "Guided Prompt Used" if retry_data.get("used_custom_prompt") else "Automatic Prompt Used"
                    st.markdown(f"**{prompt_label}:**")
                    st.code(guided_prompt_text.strip())

                # Analysis text with Markdown rendering
                analysis_text = ai_section.get("Analysis", "")
                if analysis_text:
                    st.markdown("**Analysis (Retry):**")
                    # Render limited Markdown in retry analysis
                    rendered_retry_analysis = render_limited_markdown(analysis_text)
                    st.markdown(rendered_retry_analysis, unsafe_allow_html=True)
                else:
                    st.info("No analysis text available from retry.")

                # Verified supporting citations
                supporting = ai_section.get("Supporting_Phrases", []) or []
                ver_results = retry_data.get("verification_results", {}) or {}
                phrase_locs = retry_data.get("phrase_locations", {}) or {}
                if supporting and supporting != ["No relevant phrase found."]:
                    st.markdown("**Verified Supporting Citations:**")
                    for idx, phrase in enumerate(supporting, start=1):
                        v = ver_results.get(phrase, False)
                        is_verified = v.get("verified", False) if isinstance(v, dict) else bool(v)
                        icon = "✅" if is_verified else "⚠️"
                        # Choose best location instead of first
                        pinfo = ""
                        locs = phrase_locs.get(phrase, [])
                        candidate_locs = [loc for loc in locs if isinstance(loc, dict)] if isinstance(locs, list) else []
                        if candidate_locs:
                            method_priority = {
                                'exact': 5,
                                'exact_cleaned_search': 5,
                                'special_case_quotes_handling': 3,
                                'cross_page_fuzzy_match_part1': 2,
                                'cross_page_fuzzy_match_part2': 2,
                                'fuzzy': 2,
                                'fuzzy_chunk_fallback_individual': 1,
                                'fuzzy_chunk_fallback': 0
                            }

                            def loc_key(loc):
                                method = loc.get('method', '')
                                score_val = loc.get('match_score', 0) or 0
                                try:
                                    score_val = float(score_val)
                                except Exception:
                                    score_val = 0.0
                                return (method_priority.get(method, -1), score_val)

                            best_loc = max(candidate_locs, key=loc_key)
                            page_val = best_loc.get("page_num")
                            try:
                                # Log all candidates for diagnostics
                                cand_summaries = []
                                for loc in candidate_locs:
                                    p = loc.get('page_num')
                                    p_disp = (p + 1) if isinstance(p, int) else p
                                    m = loc.get('method', '')
                                    s = loc.get('match_score', 0)
                                    try:
                                        s = float(s) if s is not None else 0.0
                                    except:
                                        s = 0.0
                                    cand_summaries.append(f"p={p_disp}, m={m}, s={s:.2f}")
                                logger.debug(f"[Retry] Candidates for phrase '{phrase[:50]}...': [" + "; ".join(cand_summaries) + "]")
                                logger.debug(f"[Retry] Selected best location page={page_val}")
                            except Exception:
                                pass
                            if isinstance(page_val, int):
                                pinfo = f" (Page {page_val + 1})"
                            elif page_val is not None:
                                pinfo = f" (Page {page_val})"
                        st.markdown(f"{icon} [{idx}] {phrase}{pinfo}")
                else:
                    st.info("No supporting phrases identified by the retry analysis.")

                # Facts display removed per request (use Export Results > Export Facts)


def display_section_facts_expander(section_key: str, section_data: Dict[str, Any], result: Dict[str, Any], citation_counter: int = 0):
    """
    Display extracted facts in an expander for a specific section.

    Args:
        section_key: The section identifier
        section_data: The specific section data containing Analysis text
        result: Result dictionary containing filename and other metadata
        citation_counter: Current citation counter for consistent numbering
    """

    # Check if we have facts for this specific section
    section_facts_key = f"section_facts_{section_key}_{result.get('filename', 'unknown')}"

    # Check if facts extraction is in progress or completed for this section
    if hasattr(st.session_state, 'section_facts') and section_facts_key in st.session_state.section_facts:
        facts_data = st.session_state.section_facts[section_facts_key]

        if facts_data.get("status") == "completed" and facts_data.get("facts"):
            facts = facts_data["facts"]

            st.markdown("--- ")
            st.markdown("##### 📊 Extracted Facts")
            # Group facts by category
            facts_by_category = {}
            for fact in facts:
                category = fact.get("category", "General")
                if category not in facts_by_category:
                    facts_by_category[category] = []
                facts_by_category[category].append(fact)

            # Display each category
            for category, category_facts in facts_by_category.items():
                if len(facts_by_category) > 1:
                    st.markdown(f"**{category.replace('_', ' ').title()}:**")

                for fact in category_facts:
                    fact_text = fact.get("text", "")
                    attributes = fact.get("attributes", {})

                    # Compact, badge-like fact display
                    fact_html = f'<div style="margin: 4px 0; font-size: 0.92em; color: #333;">'
                    fact_html += f'<span style="display:inline-block; padding:2px 8px; border-radius:999px; background:#eef5ff; color:#1e88e5; border:1px solid #cfe3ff; font-weight:600; margin-right:8px;">{category.replace("_", " ").title()}</span>'
                    fact_html += f'{fact_text}'

                    if attributes:
                        fact_html += '<span style="margin-left:8px; color:#666;">'
                        for key, value in list(attributes.items())[:3]:
                            fact_html += f'<span style="display:inline-block; padding:1px 6px; border-radius:999px; background:#f5f5f5; border:1px solid #e0e0e0; margin-right:6px; font-size:0.85em;">{key.replace("_", " ").title()}: {value}</span>'
                        fact_html += '</span>'

                    fact_html += '</div>'
                    st.markdown(fact_html, unsafe_allow_html=True)

            # Show extraction metadata
            metadata = facts_data.get("metadata", {})
            if metadata:
                st.caption(f"Extracted {len(facts)} facts using {metadata.get('model_used', 'Unknown model')}")

        elif facts_data.get("status") == "processing":
            st.markdown("--- ")
            st.markdown("##### 📊 Extracted Facts")
            st.info("🔄 Extracting facts from analysis...")

        elif facts_data.get("status") == "error":
            st.markdown("--- ")
            st.markdown("##### 📊 Extracted Facts")
            error_msg = facts_data.get("error", "Unknown error")
            st.error(f"❌ Error extracting facts from analysis: {error_msg}")

            # Add retry button
            if st.button(f"🔄 Retry Fact Extraction", key=f"retry_facts_{section_facts_key}"):
                # Reset the request status to trigger retry
                if "section_facts_requests" in st.session_state:
                    st.session_state.section_facts_requests[section_facts_key] = {
                        "status": "requested",
                        "section_key": section_key,
                        "analysis_text": section_data.get("Analysis", ""),
                        "section_data": section_data,
                        "result": result
                    }
                st.session_state.section_facts[section_facts_key] = {"status": "processing"}
                st.rerun()

    else:
        # Always show the facts section, even if no facts are available yet
        st.markdown("--- ")
        st.markdown("##### 📊 Extracted Facts")

        # Trigger fact extraction for this section if not already done
        if section_data.get("Analysis"):
            # Perform extraction synchronously to avoid per-section reruns
            from src.keyword_code.services.fact_extraction_service import FactExtractionService
            try:
                fact_service = FactExtractionService()
                extracted_facts = fact_service.extract_facts_from_text(
                    text=section_data.get("Analysis", ""),
                    context=f"Legal/Financial Analysis - Section: {section_key}",
                    section_name=section_key,
                    filename=result.get("filename", "Unknown")
                )
                if "section_facts" not in st.session_state:
                    st.session_state.section_facts = {}
                st.session_state.section_facts[section_facts_key] = {
                    "status": "completed",
                    "facts": extracted_facts.get("extracted_facts", []),
                    "metadata": {
                        "model_used": "Pydantic-AI Fact Extraction",
                        "total_extractions": len(extracted_facts.get("extracted_facts", [])),
                        "section_key": section_key
                    }
                }
            except Exception as _fe_err:
                logger.error(f"Synchronous fact extraction error for section {section_key}: {_fe_err}")
                st.session_state.section_facts[section_facts_key] = {"status": "error", "error": str(_fe_err)}
            st.caption("Facts extracted using LLM-based analysis.")
        else:
            st.info("No analysis text available for fact extraction.")

