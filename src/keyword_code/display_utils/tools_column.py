"""
Tools column display for analysis results.
"""

import streamlit as st
import base64
import json
import zipfile
import fitz
import pandas as pd
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any
from ..config import logger, RAG_TOP_K
from ..models.embedding import load_embedding_model, load_reranker_model
from ..rag.retrieval import retrieve_relevant_chunks_for_chat
from ..utils.async_utils import run_async
from ..ai.analyzer import DocumentAnalyzer
from ..ai.chat import generate_chat_response
from .pdf_utils import update_pdf_view, regenerate_annotated_pdfs_from_chat_chunks
from .citation_utils import (
    process_chat_response_for_numbered_citations,
    display_chat_message_with_citations
)
from .export_utils import export_to_word

def _get_embedding_model():
    """Lazily retrieve the shared embedding model."""
    return load_embedding_model()


def _get_reranker_model():
    """Lazily retrieve the reranker model."""
    return load_reranker_model()


def display_tools_column(results_with_real_analysis: List, tools_col):
    """
    Display the tools column with PDF viewer, chat, and export options.
    
    Args:
        results_with_real_analysis: List of tuples containing (result, ai_analysis) pairs
        tools_col: Streamlit column object for the tools
    """
    embedding_model = _get_embedding_model()
    reranker_model = _get_reranker_model()

    with tools_col:
        # Header style from app.py.bak
        st.markdown('<div class="header-title">Analysis Tools & PDF Viewer</div>', unsafe_allow_html=True)
        st.markdown('<hr style="margin: 12px 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)

        # Container for tools
        with st.container():
            # SmartChat Expander
            with st.expander("💬 SmartChat (Multi-Document Chat)", expanded=False):
                st.caption("Chat with multiple documents simultaneously to get cross-referenced answers and insights.")
                if not st.session_state.get("preprocessed_data"):
                    st.info("Upload and process documents to enable chat.")
                else:
                    chat_container = st.container(height=400, border=True)
                    with chat_container:
                        # Use enumerate to get the index of each message in the session state list
                        for msg_idx, message in enumerate(st.session_state.chat_messages):
                            with st.chat_message(message["role"]):
                                if message["role"] == "assistant":
                                    processed_text = message.get("processed_text", message["content"])
                                    citation_details = message.get("citation_details", [])
                                    # Pass the message index (msg_idx) to the display function
                                    display_chat_message_with_citations(processed_text, citation_details, msg_idx)
                                else:
                                    # Apply the same word-wrap styling to user messages for consistency
                                    st.markdown(
                                        f'<div style="word-wrap: break-word; overflow-wrap: break-word; white-space: normal;">{message["content"]}</div>',
                                        unsafe_allow_html=True
                                    )

                    if prompt := st.chat_input("Ask about the uploaded documents...", key="chat_input_main"):
                        st.session_state.chat_messages.append({"role": "user", "content": prompt})
                        processed_chat_text = "Error: Could not generate response."
                        chat_citation_details = []
                        raw_ai_response_content = ""
                        try:
                            with st.spinner("Thinking..."):
                                logger.info(f"Chat RAG started for: {prompt[:50]}...")
                                # Use same retrieval depth as main analysis for consistency
                                relevant_chunks = retrieve_relevant_chunks_for_chat(
                                    prompt=prompt,
                                    top_k_per_doc=RAG_TOP_K,
                                    embedding_model=embedding_model,
                                    reranker_model=reranker_model,  # Use local reranker model
                                    preprocessed_data=st.session_state.get("preprocessed_data", {})
                                )
                                analyzer = DocumentAnalyzer()
                                logger.info(f"Generating chat response for: {prompt[:50]}...")
                                raw_ai_response_content = run_async(
                                    generate_chat_response(
                                        analyzer,
                                        prompt,
                                        relevant_chunks
                                    )
                                )
                                logger.info("Chat response generated.")
                                processed_chat_text, chat_citation_details = process_chat_response_for_numbered_citations(raw_ai_response_content)

                                # Refresh PDF highlighting to reflect the new RAG chunks
                                try:
                                    regenerate_annotated_pdfs_from_chat_chunks(relevant_chunks)
                                except Exception as _e:
                                    logger.warning(f"Could not refresh PDF highlights for chat: {_e}")

                        except Exception as chat_err:
                            logger.error(f"Error during chat processing: {chat_err}", exc_info=True)
                            processed_chat_text = f"Sorry, an error occurred while processing your request: {str(chat_err)}"
                            chat_citation_details = []
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": raw_ai_response_content,
                            "processed_text": processed_chat_text,
                            "citation_details": chat_citation_details
                        })
                        st.rerun()

            # Export Results Expander
            with st.expander("📊 Export Results", expanded=False):
                st.caption("Export analysis results in Excel or Word format for further review and documentation.")
                # Prepare data for export
                exportable_results_list = []

                for result, ai_analysis in results_with_real_analysis:
                    filename = result.get("filename", "Unknown File")
                    verification_results = result.get("verification_results", {})
                    phrase_locations = result.get("phrase_locations", {})

                    # Prepare a list of flattened data for this file
                    file_data = []
                    phrase_details: Dict[str, Any] = {}

                    for section_key, section_data in ai_analysis.get("analysis_sections", {}).items():
                        section_name = section_key.replace("_", " ").title()
                        analysis_text = section_data.get("Analysis", "")
                        context = section_data.get("Context", "")

                        # Process each supporting phrase
                        supporting_phrases = section_data.get("Supporting_Phrases", [])
                        if supporting_phrases and supporting_phrases != ["No relevant phrase found."]:
                            for phrase in supporting_phrases:
                                phrase_verification = verification_results.get(phrase, {})
                                phrase_location_data = phrase_locations.get(phrase, {})

                                is_verified = False
                                score = 0
                                best_location = None

                                if isinstance(phrase_verification, bool):
                                    is_verified = phrase_verification
                                elif isinstance(phrase_verification, dict):
                                    is_verified = phrase_verification.get("verified", False)
                                    score = phrase_verification.get("score", 0)

                                # Find best location
                                candidate_locs = []
                                if isinstance(phrase_location_data, list):
                                    candidate_locs = [loc for loc in phrase_location_data if isinstance(loc, dict)]
                                elif isinstance(phrase_location_data, dict):
                                    if 'best_match' in phrase_location_data:
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

                                    best_location = max(candidate_locs, key=loc_key)
                                    phrase_details.setdefault(phrase, {})["candidate_locations"] = candidate_locs
                                else:
                                    phrase_details.setdefault(phrase, {})["candidate_locations"] = []

                                # Calculate page number
                                if isinstance(best_location, dict) and "page_num" in best_location:
                                    page_num = best_location.get("page_num", -1) + 1
                                else:
                                    page_num = "Unknown"

                                # Determine match score display value
                                match_score_value = None
                                if isinstance(best_location, dict):
                                    match_score_value = best_location.get("match_score")
                                if match_score_value is None:
                                    match_score_value = score

                                if match_score_value:
                                    try:
                                        match_score_display = f"{float(match_score_value):.1f}%"
                                    except Exception:
                                        match_score_display = str(match_score_value)
                                else:
                                    match_score_display = "N/A"

                                phrase_details.setdefault(phrase, {}).update({
                                    "verified": is_verified,
                                    "best_location": best_location,
                                    "match_score": match_score_value,
                                })

                                file_data.append({
                                    "Filename": filename,
                                    "Section": section_name,
                                    "Analysis": analysis_text,
                                    "Context": context,
                                    "Supporting Phrase": phrase,
                                    "Verified": "Yes" if is_verified else "No",
                                    "Page": page_num,
                                    "Match Score": match_score_display
                                })

                    # Add this file's data to the exportable results
                    exportable_results_list.append({
                        "filename": filename,
                        "data": file_data,
                        "analysis": ai_analysis,
                        "phrase_details": phrase_details,
                        "phrase_locations": phrase_locations,
                        "annotated_pdf": result.get("annotated_pdf")
                    })

                # Excel Export
                if exportable_results_list:
                    # Flatten all data for Excel export
                    flat_data = []
                    for file_result in exportable_results_list:
                        flat_data.extend(file_result["data"])

                    # Add follow-up Q&A data if any exist
                    followup_qa = st.session_state.get("followup_qa", [])
                    if followup_qa:
                        for i, qa_pair in enumerate(followup_qa):
                            flat_data.append({
                                "Filename": "Follow-up Q&A",
                                "Section": f"Question {i+1}",
                                "Analysis": f"Q: {qa_pair.get('question', '')}\n\nA: {qa_pair.get('answer', '')}",
                                "Context": f"Asked on: {qa_pair.get('timestamp', 'Unknown')}",
                                "Supporting Phrase": f"Citations: {len(qa_pair.get('citation_details', []))} found",
                                "Verified": "N/A",
                                "Page": "N/A",
                                "Match Score": "N/A"
                            })

                    # Create DataFrame and export to Excel
                    df = pd.DataFrame(flat_data)
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Analysis Results')
                    excel_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel",
                        use_container_width=True
                    )

                # Word Export
                if exportable_results_list:
                    word_bytes = export_to_word(exportable_results_list)
                    st.download_button(
                        label="📥 Download Word",
                        data=word_bytes,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_word",
                        use_container_width=True
                    )

            # Fact Extraction (beta) Expander - Separate expander as in original
            with st.expander("🧪 Fact Extraction (beta)", expanded=False):
                st.caption("Identify fact types and extract structured information from analysis text.")

                if st.button("Generate Facts", key="compute_fact_definitions_beta"):
                    with st.spinner("Extracting facts using LLM-based analysis..."):
                        try:
                            from src.keyword_code.services.fact_extraction_service import FactExtractionService

                            # Initialize the fact extraction service
                            fact_service = FactExtractionService()

                            # Show progress and debug info
                            st.info(f"Processing {len(results_with_real_analysis)} document(s)...")

                            # Debug: Show what sections we're processing
                            total_sections = 0
                            for res, ai in results_with_real_analysis:
                                sections = (ai or {}).get("analysis_sections", {}) or {}
                                total_sections += len([s for s in sections.values() if s.get("Analysis")])
                            st.info(f"Found {total_sections} sections with analysis text")

                            # Extract facts using the new service
                            rows = fact_service.extract_fact_definitions_for_results(results_with_real_analysis)

                            if rows:
                                try:
                                    # Use the new multi-sheet export helper from the service
                                    from src.keyword_code.services.fact_extraction_service import export_fact_definitions_to_excel_bytes

                                    excel_bytes = export_fact_definitions_to_excel_bytes(rows)
                                    st.session_state["facts_defs_excel"] = excel_bytes
                                    st.success(f"✅ Extracted {len(rows)} facts using intelligent LLM-based analysis.")

                                    # Show a preview (first 10 facts across all rows)
                                    st.subheader("Preview (first 10 facts)")
                                    preview_df = pd.DataFrame([
                                        {"Fact": r.get("Fact", ""), "Definition": r.get("Definition", "")}
                                        for r in rows[:10]
                                    ])
                                    st.dataframe(preview_df, use_container_width=True)
                                except Exception as _exp:
                                    logger.error(f"Error creating multi-sheet Excel for facts: {_exp}", exc_info=True)
                                    st.error(f"❌ Error preparing Excel export: {_exp}")
                                    # Fallback to previous single-sheet behavior
                                    df_two = pd.DataFrame([{"Fact": r.get("Fact", ""), "Definition": r.get("Definition", "")} for r in rows])
                                    buf = BytesIO()
                                    df_two.to_excel(buf, index=False, engine="openpyxl")
                                    buf.seek(0)
                                    st.session_state["facts_defs_excel"] = buf.getvalue()
                                    st.warning("⚠️ Export used fallback single-sheet format due to an error creating multi-sheet workbook.")

                            else:
                                st.warning("⚠️ No facts extracted. This could be due to:")
                                st.write("- No analysis text in the selected documents")
                                st.write("- LLM extraction did not identify any facts")
                                st.write("- Analysis text may not contain extractable factual information")

                        except Exception as _fd_err:
                            st.error(f"❌ Error extracting facts: {_fd_err}")
                            logger.error(f"Fact extraction error: {_fd_err}", exc_info=True)

                if st.session_state.get("facts_defs_excel"):
                    st.download_button(
                        label="📥 Export Fact Definitions (Excel)",
                        data=st.session_state.get("facts_defs_excel"),
                        file_name=f"fact_definitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_fact_defs_excel"
                    )

            # Report Issue Expander
            with st.expander("🐞 Report Issue", expanded=False):
                st.markdown("""
                ### Report an Issue

                If you encounter any problems with the analysis or have feedback, please describe the issue below.
                A report package will be generated that you can send to the CNT Automations team.

                Positive feedback is good, negative feedback is even better!

                """)

                # Issue description input
                issue_description = st.text_area(
                    "Issue Description",
                    placeholder="Please describe the issue you're experiencing...",
                    height=150
                )

                # Create report package filename
                report_filename = f'smartdocs_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'

                # Check if description is provided
                download_disabled = not issue_description.strip()

                # Create the report package content
                try:
                    # Generate package content only if description is provided
                    if not download_disabled:
                        # Create a function to generate the report package
                        def create_report_package_for_download(desc):
                            try:
                                report_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "issue_description": desc,
                                    "user_inputs": {
                                        "prompt": st.session_state.get('user_prompt', ''),
                                    },
                                    "analysis_results": st.session_state.get('analysis_results', None),
                                    "current_document": st.session_state.get('current_pdf_name', None),
                                    "preprocessed_data_keys": list(st.session_state.get('preprocessed_data', {}).keys()),
                                    "chat_history_summary": [
                                        {"role": msg.get("role"), "content_preview": msg.get("content", "")[:100]+"..."}
                                        for msg in st.session_state.get("chat_messages", [])
                                    ],
                                    "followup_qa": st.session_state.get("followup_qa", [])
                                }

                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    # Write report data as JSON
                                    try:
                                        zip_file.writestr('report_data.json', json.dumps(report_data, indent=2, default=str))
                                    except Exception as json_err:
                                        zip_file.writestr('report_data_error.txt', f"Error serializing report data: {json_err}")
                                        logger.error(f"Error serializing report_data.json: {json_err}", exc_info=True)

                                    # Write original uploaded files
                                    uploaded_file_objs = st.session_state.get('uploaded_file_objects')
                                    if uploaded_file_objs:
                                        for uploaded_file in uploaded_file_objs:
                                            try:
                                                if hasattr(uploaded_file, 'name') and hasattr(uploaded_file, 'getvalue'):
                                                    zip_file.writestr(f'original_docs/{uploaded_file.name}', uploaded_file.getvalue())
                                                else:
                                                    logger.warning(f"Skipping invalid file object in uploaded_file_objects during report creation: {type(uploaded_file)}")
                                            except Exception as file_read_err:
                                                zip_file.writestr(f'original_docs/ERROR_{uploaded_file.name}.txt', f"Error reading file: {file_read_err}")
                                                logger.error(f"Error reading file {uploaded_file.name} for report package: {file_read_err}", exc_info=True)

                                    # Write annotated PDFs
                                    analysis_results_list = st.session_state.get('analysis_results')
                                    if analysis_results_list:
                                        for result in analysis_results_list:
                                            if isinstance(result, dict) and 'annotated_pdf' in result and result.get('annotated_pdf'):
                                                try:
                                                    pdf_bytes = base64.b64decode(result['annotated_pdf'])
                                                    pdf_filename = result.get('filename', f'unknown_annotated_{result.get("timestamp", "ts")}.pdf')
                                                    zip_file.writestr(f'annotated_pdfs/{pdf_filename}', pdf_bytes)
                                                except Exception as pdf_err:
                                                    zip_file.writestr(f'annotated_pdfs/ERROR_{result.get("filename", "unknown")}.txt', f"Error decoding/writing PDF: {pdf_err}")
                                                    logger.error(f"Error writing annotated PDF {result.get('filename')} to report: {pdf_err}", exc_info=True)

                                zip_buffer.seek(0)
                                return zip_buffer.getvalue()
                            except Exception as zip_e:
                                logger.error(f"Error creating report package zip file: {zip_e}", exc_info=True)
                                # Create a simple error zip as fallback
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    zip_file.writestr('error_creating_report.txt', f"Failed to create full report package: {zip_e}")
                                zip_buffer.seek(0)
                                return zip_buffer.getvalue()

                        # Create the download button
                        st.download_button(
                            label="📥 Download Report Package",
                            data=create_report_package_for_download(issue_description),
                            file_name=report_filename,
                            mime='application/zip',
                            disabled=download_disabled,
                            help="Download the report package to attach to your email.",
                            key="download_report_button",
                            use_container_width=True
                        )

                        st.success("""
                        Report package created successfully. Please download and email it to cnt_automations@ifc.org.

                        The package includes:
                        - Your issue description
                        - Analysis results
                        - Original documents
                        - Annotated PDFs
                        - Chat history
                        """)
                    else:
                        # Show disabled button with message
                        st.button(
                            "📥 Download Report Package",
                            disabled=True,
                            key="disabled_download_button",
                            help="Please provide an issue description first",
                            use_container_width=True
                        )
                        st.info("Please provide a description of the issue before downloading the report package.")
                except Exception as e:
                    st.error(f"Error preparing report package: {str(e)}")
                    logger.error(f"Error preparing report package for download button: {str(e)}", exc_info=True)

                st.info("Note: The report package will include the uploaded documents and analysis results to help diagnose the issue.")

            # PDF Viewer Expander
            with st.expander("📄 PDF Viewer", expanded=st.session_state.get("show_pdf", False)):
                # Add an anchor for scrolling to PDF viewer
                st.markdown('<div id="pdf-viewer-anchor"></div>', unsafe_allow_html=True)

                if st.session_state.get("pdf_bytes") and st.session_state.get("show_pdf", False):
                    fitz_doc = None  # Initialize fitz_doc
                    try:
                        pdf_bytes = st.session_state.pdf_bytes
                        current_page = st.session_state.get("pdf_page", 1)
                        filename = st.session_state.get("current_pdf_name", "Document")

                        # Display filename
                        st.caption(f"**{filename}**")

                        # Render the current page using PyMuPDF
                        fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        page_count = len(fitz_doc)

                        if page_count == 0:
                            st.warning("The PDF document appears to have 0 pages.")
                        else:
                            # Ensure current page is valid
                            current_page = max(1, min(current_page, page_count))

                            # Page navigation
                            nav_key = f"pdf_nav_{filename}_{page_count}_{hash(pdf_bytes)}"
                            new_page = st.number_input(
                                "Page",
                                min_value=1,
                                max_value=page_count,
                                value=current_page,
                                step=1,
                                key=nav_key,
                                help=f"Enter page number (1-{page_count})"
                            )
                            if new_page != current_page:
                                update_pdf_view(pdf_bytes, new_page, filename)
                                st.rerun()

                            st.caption(f"Page {current_page} of {page_count}")

                            # Render the page
                            page = fitz_doc.load_page(current_page - 1)  # 0-indexed
                            pix = page.get_pixmap(dpi=150)
                            img_bytes = pix.tobytes("png")

                            # Display the page image
                            st.image(img_bytes, use_container_width=True)

                    except Exception as e:
                        logger.error(f"Error displaying PDF: {e}")
                        st.error(f"Error displaying PDF: {e}")
                    finally:
                        if fitz_doc:
                            fitz_doc.close()
                else:
                    st.info("Select a document to view by clicking on a 'Go' button in the analysis or using the 'View' button.")

