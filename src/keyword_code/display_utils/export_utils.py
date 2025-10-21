"""
Export utilities for analysis results.
"""

import streamlit as st
import base64
import json
import zipfile
from io import BytesIO
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime
from typing import Dict, List, Any, Tuple
from ..config import logger


def create_report_package_content(issue_description: str, results: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> bytes:
    """
    Create a ZIP package containing all relevant information for issue reporting.

    Args:
        issue_description: User's description of the issue
        results: List of tuples containing (result, ai_analysis) pairs

    Returns:
        bytes: The ZIP file as bytes
    """
    # Create a BytesIO object to store the ZIP file
    zip_buffer = BytesIO()

    # Create a ZIP file
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Add report metadata
        report_data = {
            "issue_description": issue_description,
            "timestamp": datetime.now().isoformat(),
            "user_inputs": {
                "prompt": st.session_state.get("user_prompt", ""),
                "uploaded_files": list(st.session_state.get("preprocessed_data", {}).keys())
            },
            "chat_history": st.session_state.get("chat_messages", []),
            "followup_qa": st.session_state.get("followup_qa", []),
            "current_document": st.session_state.get("current_pdf_name", "")
        }

        # Add report data as JSON
        zip_file.writestr("report_data.json", json.dumps(report_data, indent=2))

        # Add analysis results
        for i, (result, ai_analysis) in enumerate(results):
            filename = result.get("filename", f"unknown_file_{i}")

            # Add the analysis result as JSON
            zip_file.writestr(
                f"analysis_results/{filename}_analysis.json",
                json.dumps(ai_analysis, indent=2)
            )

            # Add the annotated PDF if available
            if result.get("annotated_pdf"):
                try:
                    annotated_pdf_bytes = base64.b64decode(result["annotated_pdf"])
                    zip_file.writestr(
                        f"annotated_pdfs/{filename}_annotated.pdf",
                        annotated_pdf_bytes
                    )
                except Exception as e:
                    logger.error(f"Error adding annotated PDF for {filename} to report package: {e}")

            # Add the original document if available
            if "preprocessed_data" in st.session_state and filename in st.session_state.preprocessed_data:
                try:
                    original_bytes = st.session_state.preprocessed_data[filename].get("original_bytes")
                    if original_bytes:
                        zip_file.writestr(
                            f"original_documents/{filename}",
                            original_bytes
                        )
                except Exception as e:
                    logger.error(f"Error adding original document for {filename} to report package: {e}")

    # Reset buffer position and return the bytes
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_to_word(exportable_results_list: List[Dict[str, Any]]) -> bytes:
    """
    Export analysis results to a Word document.

    Args:
        exportable_results_list: List of dictionaries containing analysis results

    Returns:
        bytes: The Word document as bytes
    """
    # Create a new Word document
    doc = Document()

    # Add title
    title = doc.add_heading("Document Analysis Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add report generation date
    date_paragraph = doc.add_paragraph()
    date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_run = date_paragraph.add_run(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    date_run.italic = True

    # Add a page break after the title page
    doc.add_page_break()

    # Process each file's results
    for file_result in exportable_results_list:
        filename = file_result.get("filename", "Unknown File")
        analysis = file_result.get("analysis", {})

        # Add file heading
        doc.add_heading(f"Document: {filename}", 1)

        # Process each analysis section
        for section_key, section_data in analysis.get("analysis_sections", {}).items():
            # Add section heading
            section_name = section_key.replace("_", " ").title()
            doc.add_heading(section_name, 2)

            # Add analysis text
            if section_data.get("Analysis"):
                p = doc.add_paragraph()
                p.add_run("Analysis: ").bold = True
                p.add_run(section_data.get("Analysis"))

            # Add context if available
            if section_data.get("Context"):
                p = doc.add_paragraph()
                p.add_run("Context: ").bold = True
                p.add_run(section_data.get("Context"))

            # Add supporting phrases
            supporting_phrases = section_data.get("Supporting_Phrases", [])
            if supporting_phrases and supporting_phrases != ["No relevant phrase found."]:
                doc.add_heading("Supporting Citations", 3)

                for phrase in supporting_phrases:
                    # Get verification info from the file_result data
                    data_rows = []
                    try:
                        file_data = file_result.get("data", [])
                        if isinstance(file_data, list):
                            data_rows = [row for row in file_data
                                        if isinstance(row, dict) and row.get("Supporting Phrase") == phrase]
                    except Exception as e:
                        logger.error(f"Error getting data rows for phrase '{phrase}': {e}")

                    if data_rows:
                        try:
                            data_row = data_rows[0]
                            # Check if Verified is "Yes" or True
                            verified_value = data_row.get("Verified")
                            if isinstance(verified_value, str):
                                is_verified = verified_value.lower() == "yes"
                            elif isinstance(verified_value, bool):
                                is_verified = verified_value
                            else:
                                is_verified = False

                            # Get page number info
                            page_num_info = data_row.get("Page", "Unknown")

                            # Get score info
                            score_info = data_row.get("Match Score", "N/A")
                        except Exception as e:
                            logger.error(f"Error extracting verification info from data row: {e}")
                            is_verified = False
                            page_num_info = "Unknown"
                            score_info = "N/A"
                    else:
                        is_verified = False
                        page_num_info = "Unknown"
                        score_info = "N/A"

                    # Add the phrase with verification status
                    p = doc.add_paragraph()
                    if is_verified:
                        p.add_run("✓ ").bold = True
                        p.add_run(phrase)
                        details_run = p.add_run(f" (Pg: {page_num_info}, Score: {score_info})")
                        details_run.italic = True
                        details_run.font.size = Pt(9)
                    else:
                        p.add_run("❓ ").bold = True
                        p.add_run(phrase)
                        details_run = p.add_run(" (Not verified in document)")
                        details_run.italic = True
                        details_run.font.size = Pt(9)

            # Add a separator after each section
            doc.add_paragraph("---")

        # Add a page break after each file
        doc.add_page_break()

    # Add Follow-up Q&A section if any exist
    followup_qa = st.session_state.get("followup_qa", [])
    if followup_qa:
        doc.add_heading("Follow-up Questions & Answers", 1)

        for i, qa_pair in enumerate(followup_qa):
            # Add question
            doc.add_heading(f"Question {i+1}", 2)
            q_paragraph = doc.add_paragraph()
            q_paragraph.add_run("Q: ").bold = True
            q_paragraph.add_run(qa_pair.get("question", ""))

            # Add answer
            a_paragraph = doc.add_paragraph()
            a_paragraph.add_run("A: ").bold = True
            a_paragraph.add_run(qa_pair.get("answer", ""))

            # Add timestamp if available
            if qa_pair.get("timestamp"):
                timestamp_paragraph = doc.add_paragraph()
                timestamp_run = timestamp_paragraph.add_run(f"Asked on: {qa_pair['timestamp']}")
                timestamp_run.italic = True
                timestamp_run.font.size = Pt(9)

            # Add citations if available
            citation_details = qa_pair.get("citation_details", [])
            if citation_details:
                doc.add_heading("Citations", 3)
                for citation in citation_details:
                    cite_paragraph = doc.add_paragraph()
                    cite_paragraph.style = 'List Bullet'
                    cite_paragraph.add_run(f"[{citation.get('number', '')}] ")
                    cite_paragraph.add_run(f"{citation.get('filename', 'Unknown')}, ")
                    cite_paragraph.add_run(f"Page {citation.get('page_num', 'Unknown')}")

            # Add separator between Q&A pairs
            if i < len(followup_qa) - 1:
                doc.add_paragraph("---")

    # Save the document to a BytesIO object
    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)

    # Return the document as bytes
    return docx_buffer.getvalue()

