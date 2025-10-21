"""
PDF-related utilities for display.
"""

import streamlit as st
import base64
import fitz
from typing import Dict, List, Any, Optional
from ..config import logger
from ..processors.pdf_processor import PDFProcessor


def find_annotated_pdf_for_filename(filename: str) -> Optional[bytes]:
    """Finds the base64 decoded annotated PDF bytes for a given filename from session state."""
    for result in st.session_state.get("analysis_results", []):
        if isinstance(result, dict) and result.get("filename") == filename and result.get("annotated_pdf"):
            try:
                return base64.b64decode(result["annotated_pdf"])
            except Exception as e:
                logger.error(f"Failed to decode annotated PDF for {filename} in chat citation: {e}")
                return None
    logger.warning(f"Could not find annotated PDF data for {filename} in session state analysis_results.")
    return None


def regenerate_annotated_pdfs_from_chat_chunks(relevant_chunks: List[Dict[str, Any]]):
    """Regenerate annotated PDFs based on chat/follow-up RAG chunks so highlights stay current.

    For each filename present in relevant_chunks, this finds the original PDF bytes
    and chunk metadata (including bboxes) from session_state.preprocessed_data,
    builds a minimal phrase_locations structure using the union bbox of each retrieved
    chunk, and regenerates the annotated PDF via PDFProcessor.add_annotations.
    """
    try:
        if not relevant_chunks:
            return

        # Group retrieved chunks by filename
        chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for rc in relevant_chunks:
            fname = rc.get("filename")
            if not fname:
                continue
            chunks_by_file.setdefault(fname, []).append(rc)

        pre = st.session_state.get("preprocessed_data", {}) or {}

        for filename, file_chunks in chunks_by_file.items():
            pre_doc = pre.get(filename, {}) or {}
            orig_bytes = pre_doc.get("original_bytes")
            if not orig_bytes:
                logger.warning(f"Original PDF bytes not found for {filename}; skipping re-annotation.")
                continue

            # Map chunk_id -> chunk metadata to access page_num and bboxes
            meta_chunks: List[Dict[str, Any]] = pre_doc.get("chunks", []) or []
            by_id: Dict[Any, Dict[str, Any]] = {c.get("chunk_id"): c for c in meta_chunks if isinstance(c, dict)}

            # Build locations from the union of bboxes per relevant chunk
            locations: List[Dict[str, Any]] = []
            for rc in file_chunks:
                cid = rc.get("chunk_id")
                cmeta = by_id.get(cid)
                if not cmeta:
                    # Fallback attempt: match by text if chunk_id missing
                    ctext = rc.get("text")
                    if ctext:
                        cmeta = next((m for m in meta_chunks if m.get("text") == ctext), None)
                if not cmeta:
                    continue

                try:
                    rect = fitz.Rect()
                    for bbox in (cmeta.get("bboxes") or []):
                        try:
                            if isinstance(bbox, fitz.Rect):
                                rect.include_rect(bbox)
                            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                rect.include_rect(fitz.Rect(bbox))
                        except Exception:
                            continue
                    if rect.is_empty:
                        # If no bbox, skip this chunk (nothing to highlight)
                        continue

                    page_num = cmeta.get("page_num", rc.get("page_num"))
                    locations.append({
                        "page_num": page_num,
                        "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "chunk_id": cid,
                        "match_score": rc.get("score"),
                        "method": "chat_rag",
                        "phrase_text": (rc.get("text") or "RAG Chunk")[:120],
                    })
                except Exception as loc_err:
                    logger.warning(f"Skipping location build for {filename} chunk {cid}: {loc_err}")

            if not locations:
                logger.info(f"No highlightable locations for {filename} from chat RAG; keeping existing annotations.")
                continue

            phrase_locations = {"[Chat RAG]": locations}
            try:
                processor = PDFProcessor(orig_bytes)
                annotated_bytes = processor.add_annotations(phrase_locations)
            except Exception as ann_err:
                logger.error(f"Annotation regeneration failed for {filename}: {ann_err}")
                continue

            # Persist back into analysis_results for this filename
            updated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
            found = False
            for res in st.session_state.get("analysis_results", []):
                if isinstance(res, dict) and res.get("filename") == filename:
                    res["annotated_pdf"] = updated_b64
                    found = True
                    break
            if not found:
                # If result entry doesn't exist, append a minimal one
                st.session_state.setdefault("analysis_results", []).append({
                    "filename": filename,
                    "annotated_pdf": updated_b64,
                })

            # If currently viewing this PDF, update the live viewer bytes
            if st.session_state.get("current_pdf_name") == filename and st.session_state.get("show_pdf"):
                st.session_state.pdf_bytes = annotated_bytes

            logger.info(f"Regenerated annotated PDF for {filename} based on chat RAG results ({len(locations)} highlights).")
    except Exception as e:
        logger.error(f"Error during chat-based PDF re-annotation: {e}")


def update_pdf_view(pdf_bytes, page_num=1, filename=None):
    """
    Updates the PDF view in the session state.

    Args:
        pdf_bytes: The PDF bytes to display
        page_num: The page number to display (1-based)
        filename: The name of the file
    """
    if pdf_bytes:
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_page = page_num
        st.session_state.show_pdf = True
        if filename:
            st.session_state.current_pdf_name = filename
        logger.info(f"Updated PDF view to {filename}, page {page_num}")
    else:
        logger.warning("Attempted to update PDF view with empty bytes")
        st.session_state.show_pdf = False


def display_pdf_viewer(pdf_bytes, current_page=1, filename=None):
    """
    Displays a PDF viewer with navigation controls.

    Args:
        pdf_bytes: The PDF bytes to display
        current_page: The current page number (1-based)
        filename: The name of the file
    """
    if not pdf_bytes:
        st.warning("No PDF data available to display.")
        return

    try:
        # Create a base64 encoded PDF string
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

        # Display the filename if provided
        if filename:
            st.markdown(f"**Viewing:** {filename}")

        # Create an iframe to display the PDF
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Add page navigation controls
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("◀ Previous Page", disabled=(current_page <= 1)):
                update_pdf_view(pdf_bytes, current_page - 1, filename)
                st.rerun()

        with col2:
            st.markdown(f"<div style='text-align: center;'>Page {current_page}</div>", unsafe_allow_html=True)

        with col3:
            if st.button("Next Page ▶"):
                update_pdf_view(pdf_bytes, current_page + 1, filename)
                st.rerun()

    except Exception as e:
        logger.error(f"Error displaying PDF: {e}")
        st.error(f"Error displaying PDF: {e}")

