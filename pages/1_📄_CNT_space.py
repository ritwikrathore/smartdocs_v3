# pages/1_📄_CNT_space.py

import streamlit as st
import base64
import logging
import os
import asyncio
import concurrent.futures
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
import threading
import streamlit_pills as stp
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(
    page_title="CNT SmartDocs - Document Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import necessary components directly from the modules
from src.keyword_code.models.embedding import load_embedding_model
from src.keyword_code.app import preprocess_file, process_file_wrapper
from src.keyword_code.utils.display import display_analysis_results, display_pdf_viewer, update_pdf_view
from src.keyword_code.utils.async_utils import run_async
from src.keyword_code.utils.helpers import get_base64_encoded_image
from src.keyword_code.utils.ui_helpers import apply_ui_styling, render_branding, initialize_session_state, display_welcome_features, display_review_features
from src.keyword_code.config import PROCESS_CYAN, DARK_BLUE, SAVED_PROMPTS

# Apply styling immediately to ensure it's loaded before any content is displayed
apply_ui_styling()
render_branding()

# --- Mode Switch and SmartReview integration helpers ---
# Store and render Ask/Review mode toggle at the very top of the page
if "smartdocs_mode" not in st.session_state:
    st.session_state.smartdocs_mode = "Ask"  # default

# Place the mode selector in the center column so it appears centered on the page
_left_col, _center_col, _right_col = st.columns([1, 0.5, 1])
with _center_col:
    # Inject tight CSS to make the radio look like pill buttons and use project theme blues
    st.markdown(f"""
    <style>
        /* Center the radiogroup and allow wrapping on small widths */
        div[role="radiogroup"] {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }}
        /* Hide the native radio circle */
        div[role="radiogroup"] label > div:first-child {{
            display: none !important;
        }}
        /* Base pill styling */
        div[role="radiogroup"] label > div {{
            background: transparent;
            color: inherit;
            border-radius: 999px;
            padding: 6px 14px;
            border: 1px solid rgba(0,0,0,0.08);
            transition: all .12s ease-in-out;
            cursor: pointer;
            font-weight: 600;
        }}
        /* Selected state uses the project's primary blue */
        div[role="radiogroup"] input[type="radio"]:checked + div {{
            background: {PROCESS_CYAN};
            color: #ffffff;
            border-color: {PROCESS_CYAN};
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        /* Keyboard focus state */
        div[role="radiogroup"] input[type="radio"]:focus + div {{
            outline: 2px solid {DARK_BLUE};
            outline-offset: 2px;
        }}
    </style>
    """, unsafe_allow_html=True)

    new_mode = st.radio("", ["Ask", "Review"], index=(0 if st.session_state.smartdocs_mode == "Ask" else 1), horizontal=True, key="mode_switch_main")
if new_mode != st.session_state.smartdocs_mode:
    st.session_state.smartdocs_mode = new_mode
    st.rerun()

# Import SmartReview primitives lazily to avoid circular imports at module import time
from SmartReview import (
    ProposedValidation,
    Rule as SRRule,
    ValidationTemplate,
    DocumentChunk,
    propose_validation_from_rule,
    execute_validation_template,
)

# Helper: build SmartReview DocumentChunk list from preprocessed_data entry
def _build_document_chunks(pre_doc):
    chunks = pre_doc.get("chunks", []) or []
    doc_chunks = []
    for c in chunks:
        text = c.get("text", "")
        if not text or not isinstance(text, str):
            continue
        # Convert to 1-based page index for SmartReview display-only purposes
        page_1_based = (c.get("page_num", 0) or 0) + 1
        doc_chunks.append(DocumentChunk(content=text, page_num=page_1_based))
    return doc_chunks

# Helper: try to extract a concise phrase from a validation finding to verify/highlight in PDF
import re as _re

def _extract_phrase(finding: str, context: str) -> str:
    if not isinstance(finding, str):
        finding = str(finding)
    # Prefer quoted text inside the finding, e.g., Violation: "..."
    m = _re.search(r'"([^"]{8,200})"', finding)
    if m:
        return m.group(1).strip()
    m = _re.search(r"'([^']{8,200})'", finding)
    if m:
        return m.group(1).strip()
    # Fallback: use context snippet without ellipses
    if isinstance(context, str) and context.strip():
        cleaned = context.replace("...", " ").strip()
        return cleaned[:200]
    # Last resort: truncate finding itself
    return finding[:200]

# Background runner: build template from rules and execute, then verify and annotate using SmartDocs PDF pipeline
import base64 as _b64
import json as _json
from src.keyword_code.processors.pdf_processor import PDFProcessor as _PDFProcessor

def run_auto_review_update():
    try:
        if st.session_state.smartdocs_mode != "Review":
            return
        rules_text = (st.session_state.get("user_prompt", "") or "").strip()
        if not rules_text:
            return
        pre = st.session_state.get("preprocessed_data", {}) or {}
        if not pre:
            return
        # Debounce: only re-run if rules changed
        last_rules = st.session_state.get("_last_review_rules_text")
        if last_rules == rules_text and st.session_state.get("analysis_results"):
            return
        st.session_state._last_review_rules_text = rules_text

        rule_lines = [ln.strip(" •-\t") for ln in rules_text.splitlines() if ln.strip()]
        if not rule_lines:
            return

        aggregated_results = []
        for filename, pre_doc in pre.items():
            # Build document chunks for AI proposal context
            doc_chunks = _build_document_chunks(pre_doc)
            # Build rules via AI proposals
            rules_final = []
            for rl in rule_lines:
                try:
                    pv = run_async(propose_validation_from_rule(rl, "", doc_chunks))
                    if pv:
                        rules_final.append(SRRule(description=rl, validation_type=pv.validation_type, validator=pv.validator))
                except Exception:
                    # Skip rule on failure; continue others
                    pass
            if not rules_final:
                continue
            template = ValidationTemplate(name=f"Auto Review - {filename}", rules=rules_final)
            # Execute validation
            try:
                validation_results = run_async(execute_validation_template(template, doc_chunks)) or []
            except Exception:
                validation_results = []

            # Build phrases for verification/highlighting
            phrases = []
            for vr in validation_results:
                try:
                    phrase = _extract_phrase(getattr(vr, "finding", ""), getattr(vr, "context", ""))
                    if phrase and phrase not in phrases:
                        phrases.append(phrase)
                except Exception:
                    continue

            # Build structured analysis by sub-prompt (one section per rule), to match Ask mode UI
            # Group validation results by their originating rule description
            grouped_by_rule = {}
            for vr in validation_results:
                try:
                    key = getattr(vr, "rule_description", "Rule") or "Rule"
                except Exception:
                    key = "Rule"
                grouped_by_rule.setdefault(key, []).append(vr)

            # Create analysis sections mirroring Ask mode's `section_{i}_{slug}` keys
            analysis_sections = {}
            sub_prompt_results = []

            # Use rule descriptions directly as titles (no RAG decomposition needed in Review mode)
            # This avoids unnecessary LLM calls and keeps Review mode fast and focused on validation
            for idx, (rule_desc, group_items) in enumerate(grouped_by_rule.items(), start=1):
                # Use rule description directly as title
                display_title = str(rule_desc)
                # Derive a slug from the rule description for the section key base
                slug_base = _re.sub(r"[^a-z0-9]+", "_", display_title.lower()).strip("_") or f"rule_{idx}"

                # Create one analysis section per violation (finding)
                for gidx, vr in enumerate(group_items, start=1):
                    try:
                        phrase = _extract_phrase(getattr(vr, "finding", ""), getattr(vr, "context", ""))
                    except Exception:
                        phrase = None

                    vtype = (getattr(vr, "violation_type", None) or "violation").upper()
                    page = getattr(vr, "page_num", "N/A")
                    analysis_text = (
                        getattr(vr, "analysis", None)
                        or f"[{vtype}] {getattr(vr, 'finding', '')}"
                    )
                    context_text = f"[{vtype}] Page {page} — From rule: {rule_desc}"

                    section_key = f"section_{idx}_{slug_base}_p{page}_{gidx}"
                    analysis_sections[section_key] = {
                        "Analysis": analysis_text,
                        "Supporting_Phrases": ([phrase] if phrase else ["No relevant phrase found."]),
                        "Context": context_text,
                    }

                # Provide sub-prompt metadata for downstream features (e.g., retry)
                sub_prompt_results.append({
                    "title": display_title,
                    "sub_prompt": rule_desc,
                })

            # If, for any reason, no results were produced, still create a placeholder section
            if not analysis_sections:
                analysis_sections = {
                    "section_1_validation_review": {
                        "Analysis": f"Auto review found {len(validation_results)} potential issue(s) across {len(rule_lines)} rule(s).",
                        "Supporting_Phrases": phrases or ["No relevant phrase found."],
                        "Context": "Auto Review",
                    }
                }

            aggregated = {
                "title": f"Validation of {filename}",
                "analysis_sections": analysis_sections,
            }
            aggregated_str = _json.dumps(aggregated, indent=2)

            # Verify and annotate using existing SmartDocs PDF pipeline
            orig_bytes = pre_doc.get("original_bytes")
            if not orig_bytes:
                continue
            processor = _PDFProcessor(orig_bytes)
            try:
                verification_results, phrase_locations = processor.verify_and_locate_phrases(aggregated_str)
            except Exception:
                verification_results, phrase_locations = {}, {}
            try:
                annotated_pdf_bytes = processor.add_annotations(phrase_locations)
                annotated_pdf_b64 = _b64.b64encode(annotated_pdf_bytes).decode("utf-8")
            except Exception:
                annotated_pdf_b64 = _b64.b64encode(orig_bytes).decode("utf-8")

            aggregated_results.append({
                "filename": filename,
                "annotated_pdf": annotated_pdf_b64,
                "verification_results": verification_results,
                "phrase_locations": phrase_locations,
                "ai_analysis": aggregated_str,
                "validation_results": [vr.model_dump() if hasattr(vr, "model_dump") else getattr(vr, "__dict__", vr) for vr in validation_results],
                "sub_prompt_results": sub_prompt_results,
            })

        if aggregated_results:
            st.session_state.analysis_results = aggregated_results
            st.session_state.results_just_generated = True
    except Exception as _e:
        # Non-fatal; keep UI responsive
        import logging as _lg
        _lg.getLogger(__name__).error(f"Auto review update failed: {_e}", exc_info=True)

# --- Configuration ---
# Setup logging (consider moving to a central config if used elsewhere)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants (Consider moving to a config file or defining in app.py) ---
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "true").lower() == "true"

# --- Load Embedding Model ---
# Ensure the embedding model is loaded early
embedding_model = load_embedding_model()

# --- Sidebar ---
with st.sidebar:
    st.write("Powered by CNT")

# --- Initialize Session State ---
initialize_session_state()



# --- Main Page Logic ---

def process_rag_requests(results: list[dict[str, any]]) -> tuple[list[dict[str, any]], bool]:
    """
    Process RAG analysis and retry requests from the UI.

    Args:
        results: List of analysis results

    Returns:
        A tuple containing:
        - The updated results list.
        - A boolean indicating if any requests were processed.
    """
    requests_processed = False
    try:
        # Import here to avoid circular imports
        from src.keyword_code.agents.rag_agent import RAGOptimizationAgent, RAGRetryTool, RAGContext
        from src.keyword_code.ai.databricks_llm import DatabricksLLMClient

        # NOTE: Legacy 'Analyze' requests removed per new agent/tool design.
        # Keeping a no-op block here to avoid key errors if older sessions have this state.
        if hasattr(st.session_state, 'rag_analysis_requests'):
            for section_key, request_data in list(st.session_state.rag_analysis_requests.items()):
                if request_data.get("status") == "requested":
                    requests_processed = True
                    st.session_state.rag_analysis_requests[section_key]["status"] = "skipped"
                    logger.info(f"RAG analysis request skipped for section {section_key} (removed in agent/tool design)")

        # Process section-specific fact extraction requests
        if hasattr(st.session_state, 'section_facts_requests'):
            for section_facts_key, request_data in st.session_state.section_facts_requests.items():
                if request_data.get("status") == "requested":
                    requests_processed = True
                    try:
                        # Extract facts from the analysis text for this specific section
                        analysis_text = request_data.get("analysis_text", "")
                        section_key = request_data.get("section_key", "")

                        if analysis_text:
                            # Import fact extraction service
                            from src.keyword_code.services.fact_extraction_service import FactExtractionService

                            # Create fact extraction service
                            fact_service = FactExtractionService()

                            # Extract facts from this specific analysis text
                            extracted_facts = fact_service.extract_facts_from_text(
                                text=analysis_text,
                                context=f"Legal/Financial Analysis - Section: {section_key}",
                                section_name=section_key,
                                filename=result.get("filename", "Unknown")
                            )

                            # Store results
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

                            logger.info(f"Section facts extraction completed for {section_facts_key}")

                        else:
                            # No analysis text available
                            st.session_state.section_facts[section_facts_key] = {
                                "status": "error",
                                "error": "No analysis text available"
                            }

                        # Mark request as completed
                        st.session_state.section_facts_requests[section_facts_key]["status"] = "completed"

                    except Exception as e:
                        logger.error(f"Error in section facts extraction for {section_facts_key}: {e}", exc_info=True)

                        # Store error state
                        if "section_facts" not in st.session_state:
                            st.session_state.section_facts = {}

                        st.session_state.section_facts[section_facts_key] = {
                            "status": "error",
                            "error": str(e)
                        }

                        st.session_state.section_facts_requests[section_facts_key]["status"] = "error"

        # Process retry requests using the Pydantic-AI agent/tool system
        if hasattr(st.session_state, 'rag_retry_requests'):
            from src.keyword_code.rag.retrieval import retrieve_relevant_chunks_async
            from src.keyword_code.models.embedding import load_reranker_model

            for section_key, request_data in st.session_state.rag_retry_requests.items():
                if request_data.get("status") == "requested":
                    requests_processed = True
                    try:
                        # Ensure results store exists
                        if "rag_retry_results" not in st.session_state:
                            st.session_state.rag_retry_results = {}

                        # Create Databricks client and optimization agent/tool
                        databricks_client = DatabricksLLMClient()
                        optimization_agent = RAGOptimizationAgent(databricks_client)
                        retry_tool = RAGRetryTool(optimization_agent)

                        # Derive context from current section/file and preprocessed data
                        section_data = request_data.get("section_data", {})
                        result_meta = request_data.get("result", {})
                        filename = result_meta.get("filename")
                        preprocessed_doc = st.session_state.get("preprocessed_data", {}).get(filename, {})

                        chunks = preprocessed_doc.get("chunks", [])
                        precomputed_embeddings = preprocessed_doc.get("chunk_embeddings")
                        valid_chunk_indices = preprocessed_doc.get("valid_chunk_indices")

                        # Use page-level analysis text as a proxy query; fallback to section key
                        base_query = section_data.get("Analysis") or section_key.replace('_', ' ')
                        query = (base_query or "").strip()[:256]

                        # Models
                        emb_model = embedding_model  # loaded at top of this page
                        reranker_model = load_reranker_model()

                        # Compute a baseline set of current results to inform the agent
                        current_results = run_async(
                            retrieve_relevant_chunks_async(
                                prompt=query,
                                chunks=chunks,
                                model=emb_model,
                                top_k=10,
                                precomputed_embeddings=precomputed_embeddings,
                                valid_chunk_indices=valid_chunk_indices,
                                reranker_model=reranker_model,
                                bm25_weight=0.5,
                                semantic_weight=0.5,
                            )
                        )

                        # Build RAG context
                        context = RAGContext(
                            query=query,
                            chunks=chunks,
                            embedding_model=emb_model,
                            reranker_model=reranker_model,
                            precomputed_embeddings=precomputed_embeddings,
                            valid_chunk_indices=valid_chunk_indices,
                            current_results=current_results,
                            bm25_weight=0.5,
                            semantic_weight=0.5,
                            top_k=10,
                        )

                        # Run retry with optimization
                        new_results, analysis = run_async(retry_tool.retry_with_optimization(context))

                        # Persist base retrieval results for UI consumption
                        st.session_state.rag_retry_results[section_key] = {
                            "new_results": new_results,
                            "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else getattr(analysis, "__dict__", analysis),
                            "original_results": current_results,
                            "filename": filename,
                        }

                        # --- Finish the pipeline: send RAG results to LLM, validate, and fact extract ---
                        try:
                            from src.keyword_code.ai.analyzer import DocumentAnalyzer
                            from src.keyword_code.processors.pdf_processor import PDFProcessor
                            from src.keyword_code.services.fact_extraction_service import FactExtractionService
                            import re

                            # Derive sub-prompt/title for this section from existing results
                            sub_prompt_results = (result_meta or {}).get("sub_prompt_results", [])

                            # Extract index from section_key like "section_3_loan_currency"
                            idx_match = re.match(r"section_(\d+)_", str(section_key))
                            sp_index = int(idx_match.group(1)) if idx_match else None

                            # Default fallbacks
                            derived_title = section_key.replace("section_", "").replace("_", " ")
                            derived_sub_prompt = section_data.get("Context", "").replace("From sub-prompt: ", "").strip() or base_query

                            if isinstance(sp_index, int) and 1 <= sp_index <= len(sub_prompt_results):
                                sp_entry = sub_prompt_results[sp_index - 1]
                                derived_title = sp_entry.get("title", derived_title)
                                derived_sub_prompt = sp_entry.get("sub_prompt", derived_sub_prompt)

                            # Prepare single sub-prompt with retried relevant chunks
                            sub_prompts_with_contexts = [{
                                "title": derived_title,
                                "sub_prompt": derived_sub_prompt,
                                "relevant_chunks": new_results or [],
                            }]

                            analyzer = DocumentAnalyzer()
                            main_prompt = st.session_state.get("user_prompt", "")

                            analyzed_list = run_async(
                                analyzer.analyze_document_with_all_contexts(
                                    filename=filename,
                                    main_prompt=main_prompt,
                                    sub_prompts_with_contexts=sub_prompts_with_contexts,
                                )
                            )

                            # Expect one analysis result
                            ai_section = None
                            if analyzed_list and isinstance(analyzed_list, list):
                                try:
                                    analysis_json_str = analyzed_list[0].get("analysis_json", "{}")
                                    parsed = json.loads(analysis_json_str)
                                    analysis_text = parsed.get("analysis_summary", "")
                                    supporting_quotes = parsed.get("supporting_quotes", [])
                                    if not isinstance(supporting_quotes, list):
                                        supporting_quotes = [str(supporting_quotes)]
                                    ai_section = {
                                        "Analysis": analysis_text,
                                        "Supporting_Phrases": supporting_quotes or ["No relevant phrase found."],
                                        "Context": f"From retried sub-prompt: {derived_sub_prompt}",
                                    }
                                except Exception as parse_err:
                                    logger.error(f"Failed to parse retried AI analysis for {section_key}: {parse_err}")

                            # Validate/verify supporting phrases against the PDF
                            verification_results, phrase_locations = {}, {}
                            try:
                                if ai_section:
                                    pdf_bytes = (st.session_state.get("preprocessed_data", {})
                                                 .get(filename, {})
                                                 .get("original_bytes"))
                                    if pdf_bytes:
                                        processor = PDFProcessor(pdf_bytes)
                                        mini_aggregated = {
                                            "title": f"Analysis of {filename} (retry)",
                                            "analysis_sections": {section_key: ai_section},
                                        }
                                        aggregated_str = json.dumps(mini_aggregated, indent=2)
                                        verification_results, phrase_locations = processor.verify_and_locate_phrases(aggregated_str)
                                    else:
                                        logger.warning("Original PDF bytes not found; skipping verification for retried section.")
                            except Exception as ver_err:
                                logger.error(f"Verification error for retried section {section_key}: {ver_err}")

                            # Fact extraction is now only performed on-demand when user clicks "Generate Facts"
                            # This improves performance by not running extraction during RAG retry
                            extracted_facts = None

                            # Save augmented results under rag_retry_results for UI rendering
                            st.session_state.rag_retry_results[section_key].update({
                                "ai_section": ai_section,
                                "verification_results": verification_results,
                                "phrase_locations": phrase_locations,
                                "extracted_facts": extracted_facts,
                                "sub_prompt": derived_sub_prompt,
                                "title": derived_title,
                            })

                        except Exception as pipeline_err:
                            logger.error(f"Post-retry pipeline failed for section {section_key}: {pipeline_err}", exc_info=True)

                        # Integrate retried analysis into the main results seamlessly
                        try:
                            target_result = None
                            for res_item in results:
                                if isinstance(res_item, dict) and res_item.get("filename") == filename:
                                    target_result = res_item
                                    break

                            if target_result and st.session_state.get("preprocessed_data", {}).get(filename):
                                # Update aggregated analysis JSON
                                try:
                                    agg_str = target_result.get("ai_analysis", "{}")
                                    agg = json.loads(agg_str) if isinstance(agg_str, str) else (agg_str or {})
                                except Exception:
                                    agg = {"title": f"Analysis of {filename}", "analysis_sections": {}}
                                if "analysis_sections" not in agg:
                                    agg["analysis_sections"] = {}
                                if st.session_state.rag_retry_results.get(section_key, {}).get("ai_section"):
                                    agg["analysis_sections"][section_key] = st.session_state.rag_retry_results[section_key]["ai_section"]
                                target_result["ai_analysis"] = json.dumps(agg, indent=2)

                                # Merge verification and locations
                                existing_ver = target_result.get("verification_results", {}) or {}
                                new_ver = st.session_state.rag_retry_results.get(section_key, {}).get("verification_results", {}) or {}
                                existing_ver.update(new_ver)
                                target_result["verification_results"] = existing_ver

                                existing_locs = target_result.get("phrase_locations", {}) or {}
                                new_locs = st.session_state.rag_retry_results.get(section_key, {}).get("phrase_locations", {}) or {}
                                for k, v in new_locs.items():
                                    existing_locs[k] = v
                                target_result["phrase_locations"] = existing_locs

                                # Regenerate annotated PDF based on merged locations (optional but improves UX)
                                try:
                                    orig_bytes = st.session_state.get("preprocessed_data", {}).get(filename, {}).get("original_bytes")
                                    if orig_bytes:
                                        processor_for_update = PDFProcessor(orig_bytes)
                                        updated_pdf_bytes = processor_for_update.add_annotations(existing_locs)
                                        target_result["annotated_pdf"] = base64.b64encode(updated_pdf_bytes).decode("utf-8")
                                except Exception as ann_err:
                                    logger.warning(f"Could not regenerate annotated PDF for {filename}: {ann_err}")

                                # Persist back to session for consumers relying on session state list
                                if isinstance(st.session_state.get("analysis_results"), list):
                                    for idx, sess_item in enumerate(st.session_state.analysis_results):
                                        if isinstance(sess_item, dict) and sess_item.get("filename") == filename:
                                            st.session_state.analysis_results[idx] = target_result
                                            break

                                # Also persist section facts (if available) into the per-section store used by UI
                                sect_key = f"section_facts_{section_key}_{filename}"
                                facts_payload = st.session_state.rag_retry_results.get(section_key, {}).get("extracted_facts")
                                if isinstance(facts_payload, dict):
                                    if "section_facts" not in st.session_state:
                                        st.session_state.section_facts = {}
                                    st.session_state.section_facts[sect_key] = {
                                        "status": "completed",
                                        "facts": facts_payload.get("extracted_facts", []),
                                        "metadata": facts_payload.get("extraction_metadata", {}),
                                    }

                            # Mark as completed BEFORE any UI refresh to avoid loops
                            st.session_state.rag_retry_requests[section_key]["status"] = "completed"
                            logger.info(f"RAG retry completed for section {section_key} with {len(new_results)} results (pipeline finalized)")
                            # No st.rerun() here — updated state should render within this pass
                        except Exception as integrate_err:
                            logger.error(f"Failed to integrate retried results into main view: {integrate_err}", exc_info=True)

                    except Exception as e:
                        logger.error(f"Error in RAG retry for section {section_key}: {e}", exc_info=True)
                        st.session_state.rag_retry_requests[section_key]["status"] = "error"

    except Exception as e:
        logger.error(f"Error processing RAG requests: {e}", exc_info=True)

    return results, requests_processed


# Check if results exist to determine view
if st.session_state.get("analysis_results"):
    # --- RESULTS VIEW ---
    st.markdown(
        """
        <div class="smartdocs-logo-container">
            <h1><span style='color: #002345;'>CNT</span> <span style='color: #00ADE4;'>SmartDocs</span><sup style='font-size: 1rem; color: #FF5733;'>BETA</sup></h1>
            <p>AI Powered Document Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Start New Analysis", key="new_analysis_button", use_container_width=True, type="primary"):
        # Clear relevant session state variables
        keys_to_clear = [
            "analysis_results", "pdf_bytes", "show_pdf",
            "current_pdf_name", "chat_messages", "followup_qa", "results_just_generated",
            "user_prompt", "uploaded_file_objects", "last_uploaded_filenames",
            "preprocessed_data", "preprocessing_status"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        logger.info("Cleared state for new analysis.")
        st.rerun()

    st.divider()
    # Add an anchor for scrolling
    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

    results_to_display = st.session_state.get("analysis_results", [])
    errors = [r for r in results_to_display if isinstance(r, dict) and "error" in r]
    success_results = [r for r in results_to_display if isinstance(r, dict) and "error" not in r]

    # Status Summary
    total_processed = len(results_to_display)
    if errors:
        if not success_results:
            st.error(f"Processing failed for all {total_processed} file(s). See details below.")
        else:
            st.warning(f"Processing complete for {total_processed} file(s). {len(success_results)} succeeded, {len(errors)} failed.")

    # Error Details Expander
    if errors:
        with st.expander("⚠️ Processing Errors", expanded=True):
            for error_res in errors:
                st.error(f"**{error_res.get('filename', 'Unknown File')}**: {error_res.get('error', 'Unknown error details.')}")

    # Display Successful Analysis using the function from app.py
    if success_results:
        # Process RAG requests before displaying results
        success_results, requests_processed = process_rag_requests(success_results)

        # display_analysis_results expects a list of result dictionaries
        display_analysis_results(success_results)
    elif not errors:
        st.warning("Processing finished, but no primary analysis content was generated.")

    # Auto-scroll logic
    if st.session_state.get("results_just_generated", False):
        js = """
        <script>
            setTimeout(function() {
                const anchor = document.getElementById('results-anchor');
                if (anchor) {
                    console.log("Scrolling to results anchor...");
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    console.log("Results anchor not found for scrolling.");
                }
            }, 150); // Increased delay slightly
        </script>
        """
        st.components.v1.html(js, height=0)
        st.session_state.results_just_generated = False # Reset flag

else:
    # --- INPUT VIEW ---
    st.markdown(
        """
        <div class="smartdocs-logo-container">
            <h1><span style='color: #002345;'>CNT</span> <span style='color: #00ADE4;'>SmartDocs</span><sup style='font-size: 1rem; color: #FF5733;'>BETA</sup></h1>
            <p>AI Powered Document Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if embedding model loaded successfully
    if embedding_model is None:
        st.error(
            "Embedding model failed to load. Document processing is disabled. "
            "Please check logs and ensure dependencies are installed correctly."
        )
        # Use return to stop rendering the rest of the page if model is critical
        st.stop()

    # File Upload Callback
    def handle_file_change():
        # This function runs when the file uploader widget state changes
        current_files = st.session_state.get("file_uploader_main", []) # Use the key assigned below
        st.session_state.current_file_objects_from_change = current_files
        st.session_state.file_selection_changed_by_user = True
        logger.debug(f"handle_file_change triggered: Stored {len(current_files) if current_files else 0} files. Flag set.")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader_main", # Changed key to avoid conflicts if app.py also runs
        on_change=handle_file_change,
    )

    preprocessing_or_features_container = st.empty()

    # File Change Logic
    if st.session_state.file_selection_changed_by_user:
        logger.debug("Processing detected file change from user action.")
        st.session_state.file_selection_changed_by_user = False
        current_files = st.session_state.current_file_objects_from_change
        # Ensure current_files is a list, handle None case
        if current_files is None:
            current_files = []

        current_uploaded_filenames = set(f.name for f in current_files)
        last_filenames = st.session_state.get('last_uploaded_filenames', set())

        if current_uploaded_filenames != last_filenames:
            logger.info(f"Actual file change confirmed: New={current_uploaded_filenames - last_filenames}, Removed={last_filenames - current_uploaded_filenames}")
            new_files = current_uploaded_filenames - last_filenames
            removed_files = last_filenames - current_uploaded_filenames
            st.session_state.uploaded_file_objects = current_files # Store the actual file objects
            st.session_state.last_uploaded_filenames = current_uploaded_filenames

            # Remove data for removed files
            for removed_file in removed_files:
                if removed_file in st.session_state.preprocessed_data:
                    del st.session_state.preprocessed_data[removed_file]
                    if removed_file in st.session_state.preprocessing_status:
                        del st.session_state.preprocessing_status[removed_file]
                    logger.info(f"Removed preprocessing data for {removed_file}")

            # Clear results if files change
            if new_files or removed_files:
                 st.session_state.analysis_results = []
                 st.session_state.chat_messages = []
                 st.session_state.followup_qa = []
                 st.session_state.pdf_bytes = None # Clear PDF view
                 st.session_state.current_pdf_name = None
                 st.session_state.show_pdf = False
                 logger.info("Cleared relevant state due to file change.")

            # Preprocess new files
            if new_files:
                with preprocessing_or_features_container.container():
                    # Use st.status for better progress indication
                    with st.status(f"Preprocessing {len(new_files)} new document(s)...", expanded=True) as status:
                        preprocessing_failed = False
                        success_count = 0
                        files_processed_list = sorted(list(new_files))

                        for i, filename in enumerate(files_processed_list):
                            # Find the corresponding file object
                            file_obj = next((f for f in current_files if f.name == filename), None)
                            if file_obj:
                                try:
                                    status.update(label=f"Preprocessing file {i+1}/{len(new_files)}: {filename}")
                                    st.write(f"Processing {filename}...") # Show progress within status

                                    # Ensure file pointer is at the beginning if reading multiple times
                                    file_obj.seek(0)
                                    file_data = file_obj.getvalue()

                                    # Call the preprocessing function (imported from app.py)
                                    result = preprocess_file(
                                        file_data,
                                        filename,
                                    )
                                    st.session_state.preprocessing_status[filename] = result
                                    logger.info(f"Preprocessed {filename}: {result['status']}")

                                    if result['status'] == 'success':
                                        success_count += 1
                                        st.write(f"✅ {filename} processed successfully.")
                                    elif result['status'] == 'warning':
                                        st.write(f"⚠️ {filename}: {result['message']}")
                                        preprocessing_failed = True # Treat warning as needing attention
                                    else: # Error
                                        st.write(f"❌ Error processing {filename}: {result['message']}")
                                        preprocessing_failed = True

                                except Exception as e:
                                    logger.error(f"Critical error during preprocessing loop for {filename}: {str(e)}", exc_info=True)
                                    st.session_state.preprocessing_status[filename] = {"status": "error", "message": f"Critical error: {str(e)}"}
                                    st.write(f"❌ Critical Error processing {filename}: {str(e)}")
                                    preprocessing_failed = True
                            else:
                                logger.error(f"Could not find file object for {filename} during preprocessing.")
                                st.write(f"❌ Error: Could not find file object for {filename}.")
                                preprocessing_failed = True


                        # Final status update
                        if preprocessing_failed:
                            final_label = f"Preprocessing issues. {success_count}/{len(new_files)} processed. Check messages."
                            final_state = "warning" if success_count > 0 else "error"
                        else:
                            final_label = f"Preprocessing complete! {success_count}/{len(new_files)} processed."
                            final_state = "complete"
                        status.update(label=final_label, state=final_state, expanded=False)

            else:
                logger.debug("File change flag was True, but filename sets match or no new files. Ignoring spurious flag/no action needed.")

            # Clear the temporary variable
            st.session_state.current_file_objects_from_change = None
            # Rerun to reflect preprocessing status and potentially hide welcome message
            st.rerun()

    # Welcome Features Section - Ask vs Review
    if not st.session_state.get("preprocessed_data") and not st.session_state.get("file_selection_changed_by_user"):
        with preprocessing_or_features_container.container():
            if st.session_state.smartdocs_mode == "Review":
                display_review_features()
            else:
                display_welcome_features()

    # Analysis Inputs - Only show if files are uploaded and preprocessed
    if st.session_state.get("preprocessed_data"):
        with st.container(border=False):  # Changed border to False
            # --- Saved Prompts (from configuration, per mode) ---
            mode_key = "Ask" if st.session_state.smartdocs_mode == "Ask" else "Review"
            configured = SAVED_PROMPTS.get(mode_key, {})
            # Flatten categories to a single list of suggestions
            prompt_suggestions = []
            for _cat_name, _items in configured.items():
                for _it in _items:
                    entry = {
                        "label": _it.get("label", "Unnamed"),
                        "prompt": _it.get("prompt", ""),
                        "explanation": _it.get("explanation"),
                        "category": _cat_name,
                    }
                    prompt_suggestions.append(entry)

            # In Review mode, only show the single merged checklist prompt to avoid confusion
            if mode_key == "Review":
                prompt_suggestions = [
                    s for s in prompt_suggestions
                    if s.get("label") == "Comprehensive Financial Statement Validation"
                ]

            suggestion_labels = [s["label"] for s in prompt_suggestions]
            suggestion_prompts = [s["prompt"] for s in prompt_suggestions]
            explanations_map = {s["label"]: (s.get("explanation"), s.get("category")) for s in prompt_suggestions}

            selected_pill = stp.pills(
                "Saved Prompts:",
                suggestion_labels,
                clearable=True,
                index=None,  # No default selection
                label_visibility="visible"
            )

            if selected_pill:
                try:
                    idx = suggestion_labels.index(selected_pill)
                    st.session_state["user_prompt"] = suggestion_prompts[idx]
                    exp_cat = explanations_map.get(selected_pill)
                    if exp_cat and exp_cat[0]:
                        st.caption(f"{exp_cat[1]} • {exp_cat[0]}")
                except ValueError:
                    logger.warning(f"Selected pill '{selected_pill}' not found in suggestion labels.")

            label = "Analysis Prompt" if st.session_state.smartdocs_mode == "Ask" else "Validation Rules"
            st.session_state.user_prompt = st.text_area(
                label,
                placeholder="Ask: enter your analysis instructions. Review: type validation checks (one per line); they run automatically.",
                height=150,
                key="prompt_input_main",
                value=st.session_state.get("user_prompt", ""),
            )


        # Process Button
        # Disable if model not loaded, no files uploaded, no prompt, or if files haven't finished preprocessing
        files_available = st.session_state.get('uploaded_file_objects')
        prompt_entered = st.session_state.get('user_prompt', '').strip()
        # Check if all uploaded files have a status
        all_preprocessed = all(
            fname in st.session_state.get("preprocessing_status", {})
            for fname in st.session_state.get("last_uploaded_filenames", set())
        ) if files_available else False

        process_button_disabled = not (embedding_model and files_available and prompt_entered and all_preprocessed)

        if st.button("Process Documents", type="primary", use_container_width=True, disabled=process_button_disabled):
            files_to_process_objs = st.session_state.get("uploaded_file_objects", [])
            current_user_prompt = st.session_state.get("user_prompt", "")

            if not files_to_process_objs: st.warning("Please upload one or more documents.")
            elif not current_user_prompt.strip(): st.error(f"Please enter {'Validation Rules' if st.session_state.smartdocs_mode == 'Review' else 'an Analysis Prompt'}.")
            elif not all_preprocessed: st.error("Please wait for file preprocessing to complete.")
            else:
                # --- Processing Logic ---
                st.session_state.analysis_results = [] # Clear previous results
                st.session_state.show_pdf = False
                st.session_state.pdf_bytes = None
                st.session_state.current_pdf_name = None

                total_files = len(files_to_process_objs)
                overall_start_time = datetime.now()
                # Use a dictionary for results placeholder for easier mapping back
                # If in Review mode, run the SmartReview pipeline now and then rerun to display results all at once
                if st.session_state.smartdocs_mode == "Review":
                    with st.spinner("Running review...", show_time=True):
                        run_auto_review_update()
                    st.rerun()

                results_placeholder = {} # Use filename as key
                file_map = {f.name: f for f in files_to_process_objs} # Map name to object

                process_args_list = []
                files_read_ok = True
                # Determine mode for process_file_wrapper: 'ask' or 'review'
                current_mode = st.session_state.smartdocs_mode.lower()  # 'ask' or 'review'
                logger.info(f"Preparing to process {len(files_to_process_objs)} files in '{current_mode}' mode")

                for uploaded_file in files_to_process_objs:
                    try:
                        # Ensure pointer is at start before reading
                        uploaded_file.seek(0)
                        file_data = uploaded_file.getvalue()
                        filename = uploaded_file.name
                        # Get preprocessed data for this file (should exist)
                        preprocessed_file_data = st.session_state.get("preprocessed_data", {}).get(filename)
                        if not preprocessed_file_data:
                             logger.error(f"Missing preprocessed data for {filename} during process button click.")
                             raise ValueError(f"Preprocessing data missing for {filename}.")

                        # Add mode parameter to process_args
                        process_args_list.append(
                            (file_data, filename, current_user_prompt, False, preprocessed_file_data, current_mode)
                        )
                        results_placeholder[filename] = None # Initialize placeholder
                    except Exception as read_err:
                        logger.error(f"Failed to read/prepare file {uploaded_file.name} for processing: {read_err}", exc_info=True)
                        results_placeholder[uploaded_file.name] = {"filename": uploaded_file.name, "error": f"Failed to prepare file: {read_err}"}
                        files_read_ok = False

                if files_read_ok and process_args_list:
                    files_to_run_count = len(process_args_list)

                    with st.expander("Document Processing Status", expanded=True):
                        status_updates_container = st.container() # Container for updates
                        processed_files_count = 0

                        def update_status_display():
                            # Simple text update for now
                            status_updates_container.info(f"Processing... {processed_files_count}/{files_to_run_count} files complete.")

                        update_status_display() # Initial status

                        with st.spinner("Analyzing documents...", show_time=True): # Added spinner here
                            # Use the wrapper function directly
                            def run_task_wrapper(args_tuple):
                                filename = args_tuple[1]
                                logger.info(f"Thread {threading.current_thread().name} starting task for {filename}")
                                try:
                                    result = process_file_wrapper(args_tuple) # Call the imported function
                                    logger.info(f"Thread {threading.current_thread().name} finished task for {filename}")
                                    return filename, result
                                except Exception as thread_err:
                                    logger.error(f"Unhandled error in thread task for {filename}: {thread_err}", exc_info=True)
                                    return filename, {"filename": filename, "error": f"Unhandled thread error: {thread_err}"}

                            try:
                                futures = []
                                # Decide parallel or sequential
                                use_parallel = ENABLE_PARALLEL and files_to_run_count > 1
                                executor = None
                                if use_parallel:
                                    logger.info(f"Executing {files_to_run_count} tasks in parallel with max workers: {MAX_WORKERS}")
                                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
                                    futures = [executor.submit(run_task_wrapper, args) for args in process_args_list]
                                else:
                                    logger.info(f"Executing {files_to_run_count} tasks sequentially.")
                                    # Run sequentially and store results directly
                                    for args in process_args_list:
                                        fname, result_data = run_task_wrapper(args)
                                        results_placeholder[fname] = result_data
                                        processed_files_count += 1
                                        update_status_display()


                                # Collect results if parallel
                                if use_parallel and executor:
                                    for future in concurrent.futures.as_completed(futures):
                                        try:
                                            fname, result_data = future.result()
                                            results_placeholder[fname] = result_data
                                        except Exception as exc:
                                            # Attempt to find which filename this future was for (can be tricky)
                                            logger.error(f'A parallel task failed: {exc}', exc_info=True)
                                        finally:
                                            processed_files_count += 1
                                            update_status_display()
                                    executor.shutdown(wait=True) # Ensure cleanup

                            except Exception as pool_err:
                                 logger.error(f"Error during task execution management: {pool_err}", exc_info=True)
                                 status_updates_container.error(f"Error during processing: {pool_err}. Some files may not have been processed.") # Use status_updates_container
                                 # Mark remaining placeholders as errored
                                 for fname, res in results_placeholder.items():
                                      if res is None:
                                           results_placeholder[fname] = {"filename": fname, "error": f"Processing cancelled due to execution error: {pool_err}"}


                            # --- Final Result Handling ---
                            # Convert placeholder dict back to list in original order
                            final_results_list = []
                            for uploaded_file in files_to_process_objs:
                                 fname = uploaded_file.name
                                 if fname in results_placeholder:
                                     final_results_list.append(results_placeholder[fname])
                                 else:
                                     # Should not happen if placeholder initialized correctly
                                     logger.error(f"Result missing for {fname} in final processing step.")
                                     final_results_list.append({"filename": fname, "error": "Result missing after processing."})


                            st.session_state.analysis_results = final_results_list
                            total_time = (datetime.now() - overall_start_time).total_seconds()
                            success_count = len([r for r in final_results_list if isinstance(r, dict) and "error" not in r])
                            final_status_message = f"Processing complete! Processed {success_count}/{total_files} files successfully in {total_time:.2f} seconds."
                            logger.info(final_status_message)
                            # Update the status one last time
                            status_updates_container.success(final_status_message)


                            # Set initial PDF view
                            first_success = next((r for r in final_results_list if isinstance(r, dict) and "error" not in r and r.get("annotated_pdf")), None)
                            if first_success:
                                try:
                                    pdf_bytes = base64.b64decode(first_success["annotated_pdf"])
                                    update_pdf_view(pdf_bytes=pdf_bytes, page_num=1, filename=first_success.get("filename"))
                                except Exception as decode_err:
                                    logger.error(f"Failed to decode/set initial PDF: {decode_err}", exc_info=True)
                                    st.error("Failed to load initial PDF view.")
                                    st.session_state.show_pdf = False # Ensure PDF view is hidden on error
                            else:
                                logger.warning("No successful result with annotated PDF found. No initial PDF view shown.")
                                st.session_state.show_pdf = False # Ensure PDF view is hidden

                            # Set flag for scroll and rerun
                            st.session_state.results_just_generated = True
                            st.rerun()

# --- Disclaimer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em; color: #777;'>"
    "⚠️ AI generated responses might be inaccurate or incomplete, please verify responses carefully before using them in official work."
    "</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 0.9em; margin-top: 10px;'>"
    "<a href='https://worldbankgroup.sharepoint.com/sites/CNTHub/SitePages/AI-@-IFC-Controllers--Introducing-CNT-SmartDocs.aspx' target='_blank' style='color: #00ADE4; text-decoration: none;'>"
    "Learn More about AI @ CNT"
    "</a>"
    "</p>",
    unsafe_allow_html=True
)
# add baloons to the page, on click st.balloons (easter egg)
balloon_container = st.container()
with balloon_container:
    if st.button("⠀", key="balloon_button", help="Click for a surprise!", type="tertiary"):
        st.balloons()