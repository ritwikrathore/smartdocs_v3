"""
UI helper functions for the keyword_code package.
"""

import streamlit as st
import time
from ..config import PROCESS_CYAN, DARK_BLUE, LIGHT_BLUE_TINT, logger
from .helpers import get_base64_encoded_image
from .file_manager import (
    cleanup_session_files, get_session_id, update_session_access,
    cleanup_expired_sessions, remove_temp_file
)
import logging


def apply_ui_styling():
    """Apply CSS styling for the app UI"""
    st.markdown(f"""
    <style>
        /* Base Styling */
        .stApp {{ background-color: white; }}
        h1, h2, h3, h4, h5, h6 {{ color: {DARK_BLUE}; }}
        h1 {{ text-align: center; }}
        /* Buttons */
        .stButton > button[kind="primary"] {{ background-color: {PROCESS_CYAN}; color: white; border: 1px solid {PROCESS_CYAN}; }}
        .stButton > button[kind="primary"]:hover {{ background-color: {DARK_BLUE}; color: white; border: 1px solid {DARK_BLUE}; }}
        .stButton > button[kind="primary"]:disabled {{ background-color: #cccccc; color: #666666; border: 1px solid #cccccc; }}
        .stButton > button[kind="secondary"] {{ color: {DARK_BLUE}; border: 1px solid {DARK_BLUE}; }}
        .stButton > button[kind="secondary"]:hover {{ border-color: {PROCESS_CYAN}; color: {PROCESS_CYAN}; background-color: rgba(0, 173, 228, 0.1); }}
        .stButton > button[kind="secondary"]:disabled {{ color: #aaaaaa; border-color: #dddddd; }}
        /* Expanders */
        .stExpander > summary {{ background-color: {LIGHT_BLUE_TINT}; color: {DARK_BLUE}; border-radius: 0.25rem; border: 1px solid rgba(0, 173, 228, 0.2); }}
        .stExpander > summary:hover {{ background-color: rgba(0, 173, 228, 0.2); }}
        .stExpander > summary svg {{ fill: {DARK_BLUE}; }}
        /* Container Borders */
        .st-emotion-cache-1r6slb0, .st-emotion-cache-lrl5gf {{ border: 1px solid {LIGHT_BLUE_TINT}; }}
        /* Download Button */
        .stDownloadButton > button {{ background-color: {DARK_BLUE}; color: white; border: 1px solid {DARK_BLUE}; }}
        .stDownloadButton > button:hover {{ background-color: {PROCESS_CYAN}; color: {DARK_BLUE}; border: 1px solid {PROCESS_CYAN}; }}
        /* Text Input / Area */
        .stTextInput, .stTextArea {{ border-color: rgba(0, 35, 69, 0.2); }}
        /* Mascot Image */
        .mascot-image {{ position: fixed; bottom: 20px; right: 20px; width: 150px; z-index: 0; opacity: 0.7; pointer-events: none; }}
        /* IF Controller Logo */
        .ifcontroller-logo {{ position: absolute; top: 10px; left: 10px; z-index: 999; max-width: 450px; width: 55%; /* Responsive width based on viewport */ }}
        .ifcontroller-logo img {{
            width: 100%;
            height: auto;
            max-height: 80px;
            object-fit: contain; /* Maintains aspect ratio */
        }}
        /* SmartDocs Logo Container - Modified for text header */
        .smartdocs-logo-container {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
            /* Replace margin with direct positioning */
            position: relative;
            margin-top: 60px; /* Reduced from 120px to 70px to decrease the gap */
            padding-top: 10px; /* Reduced from 30px to 20px for better spacing */
            z-index: 100; /* Ensure it appears above other elements */
            width: 100%; /* Full width */
            clear: both; /* Ensure it doesn't float with other elements */
        }}
        .smartdocs-logo-container h1 {{
            margin: 0;
            padding: 0;
            font-size: 2.5rem;
            font-weight: bold;
        }}
        .smartdocs-logo-container p {{
            margin: 0;
            padding: 0;
            font-size: 1.2rem;
            margin-top: 0.3rem;
        }}
        /* Features Container Styling */
        .features-container {{ display: flex; flex-direction: column; gap: 0.75rem; margin: 0.5rem auto; max-width: 1000px; }} /* Reduced margin from 1rem to 0.5rem */
        .features-row {{ display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; /* Allow wrapping on smaller screens */ }}
        .feature-text {{ flex: 1 1 300px; /* Flex grow, shrink, basis */ max-width: 450px; padding: 1rem; background: #f0f8ff; border: 1px solid #e0e0e0; border-radius: 8px; font-size: .9rem; line-height: 1.4; display: flex; align-items: flex-start; gap: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .check-icon {{ width: 18px; height: 18px; object-fit: contain; margin-top: 0.15rem; flex-shrink: 0; }}
        .welcome-header {{ color: {DARK_BLUE}; font-size: 24px; text-align: center; margin-bottom: 20px; font-weight: 500; }}
        /* Analysis display styles */
        .sleek-container {{ background-color: #f5f5f5; border-radius: 8px; padding: 8px 16px; margin: 0 0 16px 0; display: flex; align-items: center; justify-content: space-between; border: 1px solid #e0e0e0; }}
        .header-title {{ font-weight: 700; font-size: 1.5rem; color: #333; margin: 0; padding: 0; }}
        .file-name {{ font-weight: 600; color: #424242; font-size: 1rem; display: flex; align-items: center; margin: 0; padding: 0; }}
        .file-icon {{ color: #1976d2; margin-right: 8px; }}
        .stButton > button {{ margin-top: 0 !important; padding-top: 2px !important; padding-bottom: 2px !important; line-height: 1.2 !important; }}
    </style>
    """, unsafe_allow_html=True)


def render_branding():
    """Render branding elements including logos and mascot"""
    try:
        mascot_base64 = get_base64_encoded_image("src/keyword_code/assets/mascot.png")
        ifcontrollers_base64 = get_base64_encoded_image("src/keyword_code/assets/ifcontrollers.png")

        st.markdown(f"""
        <!-- Fixed Mascot Image -->
        <div class="mascot-image">
            <img src="data:image/png;base64,{mascot_base64}" alt="Mascot">
        </div>
        <!-- Fixed IF Controller Logo -->
        <div class="ifcontroller-logo">
            <img src="data:image/png;base64,{ifcontrollers_base64}" alt="IF Controllers Logo">
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # Log error but continue execution
        logging.error(f"Failed to load one or more branding images: {e}")


def initialize_session_state():
    """Initialize all session state variables needed for the application"""
    # Get the current session ID and update access time
    session_id = get_session_id()
    update_session_access(session_id)

    # Clean up expired sessions
    cleanup_expired_sessions()

    # Check if this is a new session
    if "session_initialized" not in st.session_state:
        # This is a new session, clean up any files from previous sessions
        logger.info(f"Initializing new session: {session_id}")
        st.session_state.session_initialized = True
        st.session_state.session_id = session_id
        st.session_state.session_start_time = time.time()

        # Clean up any files that might be associated with this session ID
        # (in case of browser refresh or session reuse)
        cleanup_session_files(session_id)

    # Initialize standard session state variables
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = []
    if "show_pdf" not in st.session_state: st.session_state.show_pdf = False
    if "pdf_page" not in st.session_state: st.session_state.pdf_page = 1
    if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
    if "current_pdf_name" not in st.session_state: st.session_state.current_pdf_name = None
    if "user_prompt" not in st.session_state: st.session_state.user_prompt = ""
    if "last_uploaded_filenames" not in st.session_state: st.session_state.last_uploaded_filenames = set()
    if "uploaded_file_objects" not in st.session_state: st.session_state.uploaded_file_objects = []
    if "preprocessed_data" not in st.session_state: st.session_state.preprocessed_data = {}
    if "preprocessing_status" not in st.session_state: st.session_state.preprocessing_status = {}
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
    if "followup_qa_by_doc" not in st.session_state: st.session_state.followup_qa_by_doc = {}
    if "file_selection_changed_by_user" not in st.session_state: st.session_state.file_selection_changed_by_user = False
    if "current_file_objects_from_change" not in st.session_state: st.session_state.current_file_objects_from_change = None
    if "results_just_generated" not in st.session_state: st.session_state.results_just_generated = False

    # Track temporary files in session state
    if "temp_files" not in st.session_state: st.session_state.temp_files = []


def clear_incompatible_embeddings(current_embedding_model):
    """
    Clear preprocessed data if embeddings are incompatible with current model.
    This prevents dimension mismatch errors when switching between embedding models.
    """
    if "preprocessed_data" not in st.session_state:
        return

    try:
        # Test current model embedding dimension
        test_embedding = current_embedding_model.encode("test", convert_to_tensor=True)
        if hasattr(test_embedding, 'shape'):
            current_dim = test_embedding.shape[-1]
        else:
            current_dim = len(test_embedding)

        # Check each preprocessed file for dimension compatibility
        files_to_remove = []
        for filename, data in st.session_state.preprocessed_data.items():
            if "chunk_embeddings" in data:
                chunk_embeddings = data["chunk_embeddings"]
                if hasattr(chunk_embeddings, 'shape'):
                    stored_dim = chunk_embeddings.shape[-1]
                elif len(chunk_embeddings) > 0:
                    stored_dim = len(chunk_embeddings[0]) if hasattr(chunk_embeddings[0], '__len__') else len(chunk_embeddings)
                else:
                    continue

                if stored_dim != current_dim:
                    logger.warning(f"Dimension mismatch for {filename}: stored={stored_dim}, current={current_dim}. Marking for removal.")
                    files_to_remove.append(filename)

        # Remove incompatible files
        for filename in files_to_remove:
            # Clean up temporary files if they exist
            file_data = st.session_state.preprocessed_data[filename]
            if "temp_files" in file_data and isinstance(file_data["temp_files"], list):
                for temp_file in file_data["temp_files"]:
                    try:
                        remove_temp_file(temp_file)
                        logger.debug(f"Removed temporary file: {temp_file}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

            # Remove from session state
            del st.session_state.preprocessed_data[filename]
            if filename in st.session_state.get("preprocessing_status", {}):
                del st.session_state.preprocessing_status[filename]
            logger.info(f"Removed incompatible embeddings for {filename}")

        if files_to_remove:
            logger.warning(f"Cleared {len(files_to_remove)} files with incompatible embeddings. Files will be automatically reprocessed.")
            # Don't show warning to users - handle this silently

    except Exception as e:
        logger.error(f"Error checking embedding compatibility: {e}")


def clear_session_for_new_query():
    """
    Clear relevant session state variables and temporary files when starting a new query.
    This ensures we don't have memory leaks or leftover files.
    """
    # Get the current session ID
    session_id = get_session_id()
    logger.info(f"Clearing session state for new query in session: {session_id}")

    # Clean up any temporary files associated with this session
    cleanup_session_files(session_id)

    # Clear relevant session state variables
    keys_to_clear = [
        "analysis_results", "pdf_bytes", "show_pdf",
        "current_pdf_name", "chat_messages", "followup_qa_by_doc", "results_just_generated",
        "user_prompt", "uploaded_file_objects", "last_uploaded_filenames",
        "preprocessed_data", "preprocessing_status", "temp_files"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Reinitialize empty containers for tracking
    st.session_state.analysis_results = []
    st.session_state.preprocessed_data = {}
    st.session_state.preprocessing_status = {}
    st.session_state.temp_files = []
    st.session_state.followup_qa_by_doc = {}  # Reinitialize per-document follow-up Q&A

    logger.info("Session state cleared for new query")


def display_welcome_features(check_img=None):
    """Display welcome features section with check icons"""
    try:
        # If check_img is not provided, try to load the check icon from assets
        if check_img is None:
            try:
                from pathlib import Path
                check_path = Path(__file__).parent.parent / "assets" / "correct.png"
                logger.debug(f"Attempting to load check icon from: {check_path}")
                from ..utils.helpers import get_base64_encoded_image
                check_base64 = get_base64_encoded_image(str(check_path))
                if check_base64:
                    check_img = f'<img src="data:image/png;base64,{check_base64}" class="check-icon" alt="✓">'
                    logger.debug("Successfully loaded check icon")
                else:
                    check_img = "✅"  # Fallback if encoding fails
                    logger.warning("Failed to encode check icon, using emoji fallback")
            except Exception as img_e:
                logger.warning(f"Could not load check icon: {img_e}")
                check_img = "✅"  # Fallback

        # Use the check image in the features
        st.markdown(f"""
        <div class="features-container">
            <div class="features-row">
                <div class="feature-text">{check_img} Upload your documents, ask questions and get AI analysis with verified responses.</div>
                <div class="feature-text">{check_img} Get analysis results from multiple documents at once with the same prompt.</div>
            </div>
            <div class="features-row">
                <div class="feature-text">{check_img} Ask Followup questions and talk to your documents to get more details and insights.</div>
                <div class="feature-text">{check_img} Export annotated documents and query results in Excel or Word format for further analysis.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # Fall back to simple text if there's an error
        logging.error(f"Failed to display welcome features: {e}")
        st.markdown("""
        <div class="features-container">
            <div class="features-row">
                <div class="feature-text">✅ Upload your documents, ask questions and get AI analysis with verified responses.</div>
                <div class="feature-text">✅ Get analysis results from multiple documents at once with the same prompt.</div>
            </div>
            <div class="features-row">
                <div class="feature-text">✅ Ask Followup questions and talk to your documents to get more details and insights.</div>
                <div class="feature-text">✅ Export annotated documents and query results in Excel or Word format for further analysis.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)



def display_review_features(check_img=None):
    """Display welcome features section (Review mode) with check icons"""
    try:
        # If check_img is not provided, load the same icon used for Ask features
        if check_img is None:
            try:
                from pathlib import Path
                from ..utils.helpers import get_base64_encoded_image
                check_path = Path(__file__).parent.parent / "assets" / "correct.png"
                check_base64 = get_base64_encoded_image(str(check_path))
                if check_base64:
                    check_img = f'<img src="data:image/png;base64,{check_base64}" class="check-icon" alt="\u2713">'
                else:
                    check_img = "✅"
            except Exception:
                check_img = "✅"

        st.markdown(f"""
        <div class="features-container">
            <div class="features-row">
                <div class="feature-text">{check_img} Validate numbers, currencies, and dates with automated checks and suggested fixes.</div>
                <div class="feature-text">{check_img} Apply the same validation rules across multiple documents simultaneously.</div>
            </div>
            <div class="features-row">
                <div class="feature-text">{check_img} Review findings with page references and highlights directly in annotated PDFs.</div>
                <div class="feature-text">{check_img} Export validation reports and annotated PDFs/Excel for audit and sharing.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Failed to display review features: {e}")
        st.markdown("""
        <div class="features-container">
            <div class="features-row">
                <div class="feature-text">✅ Validate numbers, currencies, and dates with automated checks and suggested fixes.</div>
                <div class="feature-text">✅ Apply the same validation rules across multiple documents simultaneously.</div>
            </div>
            <div class="features-row">
                <div class="feature-text">✅ Review findings with page references and highlights directly in annotated PDFs.</div>
                <div class="feature-text">✅ Export validation reports and annotated PDFs/Excel for audit and sharing.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
