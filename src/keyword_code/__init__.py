"""
Keyword Code RAG package.
"""

from .app import (
    # Core functions
    preprocess_file,
    process_file_wrapper,

    # Utility functions
    get_base64_encoded_image,
    run_async,

    # UI functions
    apply_ui_styling,
    render_branding,
    initialize_session_state,
    display_welcome_features,

    # Display functions
    display_analysis_results,
    display_pdf_viewer,
    update_pdf_view,

    # Model loading
    load_embedding_model
)

__all__ = [
    'preprocess_file',
    'process_file_wrapper',
    'get_base64_encoded_image',
    'run_async',
    'apply_ui_styling',
    'render_branding',
    'initialize_session_state',
    'display_welcome_features',
    'display_analysis_results',
    'display_pdf_viewer',
    'update_pdf_view',
    'load_embedding_model'
]
