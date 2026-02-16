"""
Display utilities package for keyword_code.

This package provides modular display functions for the Streamlit application,
organized into focused modules for better maintainability.
"""

# UI Components
from .ui_components import get_base64_encoded_image, check_img

# PDF Utilities
from .pdf_utils import (
    find_annotated_pdf_for_filename,
    regenerate_annotated_pdfs_from_chat_chunks,
    update_pdf_view,
    display_pdf_viewer
)

# Citation Utilities
from .citation_utils import (
    process_chat_response_for_numbered_citations,
    display_followup_citations_like_main_analysis,
    display_chat_message_with_citations
)

# Export Utilities
from .export_utils import create_report_package_content, export_to_word

# Analysis Display
from .analysis_display import (
    display_analysis_results,
    display_rag_retry_button_header,
    display_rag_results_section,
    display_section_facts_expander
)

# Tools Column
from .tools_column import display_tools_column

__all__ = [
    # UI Components
    "get_base64_encoded_image",
    "check_img",
    # PDF Utilities
    "find_annotated_pdf_for_filename",
    "regenerate_annotated_pdfs_from_chat_chunks",
    "update_pdf_view",
    "display_pdf_viewer",
    # Citation Utilities
    "process_chat_response_for_numbered_citations",
    "display_followup_citations_like_main_analysis",
    "display_chat_message_with_citations",
    # Export Utilities
    "create_report_package_content",
    "export_to_word",
    # Analysis Display
    "display_analysis_results",
    "display_rag_retry_button_header",
    "display_rag_results_section",
    "display_section_facts_expander",
    # Tools Column
    "display_tools_column",
]

