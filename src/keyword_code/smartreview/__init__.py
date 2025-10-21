"""
SmartReview - AI-powered document validation module.

This module provides functionality for creating validation templates and running
document compliance checks using regex and semantic validation.
"""

from .smartreview import (
    # Pydantic Models
    ProposedValidation,
    Rule,
    ValidationTemplate,
    ValidationResult,
    DocumentChunk,
    
    # Core Logic Functions
    decompose_rule_smartreview,
    extract_text_from_pdf,
    propose_validation_from_rule,
    refine_validation_from_chat,
    execute_validation_template,
    run_rule_on_chunk,
    
    # UI Rendering Functions
    render_validation_view,
    render_rule_definition_view,
    
    # Session State
    initialize_smartreview_session_state,
)

__all__ = [
    # Pydantic Models
    'ProposedValidation',
    'Rule',
    'ValidationTemplate',
    'ValidationResult',
    'DocumentChunk',
    
    # Core Logic Functions
    'decompose_rule_smartreview',
    'extract_text_from_pdf',
    'propose_validation_from_rule',
    'refine_validation_from_chat',
    'execute_validation_template',
    'run_rule_on_chunk',
    
    # UI Rendering Functions
    'render_validation_view',
    'render_rule_definition_view',
    
    # Session State
    'initialize_smartreview_session_state',
]

