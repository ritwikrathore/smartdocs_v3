"""AI functionality for the keyword_code package."""

from .analyzer import DocumentAnalyzer
from .chat import generate_chat_response
from .databricks_llm import get_databricks_llm, DatabricksLLMClient
from .decomposition import (
    decompose_prompt,  # Deprecated - use decompose_ask_mode_prompt instead
    decompose_ask_mode_prompt,
    decompose_review_mode_prompt,
    generate_regex_pattern,
    ReviewModeSubPrompt,
    RegexGenerationResult
)

__all__ = [
    'DocumentAnalyzer',
    'generate_chat_response',
    'get_databricks_llm',
    'DatabricksLLMClient',
    'decompose_prompt',  # Deprecated
    'decompose_ask_mode_prompt',
    'decompose_review_mode_prompt',
    'generate_regex_pattern',
    'ReviewModeSubPrompt',
    'RegexGenerationResult'
]
