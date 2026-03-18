"""
RLM (Recursive Language Model) module for SmartDocs.

Replaces traditional retrieve→LLM with an iterative REPL loop where the LLM
programmatically navigates document chunks to build grounded answers.
"""

from .rlm import RLMEngine, RLMResult
from .repl import REPLEnvironment, REPLResult

__all__ = ["RLMEngine", "RLMResult", "REPLEnvironment", "REPLResult"]
