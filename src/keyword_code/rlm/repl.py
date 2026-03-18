"""
Sandboxed REPL environment for RLM document exploration.

Provides a persistent Python namespace where the LLM can execute code to
search, filter, and analyze document chunks programmatically.
"""

import contextlib
import io
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import logger


@dataclass
class REPLResult:
    """Result of a single REPL code execution."""
    stdout: str = ""
    stderr: str = ""
    locals_snapshot: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    citations: Optional[List[Dict[str, str]]] = None
    error: bool = False


# Builtins we block to keep the sandbox minimal
_BLOCKED_BUILTINS = {"eval", "exec", "input", "open", "__import__", "compile", "breakpoint"}


class REPLEnvironment:
    """Persistent, sandboxed Python REPL for RLM loops."""

    def __init__(self) -> None:
        self._namespace: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._final_answer: Optional[str] = None
        self._citations: Optional[List[Dict[str, str]]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        chunks: List[Dict[str, Any]],
        full_text: str,
        llm_query_fn: Callable[[str], str],
        llm_query_batched_fn: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        """Load document data and helpers into the REPL namespace."""
        self._final_answer = None
        self._citations = None

        # Build sandboxed builtins
        import builtins
        safe_builtins = {
            k: v for k, v in vars(builtins).items()
            if k not in _BLOCKED_BUILTINS
        }
        safe_builtins["__build_class__"] = builtins.__build_class__

        # Pre-approved imports the LLM can use directly
        import re as _re
        import json as _json
        import math as _math
        from collections import Counter as _Counter, defaultdict as _defaultdict

        self._namespace = {
            "__builtins__": safe_builtins,
            # Document data
            "chunks": chunks,
            "context": full_text,
            "num_chunks": len(chunks),
            "num_pages": max((c.get("page_num", 0) for c in chunks), default=0) + 1 if chunks else 0,
            "context_length": len(full_text),
            # Pre-imported modules
            "re": _re,
            "json": _json,
            "math": _math,
            "Counter": _Counter,
            "defaultdict": _defaultdict,
            # Helper functions
            "find_chunks": self._make_find_chunks(chunks),
            "get_chunks_by_page": self._make_get_chunks_by_page(chunks),
            "get_chunks_by_section": self._make_get_chunks_by_section(chunks),
            "llm_query": llm_query_fn,
            "llm_query_batched": llm_query_batched_fn or (lambda prompts: [llm_query_fn(p) for p in prompts]),
            # Control functions
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
            "SHOW_CHUNK_SCHEMA": self._show_chunk_schema,
        }

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute_code(self, code: str) -> REPLResult:
        """Execute a code string in the persistent namespace."""
        with self._lock:
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            result = REPLResult()

            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code, self._namespace)  # noqa: S102 — sandboxed builtins
            except Exception as exc:
                stderr_buf.write(f"{type(exc).__name__}: {exc}")
                result.error = True

            result.stdout = stdout_buf.getvalue()
            result.stderr = stderr_buf.getvalue()

            if self._final_answer is not None:
                result.final_answer = self._final_answer
                result.citations = self._citations

            # Snapshot user-created variables (skip internal stuff)
            result.locals_snapshot = {
                k: repr(v)[:200]
                for k, v in self._namespace.items()
                if not k.startswith("_") and k not in {
                    "__builtins__", "chunks", "context", "re", "json", "math",
                    "Counter", "defaultdict", "find_chunks", "get_chunks_by_page",
                    "get_chunks_by_section", "llm_query", "llm_query_batched",
                    "FINAL", "FINAL_VAR", "SHOW_VARS", "SHOW_CHUNK_SCHEMA",
                    "num_chunks", "num_pages", "context_length",
                }
            }

            return result

    @property
    def has_final_answer(self) -> bool:
        return self._final_answer is not None

    @property
    def final_answer(self) -> Optional[str]:
        return self._final_answer

    @property
    def citations(self) -> Optional[List[Dict[str, str]]]:
        return self._citations

    # ------------------------------------------------------------------
    # Control functions injected into namespace
    # ------------------------------------------------------------------

    def _final(self, answer: str, citations: Optional[List[Dict[str, str]]] = None) -> str:
        """Set the final answer and optionally attach citations."""
        self._final_answer = str(answer)
        if citations is not None:
            self._citations = citations
        print(f"[FINAL ANSWER SET — {len(self._final_answer)} chars]")
        return self._final_answer

    def _final_var(self, var_name: str) -> str:
        """Set final answer from a variable in the namespace."""
        if var_name not in self._namespace:
            raise NameError(f"Variable '{var_name}' not found. Use SHOW_VARS() to see available variables.")
        value = self._namespace[var_name]
        return self._final(str(value))

    def _show_vars(self) -> str:
        """List user-created variables in the namespace."""
        skip = {
            "__builtins__", "chunks", "context", "re", "json", "math",
            "Counter", "defaultdict", "find_chunks", "get_chunks_by_page",
            "get_chunks_by_section", "llm_query", "llm_query_batched",
            "FINAL", "FINAL_VAR", "SHOW_VARS", "SHOW_CHUNK_SCHEMA",
            "num_chunks", "num_pages", "context_length",
        }
        user_vars = {
            k: f"{type(v).__name__}: {repr(v)[:100]}"
            for k, v in self._namespace.items()
            if k not in skip and not k.startswith("_")
        }
        if not user_vars:
            msg = "No user variables defined yet."
        else:
            msg = "\n".join(f"  {k} = {v}" for k, v in user_vars.items())
        print(msg)
        return msg

    def _show_chunk_schema(self) -> str:
        """Print the schema of a chunk dict for LLM reference."""
        schema = (
            "Chunk schema (each item in `chunks` list):\n"
            "  chunk_id: str          — e.g. 'chunk_0'\n"
            "  text: str              — chunk text content\n"
            "  page_num: int          — 0-indexed page number\n"
            "  metadata: dict with keys:\n"
            "    document_scope: str  — 'preamble', 'article', 'annex', 'schedule', 'recital'\n"
            "    article_type: str    — 'Article', 'Annex', 'Schedule', 'Preamble', 'Recitals'\n"
            "    article_number: str  — e.g. 'I', 'II', '1'\n"
            "    article_title: str   — title of the article/section\n"
            "    section_number: str  — e.g. '1.01', '2.03'\n"
            "    section_title: str   — title of the section\n"
            "    subsection_label: str — e.g. '(a)', '(i)'\n"
            "    hierarchy_path: str  — full path like 'Article I - Definitions > Section 1.01'\n"
        )
        print(schema)
        return schema

    # ------------------------------------------------------------------
    # Helper function factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_find_chunks(chunks: List[Dict[str, Any]]) -> Callable[[str], List[Dict[str, Any]]]:
        """Create a keyword search function over chunks."""
        def find_chunks(keyword: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
            """Search chunks for a keyword. Returns matching chunks."""
            results = []
            kw = keyword if case_sensitive else keyword.lower()
            for c in chunks:
                text = c.get("text", "")
                compare = text if case_sensitive else text.lower()
                if kw in compare:
                    results.append(c)
            print(f"find_chunks('{keyword}'): {len(results)} matches")
            return results
        return find_chunks

    @staticmethod
    def _make_get_chunks_by_page(chunks: List[Dict[str, Any]]) -> Callable[[int], List[Dict[str, Any]]]:
        """Create a page filter function."""
        def get_chunks_by_page(page_num: int) -> List[Dict[str, Any]]:
            """Get all chunks from a specific page (0-indexed)."""
            results = [c for c in chunks if c.get("page_num") == page_num]
            print(f"get_chunks_by_page({page_num}): {len(results)} chunks")
            return results
        return get_chunks_by_page

    @staticmethod
    def _make_get_chunks_by_section(chunks: List[Dict[str, Any]]) -> Callable[[str], List[Dict[str, Any]]]:
        """Create a section filter function."""
        def get_chunks_by_section(section: str) -> List[Dict[str, Any]]:
            """Get chunks whose metadata matches a section pattern (case-insensitive substring)."""
            results = []
            s_lower = section.lower()
            for c in chunks:
                meta = c.get("metadata", {})
                searchable = " ".join(str(v) for v in meta.values() if v).lower()
                if s_lower in searchable:
                    results.append(c)
            print(f"get_chunks_by_section('{section}'): {len(results)} chunks")
            return results
        return get_chunks_by_section
