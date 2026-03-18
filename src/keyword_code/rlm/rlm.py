"""
Main RLM (Recursive Language Model) engine.

Orchestrates the REPL loop: LLM generates code → REPL executes → results fed back
until FINAL() is called or max iterations reached.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..ai.databricks_llm import DatabricksLLMClient
from ..config import (
    logger,
    RLM_MAX_ITERATIONS,
    RLM_MAX_ERRORS,
    RLM_CONTEXT_PREVIEW_CHARS,
)
from .parsing import find_code_blocks, find_final_answer, truncate_output
from .prompts import SYSTEM_PROMPT, build_user_prompt, build_synthesis_prompt
from .repl import REPLEnvironment


@dataclass
class RLMResult:
    """Final output of an RLM completion."""
    answer: str
    citations: Optional[List[Dict[str, str]]] = None
    iterations: int = 0
    total_tokens: int = 0
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


class RLMEngine:
    """Drives the RLM REPL loop over document chunks."""

    def __init__(
        self,
        llm_client: DatabricksLLMClient,
        max_iterations: int = RLM_MAX_ITERATIONS,
        max_errors: int = RLM_MAX_ERRORS,
        verbose: bool = True,
    ) -> None:
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.max_errors = max_errors
        self.verbose = verbose

    def completion(
        self,
        chunks: List[Dict[str, Any]],
        full_text: str,
        question: str,
    ) -> RLMResult:
        """
        Run the RLM loop to answer a question over the given document chunks.

        Args:
            chunks: List of chunk dicts (from PDF processor + chunker with metadata)
            full_text: Raw full document text
            question: User question

        Returns:
            RLMResult with answer, citations, and execution metadata
        """
        # --- Setup REPL ---
        repl = REPLEnvironment()
        repl.setup(
            chunks=chunks,
            full_text=full_text,
            llm_query_fn=self._make_llm_query_fn(),
            llm_query_batched_fn=self._make_llm_query_batched_fn(),
        )

        # --- Build initial messages ---
        context_preview = full_text[:RLM_CONTEXT_PREVIEW_CHARS] if full_text else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    question=question,
                    num_chunks=len(chunks),
                    num_pages=max((c.get("page_num", 0) for c in chunks), default=0) + 1 if chunks else 0,
                    context_length=len(full_text),
                    context_preview=context_preview,
                ),
            },
        ]

        execution_log: List[Dict[str, Any]] = []
        total_tokens = 0
        consecutive_errors = 0

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                logger.info(f"RLM iteration {iteration}/{self.max_iterations}")

            # --- Call LLM ---
            response = self.llm.get_completion(messages)
            if response is None:
                logger.error("LLM returned None — aborting RLM loop")
                return RLMResult(
                    answer="Error: LLM failed to respond.",
                    iterations=iteration,
                    total_tokens=total_tokens,
                    execution_log=execution_log,
                )

            content = response["content"]
            usage = response.get("usage")
            if usage:
                total_tokens += usage.get("total_tokens", 0)

            # --- Parse code blocks ---
            code_blocks = find_code_blocks(content)

            iter_log: Dict[str, Any] = {
                "iteration": iteration,
                "llm_response_preview": content[:500],
                "code_blocks_found": len(code_blocks),
                "results": [],
            }

            if not code_blocks:
                # Check if LLM called FINAL in prose
                prose_final = find_final_answer(content)
                if prose_final:
                    iter_log["prose_final"] = True
                    execution_log.append(iter_log)
                    return RLMResult(
                        answer=prose_final,
                        iterations=iteration,
                        total_tokens=total_tokens,
                        execution_log=execution_log,
                    )

                # No code and no FINAL — nudge the LLM
                iter_log["no_code"] = True
                execution_log.append(iter_log)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Please write Python code in a ```repl block to explore the document. "
                               "Use find_chunks(), get_chunks_by_page(), or examine chunks directly.",
                })
                continue

            # --- Execute code blocks ---
            messages.append({"role": "assistant", "content": content})
            execution_summary_parts: List[str] = []

            for i, code in enumerate(code_blocks):
                result = repl.execute_code(code)

                block_log = {
                    "block_index": i,
                    "code_preview": code[:300],
                    "stdout": truncate_output(result.stdout),
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "error": result.error,
                    "has_final": result.final_answer is not None,
                }
                iter_log["results"].append(block_log)

                if self.verbose:
                    if result.stdout:
                        logger.info(f"  REPL stdout: {result.stdout[:200]}")
                    if result.stderr:
                        logger.warning(f"  REPL stderr: {result.stderr[:200]}")

                # Build summary for history
                parts = []
                if result.stdout:
                    parts.append(f"Output:\n{truncate_output(result.stdout)}")
                if result.stderr:
                    parts.append(f"Error:\n{result.stderr[:500]}")
                if result.error:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if parts:
                    execution_summary_parts.append(
                        f"[Block {i+1}]\n" + "\n".join(parts)
                    )

                # Check for final answer
                if result.final_answer is not None:
                    execution_log.append(iter_log)
                    return RLMResult(
                        answer=result.final_answer,
                        citations=result.citations,
                        iterations=iteration,
                        total_tokens=total_tokens,
                        execution_log=execution_log,
                    )

            execution_log.append(iter_log)

            # Abort on too many consecutive errors
            if consecutive_errors >= self.max_errors:
                logger.error(f"RLM aborting: {consecutive_errors} consecutive errors")
                return RLMResult(
                    answer="Error: Too many consecutive execution errors.",
                    iterations=iteration,
                    total_tokens=total_tokens,
                    execution_log=execution_log,
                )

            # Append execution results as user message for next iteration
            if execution_summary_parts:
                messages.append({
                    "role": "user",
                    "content": "REPL execution results:\n\n" + "\n\n".join(execution_summary_parts),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Code executed successfully with no output. Continue your analysis.",
                })

        # --- Max iterations reached: force synthesis ---
        logger.warning("RLM max iterations reached — forcing synthesis")
        messages.append({"role": "user", "content": build_synthesis_prompt()})

        response = self.llm.get_completion(messages)
        if response and response["content"]:
            content = response["content"]
            usage = response.get("usage")
            if usage:
                total_tokens += usage.get("total_tokens", 0)

            # Try to execute any final code
            code_blocks = find_code_blocks(content)
            for code in code_blocks:
                result = repl.execute_code(code)
                if result.final_answer is not None:
                    return RLMResult(
                        answer=result.final_answer,
                        citations=result.citations,
                        iterations=self.max_iterations + 1,
                        total_tokens=total_tokens,
                        execution_log=execution_log,
                    )

            # If still no FINAL, use the LLM's prose as answer
            return RLMResult(
                answer=content,
                iterations=self.max_iterations + 1,
                total_tokens=total_tokens,
                execution_log=execution_log,
            )

        return RLMResult(
            answer="Error: Failed to synthesize answer after max iterations.",
            iterations=self.max_iterations + 1,
            total_tokens=total_tokens,
            execution_log=execution_log,
        )

    # ------------------------------------------------------------------
    # LLM callback factories
    # ------------------------------------------------------------------

    def _make_llm_query_fn(self) -> Callable[[str], str]:
        """Create a sub-query function the REPL code can call."""
        def llm_query(prompt: str) -> str:
            messages = [
                {"role": "system", "content": "You are a helpful document analyst. Answer concisely based on the provided context."},
                {"role": "user", "content": prompt},
            ]
            response = self.llm.get_completion(messages, max_tokens=2048)
            if response and response.get("content"):
                return response["content"]
            return "[LLM sub-query failed]"
        return llm_query

    def _make_llm_query_batched_fn(self) -> Callable[[List[str]], List[str]]:
        """Create a batched sub-query function (sequential for now)."""
        single_fn = self._make_llm_query_fn()
        def llm_query_batched(prompts: List[str]) -> List[str]:
            return [single_fn(p) for p in prompts]
        return llm_query_batched
