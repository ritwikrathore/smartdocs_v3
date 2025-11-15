"""
Document analyzer functionality.
"""

import json
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from ..config import logger, ANALYSIS_MODEL_NAME, USE_DATABRICKS_LLM, LLM_MAX_RETRIES

# Import Databricks LLM client
from .databricks_llm import get_databricks_llm

# Import interaction logger
from ..utils.interaction_logger import log_llm_interaction

# Import Langfuse tracing
from ..utils.langfuse_tracing import (
    optional_context,
    record_generation_error,
    set_generation_output,
    set_span_output,
    start_generation,
    start_span,
)


_thread_local = threading.local()

_TRACE_METADATA_SKIP_KEYS = {"tokens", "embedding", "bbox", "bboxes"}


def _truncate_for_trace(value: str, limit: int = 400) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _coerce_trace_value(value: Any, *, max_length: int = 200, max_items: int = 10) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + "..."
        return value
    if isinstance(value, list):
        sanitized_list = [
            _coerce_trace_value(item, max_length=max_length, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            sanitized_list.append("...")
        return sanitized_list
    if isinstance(value, dict):
        sanitized_dict: Dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_items:
                sanitized_dict["..."] = f"{len(value) - max_items} more keys"
                break
            sanitized_dict[str(key)] = _coerce_trace_value(val, max_length=max_length, max_items=max_items)
        return sanitized_dict
    if isinstance(value, set):
        truncated = list(value)[:max_items]
        sanitized = [
            _coerce_trace_value(item, max_length=max_length, max_items=max_items)
            for item in truncated
        ]
        if len(value) > max_items:
            sanitized.append("...")
        return sanitized
    value_str = str(value)
    if len(value_str) > max_length:
        return value_str[:max_length] + "..."
    return value_str


def _summarize_relevant_chunk(chunk: Dict[str, Any], *, rank: int, fallback_index: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rank": rank,
        "chunk_id": chunk.get("chunk_id"),
        "chunk_index": chunk.get("chunk_index", fallback_index),
        "retrieval_method": chunk.get("retrieval_method"),
        "page_num": chunk.get("page_num"),
        "page_label": chunk.get("page_label"),
        "text_preview": _truncate_for_trace(chunk.get("text", ""), limit=300),
    }

    score_value = chunk.get("score")
    if score_value is not None:
        try:
            summary["score"] = float(score_value)
        except (TypeError, ValueError):
            summary["score"] = score_value

    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        sanitized_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key in _TRACE_METADATA_SKIP_KEYS:
                continue
            sanitized_metadata[key] = _coerce_trace_value(value)
        if sanitized_metadata:
            summary["metadata"] = sanitized_metadata

    return summary


def _sanitize_unescaped_control_chars(json_str: str) -> Tuple[str, bool]:
    """Escape control characters that appear unescaped within JSON strings."""

    sanitized_chars: List[str] = []
    in_string = False
    escape_next = False
    sanitized = False

    for ch in json_str:
        if in_string:
            if escape_next:
                sanitized_chars.append(ch)
                escape_next = False
                continue

            if ch == "\\":
                sanitized_chars.append(ch)
                escape_next = True
                continue

            if ch == '"':
                sanitized_chars.append(ch)
                in_string = False
                continue

            code_point = ord(ch)
            if ch == "\n":
                sanitized_chars.append("\\n")
                sanitized = True
                continue
            if ch == "\r":
                sanitized_chars.append("\\r")
                sanitized = True
                continue
            if ch == "\t":
                sanitized_chars.append("\\t")
                sanitized = True
                continue
            if code_point < 0x20:
                sanitized_chars.append(f"\\u{code_point:04x}")
                sanitized = True
                continue

            sanitized_chars.append(ch)
            continue

        sanitized_chars.append(ch)
        if ch == '"':
            in_string = True
            escape_next = False

    return "".join(sanitized_chars), sanitized


class DocumentAnalyzer:
    def __init__(self):
        # Initialize Databricks LLM client
        self.databricks_client = get_databricks_llm() if USE_DATABRICKS_LLM else None

        # Log client initialization status
        if self.databricks_client:
            logger.info("DocumentAnalyzer initialized with Databricks LLM client")
        else:
            logger.error("DocumentAnalyzer failed to initialize Databricks LLM client")

    def _ensure_client(self, model_name: str):
        # Return client configuration
        if self.databricks_client:
            return {"client": self.databricks_client, "model_name": model_name, "type": "databricks"}
        else:
            raise ValueError("No LLM client available - Databricks LLM client failed to initialize")

    def _normalize_metadata_key(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = str(value).strip()
        if not value_str:
            return None
        return value_str.lower()

    def _register_index_entry(self, index_map: Dict[str, List[int]], key: Any, chunk_index: int) -> None:
        normalized = self._normalize_metadata_key(key)
        if not normalized:
            return
        bucket = index_map.setdefault(normalized, [])
        if chunk_index not in bucket:
            bucket.append(chunk_index)

    def _resolve_chunk_index(self, chunk: Dict[str, Any], fallback_index: int) -> int:
        index = chunk.get("chunk_index")
        if isinstance(index, int):
            return index
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            meta_index = metadata.get("chunk_index")
            if isinstance(meta_index, int):
                return meta_index
        return fallback_index

    def _format_chunk_excerpt(self, chunk: Dict[str, Any], chunk_index: int) -> str:
        metadata = chunk.get("metadata") or {}
        location_parts: List[str] = []
        article_number = metadata.get("article_number")
        if article_number:
            article_type = metadata.get("article_type", "Article")
            location_parts.append(f"{article_type} {article_number}")
            if metadata.get("article_title"):
                location_parts.append(str(metadata["article_title"]))
        section_number = metadata.get("section_number")
        if section_number:
            location_parts.append(str(section_number))
            if metadata.get("section_title"):
                location_parts.append(str(metadata["section_title"]))
        if metadata.get("subsection_label"):
            location_parts.append(f"Subsection ({metadata['subsection_label']})")
        location_label = " - ".join(part for part in location_parts if part)
        page_label = chunk.get("page_label")
        page_num = chunk.get("page_num")
        lines = [f"CHUNK_INDEX: {chunk_index}"]
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            lines.append(f"CHUNK_ID: {chunk_id}")
        if location_label:
            lines.append(f"LOCATION: {location_label}")
        if page_label is not None:
            lines.append(f"PAGE_LABEL: {page_label}")
        elif isinstance(page_num, int) and page_num >= 0:
            lines.append(f"PAGE_NUMBER: {page_num + 1}")
        lines.append("TEXT:")
        lines.append(chunk.get("text", ""))
        return "\n".join(lines)

    def _parse_context_request_payload(self, payload: Any) -> Tuple[bool, List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return False, []

        needs_context = bool(payload.get("needs_additional_context"))
        if not needs_context:
            return False, []

        raw_requests = payload.get("context_requests")
        if raw_requests is None:
            raw_requests = payload.get("requested_context")
        if raw_requests is None:
            raw_requests = payload.get("additional_context_requests")

        if raw_requests is None:
            return True, []

        if isinstance(raw_requests, dict):
            raw_requests = [raw_requests]

        normalized_requests: List[Dict[str, Any]] = []
        if not isinstance(raw_requests, list):
            return True, []

        def _collect_values(raw_value: Any) -> List[str]:
            values: List[str] = []
            if raw_value is None:
                return values
            items: List[Any]
            if isinstance(raw_value, (list, tuple, set)):
                items = list(raw_value)
            else:
                items = [raw_value]
            for entry in items:
                if entry is None:
                    continue
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate:
                        values.append(candidate)
                else:
                    values.append(str(entry))
            return values

        def _extract_list(source: Dict[str, Any], keys: Tuple[str, ...]) -> List[str]:
            seen: Set[str] = set()
            result: List[str] = []
            for key in keys:
                if key not in source:
                    continue
                for value in _collect_values(source.get(key)):
                    if value not in seen:
                        seen.add(value)
                        result.append(value)
            return result

        for item in raw_requests:
            if not isinstance(item, dict):
                continue

            raw_index = item.get("sub_prompt_index")
            if raw_index is None:
                raw_index = item.get("sub_prompt")
            if raw_index is None:
                raw_index = item.get("index")

            try:
                sub_prompt_index = int(raw_index)
            except (TypeError, ValueError):
                continue

            raw_chunks = item.get("chunk_indices")
            if raw_chunks is None:
                raw_chunks = item.get("chunks")
            if raw_chunks is None:
                raw_chunks = item.get("requested_chunks")

            chunk_indices: List[int] = []
            if isinstance(raw_chunks, list):
                for elem in raw_chunks:
                    try:
                        chunk_indices.append(int(elem))
                    except (TypeError, ValueError):
                        continue
            elif raw_chunks is not None:
                try:
                    chunk_indices.append(int(raw_chunks))
                except (TypeError, ValueError):
                    pass

            article_numbers = _extract_list(
                item,
                ("article_numbers", "article_number", "articles", "article"),
            )
            section_numbers = _extract_list(
                item,
                ("section_numbers", "section_number", "sections", "section"),
            )
            section_titles = _extract_list(
                item,
                ("section_titles", "section_title", "section_names", "section_name"),
            )

            normalized_requests.append({
                "sub_prompt_index": sub_prompt_index,
                "chunk_indices": chunk_indices,
                "reason": item.get("reason"),
                "direction": item.get("direction"),
                "article_numbers": article_numbers,
                "section_numbers": section_numbers,
                "section_titles": section_titles,
            })

        return True, normalized_requests

    def _build_additional_context_response(
        self,
        requests: List[Dict[str, Any]],
        sub_prompts_with_contexts: List[Dict[str, Any]],
        chunk_lookup: Dict[int, Dict[str, Any]],
        provided_chunk_indices: Dict[int, Set[int]],
        metadata_index_maps: Dict[str, Dict[str, List[int]]],
        max_new_chunks: int = 12,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        response_lines: List[str] = []
        details: List[Dict[str, Any]] = []

        for request in requests:
            sub_index = request.get("sub_prompt_index")
            if not isinstance(sub_index, int) or sub_index < 1:
                continue

            title = ""
            if 0 <= sub_index - 1 < len(sub_prompts_with_contexts):
                title = sub_prompts_with_contexts[sub_index - 1].get("title", "")

            already_provided: List[int] = []
            missing_indices: List[int] = []
            provided_now: List[int] = []
            excerpts: List[str] = []
            truncated_due_to_limit = False

            requested_chunk_indices = request.get("chunk_indices") or []
            requested_chunk_indices = [int(idx) for idx in requested_chunk_indices if isinstance(idx, int)]

            article_numbers = request.get("article_numbers") or []
            section_numbers = request.get("section_numbers") or []
            section_titles = request.get("section_titles") or []

            metadata_requests = {
                "article_numbers": article_numbers,
                "section_numbers": section_numbers,
                "section_titles": section_titles,
            }

            metadata_hits: Dict[str, List[int]] = {key: [] for key in metadata_requests}
            metadata_missing: Dict[str, List[str]] = {}

            index_bucket = provided_chunk_indices.setdefault(sub_index, set())

            candidate_indices: List[int] = []
            candidate_origin: Dict[int, Set[str]] = {}

            def _register_candidate(candidate_idx: int, origin_key: str) -> None:
                origin_bucket = candidate_origin.setdefault(candidate_idx, set())
                origin_bucket.add(origin_key)
                if candidate_idx not in candidate_indices:
                    candidate_indices.append(candidate_idx)

            for chunk_idx in requested_chunk_indices:
                _register_candidate(chunk_idx, "chunk_indices")

            for map_key, values in metadata_requests.items():
                if not values:
                    continue
                index_map = metadata_index_maps.get(map_key) or {}
                hits_for_key: List[int] = []
                missing_for_key: List[str] = []
                for value in values:
                    normalized = self._normalize_metadata_key(value)
                    if not normalized:
                        missing_for_key.append(str(value))
                        continue
                    indices = index_map.get(normalized)
                    if not indices:
                        missing_for_key.append(str(value))
                        continue
                    hits_for_key.extend(indices)
                if hits_for_key:
                    dedup_hits_for_key: List[int] = []
                    for hit in hits_for_key:
                        if hit not in dedup_hits_for_key:
                            dedup_hits_for_key.append(hit)
                        _register_candidate(hit, map_key)
                    metadata_hits[map_key] = dedup_hits_for_key
                if missing_for_key:
                    metadata_missing[map_key] = missing_for_key

            # Remove duplicates while preserving order
            seen_indices: Set[int] = set()
            ordered_candidates: List[int] = []
            for candidate in candidate_indices:
                if candidate not in seen_indices:
                    ordered_candidates.append(candidate)
                    seen_indices.add(candidate)

            for chunk_idx in ordered_candidates:
                if len(provided_now) >= max_new_chunks:
                    truncated_due_to_limit = True
                    break

                if chunk_idx in index_bucket:
                    already_provided.append(chunk_idx)
                    continue

                chunk = chunk_lookup.get(chunk_idx)
                if not chunk:
                    missing_indices.append(chunk_idx)
                    continue

                index_bucket.add(chunk_idx)
                provided_now.append(chunk_idx)
                excerpts.append(self._format_chunk_excerpt(chunk, chunk_idx))

            if article_numbers or section_numbers or section_titles:
                descriptor_parts: List[str] = []
                if article_numbers:
                    descriptor_parts.append("articles " + ", ".join(article_numbers))
                if section_numbers:
                    descriptor_parts.append("sections " + ", ".join(section_numbers))
                if section_titles:
                    descriptor_parts.append("section titles " + ", ".join(section_titles))
                response_lines.append(
                    f"Metadata-based request for sub-prompt {sub_index}: " + "; ".join(descriptor_parts)
                )

            if provided_now:
                header = f"Additional context for sub-prompt {sub_index}"
                if title:
                    header += f" ({title})"
                response_lines.append(header + ":")
                response_lines.extend(excerpts)

            if truncated_due_to_limit:
                response_lines.append(
                    f"Context delivery truncated to {max_new_chunks} chunk(s). Request a narrower portion if more detail is still needed."
                )

            if already_provided and not provided_now:
                response_lines.append(
                    f"Requested material already supplied for sub-prompt {sub_index}: {already_provided}."
                )
            elif already_provided:
                response_lines.append(
                    f"Note: previously supplied chunk indices for sub-prompt {sub_index}: {already_provided}."
                )

            if missing_indices:
                response_lines.append(
                    f"Unable to locate requested chunk indices for sub-prompt {sub_index}: {missing_indices}."
                )

            if metadata_missing:
                for map_key, missing_values in metadata_missing.items():
                    if not missing_values:
                        continue
                    label = map_key.replace("_", " ")
                    response_lines.append(
                        f"Could not resolve {label} {missing_values} for sub-prompt {sub_index}."
                    )

            provided_origin_map: Dict[str, List[int]] = {}
            already_origin_map: Dict[str, List[int]] = {}
            for idx in provided_now:
                origins = candidate_origin.get(idx, {"chunk_indices"})
                for origin in origins:
                    provided_origin_map.setdefault(origin, []).append(idx)
            for idx in already_provided:
                origins = candidate_origin.get(idx, {"chunk_indices"})
                for origin in origins:
                    already_origin_map.setdefault(origin, []).append(idx)

            details.append({
                "sub_prompt_index": sub_index,
                "requested_indices": ordered_candidates,
                "provided_indices": provided_now,
                "already_provided": already_provided,
                "missing_indices": missing_indices,
                "reason": request.get("reason"),
                "direction": request.get("direction"),
                "article_numbers": article_numbers,
                "section_numbers": section_numbers,
                "section_titles": section_titles,
                "provided_from": provided_origin_map,
                "already_from": already_origin_map,
                "metadata_hits": metadata_hits,
                "metadata_missing": metadata_missing,
                "truncated": truncated_due_to_limit,
            })

        message = "\n\n".join(line for line in response_lines if line.strip())
        if message:
            message += "\n\nContinue the analysis with these excerpts."
        else:
            message = (
                "No additional excerpts could be supplied for the requested indices. Please proceed using the existing context."
            )

        return message, details

    async def _get_completion(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        *,
        max_attempts: Optional[int] = None,
    ) -> str:
        """Helper method to get completion from Databricks LLM with retry support."""

        attempts = max_attempts or LLM_MAX_RETRIES

        for attempt in range(1, attempts + 1):
            try:
                # Ensure Databricks client is available
                if not self.databricks_client:
                    raise ValueError("Databricks LLM client not initialized")

                logger.info("Sending request to Databricks LLM model")
                # Databricks client handles message formatting internally
                response_content = await self.databricks_client.get_completion_async(messages, max_tokens=8192)

                if not response_content:
                    raise ValueError("Failed to get response from Databricks LLM")

                logger.info("Received response from Databricks LLM model")

                # Log the LLM interaction
                interaction_type = "analysis"
                if model_name == "databricks-gpt-oss-120b":
                    if any("decompose" in msg.get("content", "").lower() for msg in messages if msg.get("role") == "system"):
                        interaction_type = "decomposition"
                    elif any("chat" in msg.get("content", "").lower() for msg in messages if msg.get("role") == "system"):
                        interaction_type = "chat"

                log_llm_interaction(messages, response_content, interaction_type)

                return response_content

            except Exception as e:
                logger.error(
                    "Error getting completion from Databricks LLM (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    e,
                    exc_info=attempt == attempts,
                )
                if attempt == attempts:
                    raise
                logger.info(
                    "Retrying Databricks LLM call (attempt %d/%d)",
                    attempt + 1,
                    attempts,
                )

    async def _get_json_with_retries(
        self,
        *,
        messages: List[Dict[str, str]],
        model_name: str,
        context: str,
        extractor: Optional[Callable[[str], Any]] = None,
        max_attempts: Optional[int] = None,
        return_raw: bool = False,
    ) -> Any:
        """Call the LLM and parse the response as JSON with retry logic."""

        attempts = max_attempts or LLM_MAX_RETRIES
        extractor = extractor or self._extract_json_from_response
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                response_content = await self._get_completion(
                    messages,
                    model_name,
                    max_attempts=1,
                )
            except Exception as err:
                last_error = err
                logger.error(
                    "LLM call failed during %s (attempt %d/%d): %s",
                    context,
                    attempt,
                    attempts,
                    err,
                    exc_info=attempt == attempts,
                )
                if attempt == attempts:
                    raise
                logger.info(
                    "Retrying %s after call failure (attempt %d/%d)",
                    context,
                    attempt + 1,
                    attempts,
                )
                continue

            logger.debug(
                "Raw LLM response for %s (attempt %d/%d): %s",
                context,
                attempt,
                attempts,
                response_content[:500] if response_content else "",
            )

            try:
                parsed = extractor(response_content)
                if return_raw:
                    return parsed, response_content
                return parsed
            except json.JSONDecodeError as json_err:
                last_error = json_err
                logger.error(
                    "Failed to parse LLM response as JSON during %s (attempt %d/%d): %s",
                    context,
                    attempt,
                    attempts,
                    json_err,
                    exc_info=attempt == attempts,
                )
                if attempt == attempts:
                    raise
                logger.info(
                    "Retrying %s due to JSON parsing error (attempt %d/%d)",
                    context,
                    attempt + 1,
                    attempts,
                )

        if last_error:
            raise last_error
        raise RuntimeError(f"LLM retries exhausted for {context}")

    @staticmethod
    def _extract_json_from_response(response_content: str) -> Any:
        """Extract the first JSON object from a model response."""

        cleaned_response = response_content.strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
        if match:
            json_str = match.group(1)
        elif cleaned_response.startswith("{") and cleaned_response.endswith("}"):
            json_str = cleaned_response
        else:
            first_brace = cleaned_response.find("{")
            last_brace = cleaned_response.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = cleaned_response[first_brace:last_brace + 1]
                logger.warning("Used brace slicing heuristic to extract JSON from model response.")
            else:
                raise json.JSONDecodeError("Could not locate JSON payload in response.", cleaned_response, 0)

        sanitized_json_str, sanitized = _sanitize_unescaped_control_chars(json_str)
        if sanitized:
            logger.debug("Escaped control characters in JSON payload before parsing.")

        return json.loads(sanitized_json_str)

    @property
    def output_schema_analysis(self) -> dict:
        """Defines the expected JSON structure for document analysis."""
        # Keep this schema definition as it's used in the analysis prompt
        return {
            "title": "Concise Title for the Analysis Section based on the specific sub-prompt",
            "analysis_sections": {
                "descriptive_section_name_1": {
                    "Analysis": "Detailed analysis text for this section...",
                    "Supporting_Phrases": [
                        "Exact quote 1 from the document text...",
                        "Exact quote 2, potentially longer...",
                    ],
                    "Context": "Optional context about this section (e.g., source sections)",
                },
                # Add more sections as identified by the AI FOR THIS SUB-PROMPT
            },
        }

    async def analyze_document_with_all_contexts(
        self,
        filename: str,
        main_prompt: str,
        sub_prompts_with_contexts: List[Dict[str, Any]],
        all_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyzes all sub-prompts with their relevant contexts in a single LLM call.

        Args:
            filename: Name of the document being analyzed
            main_prompt: The original user prompt
            sub_prompts_with_contexts: List of dictionaries, each containing:
                - 'title': Title of the sub-prompt
                - 'sub_prompt': The sub-prompt text
                - 'relevant_chunks': List of relevant chunks for this sub-prompt

        Returns:
            List of dictionaries, each containing the analysis for a sub-prompt
        """
        try:
            if not sub_prompts_with_contexts:
                logger.warning(f"No sub-prompts with contexts provided for {filename}")
                return []

            # Extract table of contents for document navigation context
            toc_context = ""
            if all_chunks:
                toc_chunks = [c for c in all_chunks if c.get('metadata', {}).get('document_scope') == 'table_of_contents']
                if toc_chunks:
                    toc_entries_list = []
                    for toc_chunk in toc_chunks:
                        entries = toc_chunk.get('metadata', {}).get('toc_entries', [])
                        toc_entries_list.extend(entries)
                    
                    if toc_entries_list:
                        toc_lines = [f"  - {entry.get('entry', '')}: Page {entry.get('page_number', 'N/A')}" 
                                    for entry in toc_entries_list[:50]]  # Limit to first 50 entries
                        toc_context = "\n".join(toc_lines)
                        logger.info(f"Extracted {len(toc_entries_list)} TOC entries for analyzer context")

            # Build lookup for on-demand context expansion
            chunk_lookup: Dict[int, Dict[str, Any]] = {}
            metadata_index_maps: Dict[str, Dict[str, List[int]]] = {
                "article_numbers": {},
                "section_numbers": {},
                "section_titles": {},
            }
            if all_chunks:
                for idx, chunk in enumerate(all_chunks):
                    resolved_index = chunk.get("chunk_index")
                    if not isinstance(resolved_index, int):
                        resolved_index = idx
                    if resolved_index not in chunk_lookup:
                        chunk_lookup[resolved_index] = chunk
                    metadata = chunk.get("metadata") or {}
                    article_number = metadata.get("article_number")
                    if article_number is not None:
                        self._register_index_entry(metadata_index_maps["article_numbers"], article_number, resolved_index)
                        article_type = metadata.get("article_type")
                        if article_type:
                            composite = f"{article_type} {article_number}"
                            self._register_index_entry(metadata_index_maps["article_numbers"], composite, resolved_index)
                    section_number = metadata.get("section_number")
                    if section_number is not None:
                        self._register_index_entry(metadata_index_maps["section_numbers"], section_number, resolved_index)
                    section_title = metadata.get("section_title")
                    if section_title:
                        self._register_index_entry(metadata_index_maps["section_titles"], section_title, resolved_index)
                    section_path = metadata.get("section_path")
                    if isinstance(section_path, list):
                        for segment in section_path:
                            self._register_index_entry(metadata_index_maps["section_titles"], segment, resolved_index)
                    elif section_path:
                        self._register_index_entry(metadata_index_maps["section_titles"], section_path, resolved_index)

            # Format all sub-prompts and their contexts
            formatted_sub_prompts = []
            trace_sub_prompts_payload: List[Dict[str, Any]] = []
            provided_chunk_indices: Dict[int, Set[int]] = {}
            for i, item in enumerate(sub_prompts_with_contexts):
                sub_prompt = item.get('sub_prompt', '')
                title = item.get('title', f'Sub-prompt {i+1}')
                relevant_chunks = item.get('relevant_chunks', [])

                if not relevant_chunks:
                    logger.warning(f"No relevant chunks for sub-prompt '{title}' in {filename}")
                    formatted_context = "No relevant text found for this sub-prompt."
                    trace_chunks: List[Dict[str, Any]] = []
                else:
                    # Format relevant chunks for this sub-prompt
                    # Include metadata to help the model understand document structure
                    formatted_chunks = []
                    trace_chunks = []
                    for chunk_position, chunk in enumerate(relevant_chunks):
                        resolved_index = self._resolve_chunk_index(chunk, chunk_position)
                        provided_chunk_indices.setdefault(i + 1, set()).add(resolved_index)
                        formatted_chunks.append(self._format_chunk_excerpt(chunk, resolved_index))
                        trace_chunks.append(
                            _summarize_relevant_chunk(
                                chunk,
                                rank=len(trace_chunks) + 1,
                                fallback_index=resolved_index,
                            )
                        )
                    
                    formatted_context = "\n\n---\n\n".join(formatted_chunks)
                if (i + 1) not in provided_chunk_indices:
                    provided_chunk_indices[i + 1] = set()

                formatted_sub_prompts.append({
                    "index": i + 1,
                    "title": title,
                    "sub_prompt": sub_prompt,
                    "context": formatted_context
                })

                trace_sub_prompts_payload.append({
                    "index": i + 1,
                    "title": title,
                    "sub_prompt": sub_prompt,
                    "context_preview": _truncate_for_trace(formatted_context, limit=600),
                    "num_relevant_chunks": len(relevant_chunks),
                    "chunk_summaries": trace_chunks,
                })

            # Create the system prompt for the comprehensive analysis
            system_prompt = """You are an intelligent document analyzer specializing in legal and financial documents. You will be given a main prompt, multiple sub-prompts derived from it, and relevant document excerpts for each sub-prompt. Your task is to analyze each sub-prompt using its specific context and provide a structured response.

### IMPORTANT Core Instructions:
1. **Analyze Each Sub-prompt Separately:** For each sub-prompt, provide a detailed analysis using ONLY the context provided for that specific sub-prompt.
2. **Structured Response:** Your response must follow the JSON structure specified below, with an analysis for each sub-prompt.
3. **Direct Answers:** For each sub-prompt, provide a comprehensive analysis that directly answers the question. If the question is not answerable, clearly state this in the analysis.
4. **Exact Supporting Quotes:** For each sub-prompt, include direct, verbatim quotes from its context that support your analysis.
5. **No Cross-Referencing:** Do not use context from one sub-prompt to answer another sub-prompt, even if they seem related.
6. **No Information Found:** If the context for a sub-prompt does not contain information to answer it, clearly state this in the analysis.
7. **Natural Section References:** When referring to where information is found in the document, use natural language references to document sections (e.g., "Section 9 of the Loan Agreement", "Definitions Section", "Article 5", "Schedule A") based on the content of the text excerpts. Do NOT mention chunk IDs or technical identifiers.

### IMPORTANT Formatting Guidelines:
*YOU MUST ALWAYS USE MARKDOWN* in your analysis_summary to improve readability. Apply the following options **only inside the analysis_summary field**; other JSON fields such as supporting_quotes must remain plain strings without Markdown tables or list scaffolding:
- **Bullet points** (using *, -, or +) for concise lists of up to 10 items (e.g., schedule of dates, list of violations, enumerated findings)
- **Numbered lists** (using 1., 2., etc.) for sequential or ordered information such as repayment schedules, payment terms, or fee structures and tables
- **Bold text** (using **text**) to emphasize important information like dates, percentages, amounts, facts, or key definitions
- **Italic text** (using *text*) for subtle emphasis or document references
- **Tables** (Markdown table syntax) only when the provided excerpts contain inherently tabular information that benefits from a structured layout. Keep tables concise (no more than 12 columns or 20 rows), reuse the source column labels when available, never fabricate tables when list formats suffice, and do not place tables in supporting_quotes or other JSON properties.

DO NOT use any of the following:
- Headers (#, ##, etc.) - the UI provides its own section headers
- Code blocks (```) - not applicable for document analysis
- Tables unless the relevant context clearly contains tabular data as described above
- Other complex Markdown elements

### Requesting Additional Context:
- Each excerpt lists a `CHUNK_INDEX`. Treat this as the anchor for neighbouring text.
- You may also request excerpts by citing specific article numbers or section identifiers when the provided context skips critical passages.
- If you require more context, return a JSON payload of the form:
    {
        "needs_additional_context": true,
        "context_requests": [
            {
                "sub_prompt_index": <number>,
                "chunk_indices": [<contiguous chunk indices you need>],
                "article_numbers": ["Article 5"],
                "section_numbers": ["5.2"],
                "section_titles": ["Payment Terms"],
                "reason": "Short explanation."
            }
        ]
    }
- Request only contiguous indices that extend the same passage or precise sections that directly satisfy the question. Do not ask for unrelated sections or retry retrieval.
- You have at most **3** additional context rounds. If told that no more context can be supplied, finalise your analysis with the available excerpts.

### JSON Output Schema:
```json
{
  "analyses": [
    {
      "sub_prompt_index": 1,
      "sub_prompt_title": "Title of the first sub-prompt",
      "sub_prompt_analyzed": "The exact first sub-prompt being analyzed",
      "analysis_summary": "Detailed analysis directly answering the **first sub-prompt**...",
      "supporting_quotes": [
        "Exact quote 1 from the document text for the first sub-prompt...",
        "Exact quote 2, potentially longer..."
      ],
      "analysis_context": "Natural language description of where the information was found (e.g., 'Section 9 of the Loan Agreement', 'Definitions Section', 'Article 5 - Payment Terms')"
    },
    {
      "sub_prompt_index": 2,
      "sub_prompt_title": "Title of the second sub-prompt",
      "sub_prompt_analyzed": "The exact second sub-prompt being analyzed",
      "analysis_summary": "Detailed analysis directly answering the **second sub-prompt**...",
      "supporting_quotes": [
        "Exact quote 1 from the document text for the second sub-prompt...",
        "Exact quote 2, potentially longer..."
      ],
      "analysis_context": "Natural language description of where the information was found (e.g., 'Section 9 of the Loan Agreement', 'Definitions Section', 'Article 5 - Payment Terms')"
    }
    // Additional analyses for each sub-prompt...
  ]
}
```

Your entire response MUST be a single JSON object following this schema and Formatting Guidelines. Do not include any introductory text, explanations, or markdown formatting outside the JSON structure.
"""

            # Create the human prompt with all sub-prompts and their contexts
            toc_section = ""
            if toc_context:
                toc_section = f"\n\nDocument Structure (Table of Contents):\n{toc_context}\n"
            
            human_prompt = f"""Please analyze the following document based on the main prompt and its derived sub-prompts, using the relevant excerpts provided for each sub-prompt.

Document Name:
{filename}

Main Prompt:
{main_prompt}{toc_section}

Sub-prompts and their contexts:
Each excerpt includes a CHUNK_INDEX line so you can reference neighbouring text when absolutely necessary.
"""

            # Add each sub-prompt and its context
            for item in formatted_sub_prompts:
                human_prompt += f"""
--- SUB-PROMPT {item['index']} ---
Title: {item['title']}
Sub-prompt: {item['sub_prompt']}

Relevant Document Excerpts for Sub-prompt {item['index']}:
{item['context']}

"""

            human_prompt += """
Generate a structured analysis for EACH sub-prompt, strictly following the JSON schema and Formatting Guidelines provided in the system instructions. Ensure each analysis only addresses its specific sub-prompt and uses only the context provided for that sub-prompt.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]

            logger.info(f"Sending comprehensive analysis request for {len(formatted_sub_prompts)} sub-prompts in {filename} to AI")
            
            max_context_iterations = 3
            additional_context_rounds = 0
            context_request_history: List[Dict[str, Any]] = []
            conversation_previews: List[str] = []
            forced_completion_issued = False
            total_iterations = 0
            max_total_iterations = max_context_iterations + 4

            with optional_context(
                start_generation(
                    name="analyzer.comprehensive_analysis",
                    input_data={
                        "filename": filename,
                        "num_sub_prompts": len(formatted_sub_prompts),
                        "sub_prompt_titles": [item["title"] for item in formatted_sub_prompts],
                        "main_prompt": main_prompt,
                        "sub_prompts": trace_sub_prompts_payload,
                        "toc_context_preview": _truncate_for_trace(toc_context, limit=600) if toc_context else "",
                    },
                    metadata={
                        "operation": "document_analysis.comprehensive",
                        "num_sub_prompts": len(formatted_sub_prompts),
                    },
                    model=ANALYSIS_MODEL_NAME,
                )
            ) as generation:
                try:
                    parsed_json: Optional[Dict[str, Any]] = None
                    final_raw_response: Optional[str] = None

                    while True:
                        if total_iterations >= max_total_iterations:
                            logger.warning(
                                "Maximum analyzer iterations reached (%d) for %s. Proceeding with available context.",
                                max_total_iterations,
                                filename,
                            )
                            break
                        total_iterations += 1

                        call_result = await self._get_json_with_retries(
                            messages=messages,
                            model_name=ANALYSIS_MODEL_NAME,
                            context=f"comprehensive analysis for {filename}",
                            return_raw=True,
                        )

                        if isinstance(call_result, tuple) and len(call_result) == 2:
                            current_parsed, raw_response = call_result
                        else:
                            current_parsed = call_result
                            raw_response = None

                        if isinstance(raw_response, str):
                            response_text = raw_response
                        else:
                            response_text = json.dumps(current_parsed, indent=2)

                        messages.append({"role": "assistant", "content": response_text})
                        conversation_previews.append(_truncate_for_trace(response_text, limit=600))

                        needs_more, context_requests = self._parse_context_request_payload(current_parsed)

                        if needs_more:
                            logger.info(
                                "LLM requested additional context on iteration %d for %s.",
                                len(context_request_history) + 1,
                                filename,
                            )

                            if forced_completion_issued and (
                                additional_context_rounds >= max_context_iterations or not chunk_lookup
                            ):
                                logger.warning(
                                    "Additional context request ignored after limit for %s; using current payload.",
                                    filename,
                                )
                                parsed_json = current_parsed
                                final_raw_response = response_text
                                break

                            if not context_requests:
                                forced_completion_issued = True
                                clarification_message = (
                                    "No specific chunk indices were provided. Please finalise the analysis with the existing excerpts."
                                )
                                messages.append({"role": "user", "content": clarification_message})
                                context_request_history.append({
                                    "iteration": len(context_request_history) + 1,
                                    "status": "unspecified",
                                    "requests": [],
                                })
                                continue

                            context_message, request_details = self._build_additional_context_response(
                                context_requests,
                                sub_prompts_with_contexts,
                                chunk_lookup,
                                provided_chunk_indices,
                                metadata_index_maps,
                            )
                            provided_any = any(detail.get("provided_indices") for detail in request_details)

                            history_entry: Dict[str, Any] = {
                                "iteration": len(context_request_history) + 1,
                                "status": "pending",
                                "requests": context_requests,
                                "details": request_details,
                            }

                            with optional_context(
                                start_span(
                                    name="analyzer.additional_context_request",
                                    input_data={
                                        "iteration": history_entry["iteration"],
                                        "requests": context_requests,
                                    },
                                    metadata={
                                        "operation": "document_analysis.context_request",
                                        "num_requests": len(context_requests),
                                    },
                                )
                            ) as context_span:
                                set_span_output(
                                    context_span,
                                    output={
                                        "provided_any": provided_any,
                                        "details": request_details,
                                        "message_preview": _truncate_for_trace(context_message, limit=300),
                                    },
                                    metadata={"provided_any": provided_any},
                                )

                            if additional_context_rounds >= max_context_iterations:
                                history_entry["status"] = "limit_reached"
                                context_request_history.append(history_entry)
                                forced_completion_issued = True
                                limit_message = (
                                    "Maximum additional context rounds (3) reached. Please finalise the analysis with the excerpts already supplied."
                                )
                                messages.append({"role": "user", "content": limit_message})
                                continue

                            if not provided_any:
                                history_entry["status"] = "unavailable"
                                context_request_history.append(history_entry)
                                forced_completion_issued = True
                                messages.append({"role": "user", "content": context_message})
                                continue

                            additional_context_rounds += 1
                            history_entry["status"] = "provided"
                            context_request_history.append(history_entry)
                            forced_completion_issued = False
                            messages.append({"role": "user", "content": context_message})
                            continue

                        parsed_json = current_parsed
                        final_raw_response = response_text
                        break

                    if parsed_json is None:
                        raise RuntimeError("Failed to obtain final analysis response from the LLM.")

                    logger.info(
                        "Received comprehensive AI analysis response for %s after %d iteration(s) with %d additional context rounds.",
                        filename,
                        total_iterations,
                        additional_context_rounds,
                    )

                    response_preview = (
                        final_raw_response[:500] if isinstance(final_raw_response, str) else None
                    )
                    context_request_metadata = [
                        {
                            "iteration": entry.get("iteration"),
                            "status": entry.get("status"),
                            "details": [
                                {
                                    "sub_prompt_index": detail.get("sub_prompt_index"),
                                    "requested_indices": detail.get("requested_indices"),
                                    "provided_indices": detail.get("provided_indices"),
                                    "missing_indices": detail.get("missing_indices"),
                                    "article_numbers": detail.get("article_numbers"),
                                    "section_numbers": detail.get("section_numbers"),
                                    "section_titles": detail.get("section_titles"),
                                    "provided_from": detail.get("provided_from"),
                                    "already_from": detail.get("already_from"),
                                    "metadata_missing": detail.get("metadata_missing"),
                                    "truncated": detail.get("truncated"),
                                }
                                for detail in entry.get("details", [])
                            ],
                        }
                        for entry in context_request_history
                    ]
                    set_generation_output(
                        generation,
                        output=parsed_json,
                        metadata={
                            "raw_response_preview": response_preview,
                            "context_request_rounds": additional_context_rounds,
                            "context_request_history": context_request_metadata,
                            "conversation_response_previews": conversation_previews,
                        },
                    )

                except json.JSONDecodeError as json_err:
                    logger.error(
                        "Failed to parse comprehensive AI analysis response as JSON after %d attempt(s): %s",
                        LLM_MAX_RETRIES,
                        json_err,
                    )
                    record_generation_error(
                        generation,
                        json_err,
                        metadata={"stage": "analyzer.comprehensive_analysis_parsing"},
                    )
                    return self._create_fallback_analyses(sub_prompts_with_contexts)
                except Exception as e:
                    logger.error(
                        "Error retrieving comprehensive AI analysis response: %s",
                        e,
                        exc_info=True,
                    )
                    record_generation_error(
                        generation,
                        e,
                        metadata={"stage": "analyzer.comprehensive_analysis"},
                    )
                    return self._create_fallback_analyses(sub_prompts_with_contexts)

            # Validate the response structure
            if not isinstance(parsed_json, dict) or "analyses" not in parsed_json:
                logger.error("Invalid response format: 'analyses' key missing")
                return self._create_fallback_analyses(sub_prompts_with_contexts)

            analyses = parsed_json["analyses"]
            if not isinstance(analyses, list):
                logger.error("Invalid response format: 'analyses' is not a list")
                return self._create_fallback_analyses(sub_prompts_with_contexts)

            # Process each analysis
            results = []
            for analysis in analyses:
                # Basic validation
                if not isinstance(analysis, dict):
                    logger.warning("Skipping invalid analysis entry (not a dict)")
                    continue

                # Extract the analysis data
                sub_prompt_index = analysis.get("sub_prompt_index")
                if sub_prompt_index is None:
                    logger.warning("Analysis missing sub_prompt_index, using position in list")
                    sub_prompt_index = len(results) + 1

                # Find the original sub-prompt data
                original_index = sub_prompt_index - 1
                if 0 <= original_index < len(sub_prompts_with_contexts):
                    original_sub_prompt = sub_prompts_with_contexts[original_index].get('sub_prompt', '')
                    original_title = sub_prompts_with_contexts[original_index].get('title', '')
                else:
                    # This is expected when retrying a single sub-prompt from a multi-prompt analysis
                    # The LLM may return the original index (e.g., 2) but we only have 1 item in the list
                    logger.debug(f"Sub-prompt index {sub_prompt_index} out of range for {len(sub_prompts_with_contexts)} sub-prompts (expected for retry scenarios), using LLM-provided values")
                    original_sub_prompt = analysis.get("sub_prompt_analyzed", "Unknown sub-prompt")
                    original_title = analysis.get("sub_prompt_title", "Unknown title")

                # Create the result in the expected format
                result = {
                    "sub_prompt_analyzed": analysis.get("sub_prompt_analyzed", original_sub_prompt),
                    "analysis_summary": analysis.get("analysis_summary", "No analysis provided"),
                    "supporting_quotes": analysis.get("supporting_quotes", ["No relevant phrase found."]),
                    "analysis_context": analysis.get("analysis_context", ""),
                    "title": analysis.get("sub_prompt_title", original_title)
                }

                # Ensure supporting_quotes is a list
                if not isinstance(result["supporting_quotes"], list):
                    result["supporting_quotes"] = [str(result["supporting_quotes"])]

                # Convert to JSON string for compatibility with existing code
                results.append({
                    "title": result["title"],
                    "sub_prompt": result["sub_prompt_analyzed"],
                    "analysis_json": json.dumps(result, indent=2)
                })

            # Check if we have results for all sub-prompts
            if len(results) < len(sub_prompts_with_contexts):
                logger.warning(f"Missing analyses for some sub-prompts: got {len(results)}, expected {len(sub_prompts_with_contexts)}")
                # Add fallback analyses for missing sub-prompts
                existing_indices = {r.get("sub_prompt") for r in results}
                for i, item in enumerate(sub_prompts_with_contexts):
                    if item.get("sub_prompt") not in existing_indices:
                        fallback = self._create_fallback_analysis(item)
                        results.append(fallback)

            return results

        except Exception as e:
            logger.error(f"Error during comprehensive AI document analysis for {filename}: {str(e)}", exc_info=True)
            return self._create_fallback_analyses(sub_prompts_with_contexts)

    async def analyze_keyword_occurrences(
        self,
        *,
        filename: str,
        main_prompt: str,
        keyword_sections: Dict[str, Dict[str, Any]],
        total_occurrences: int,
        keyword_reasoning: Optional[str] = None,
        user_request_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the analysis model to summarise keyword occurrences and return structured citations.

        If the decomposition captured additional instructions from the user (user_request_context),
        pass them through so the analysis summary can reflect that guidance.
        """

        keyword_payload = []
        for section_key, section in keyword_sections.items():
            keyword = section.get("keyword")
            occurrences = section.get("occurrences", []) or []
            keyword_payload.append({
                "section_key": section_key,
                "keyword": keyword,
                "occurrences": [
                    {
                        "occurrence_id": occ.get("id"),
                        "page_label": occ.get("page_label"),
                        "match_score": occ.get("match_score"),
                        "snippet": occ.get("snippet"),
                    }
                    for occ in occurrences if isinstance(occ, dict)
                ],
            })

        structured_input = {
            "document": filename,
            "user_prompt": main_prompt,
            "keyword_reasoning": keyword_reasoning,
            "total_occurrences": total_occurrences,
            "keywords": keyword_payload,
        }

        if isinstance(user_request_context, str) and user_request_context.strip():
            structured_input["user_request_context"] = user_request_context.strip()

        system_prompt = """You are assisting with keyword-based document analysis. The decomposition model has already determined that keyword mode should run and has supplied exact keyword occurrences (IDs, page information, and snippets).

You MUST follow these requirements:
1. For each keyword, write an "analysis_summary" that references the provided occurrences and mentions the exact count.
2. Produce a "occurrence_citations" list that contains one entry per occurrence. Do not omit or add occurrences.
3. Each citation entry must include:
   - "occurrence_id" (exactly as provided)
   - "page_label" (use the provided number)
   - "match_score" (reuse the provided value)
   - "citation_text" (a concise supporting quote or paraphrase grounded in the provided snippet)
   - "context_note" (short explanation of why this occurrence matters)
4. Do NOT invent new snippets, page numbers, or occurrences. Use only the supplied data.
5. Write clear, professional prose; reference page numbers naturally (e.g., "on page 4").
6. If "user_request_context" is provided, incorporate its guidance into the analysis_summary phrasing while staying grounded in the supplied occurrences.

Return a single JSON object with the structure:
{
  "keywords": [
    {
      "keyword": "...",
      "total_occurrences": <int>,
      "analysis_summary": "...",
      "occurrence_citations": [
        {
          "occurrence_id": "...",
          "page_label": <int>,
          "match_score": <float>,
          "citation_text": "...",
          "context_note": "..."
        }
      ]
    }
  ]
}

Do not include any additional keys and do not wrap the JSON in Markdown fences."""

        additional_context = ""
        if isinstance(user_request_context, str) and user_request_context.strip():
            additional_context = (
                "\n\nAdditional user context to respect in your analysis summary: "
                f"{user_request_context.strip()}"
            )

        human_prompt = (
            "Summarise the provided keyword occurrences. Here is the structured data you must use:\n\n"
            f"{json.dumps(structured_input, indent=2)}"
            f"{additional_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        fallback_result = self._keyword_analysis_fallback(keyword_sections, keyword_reasoning)

        try:
            parsed_json = await self._get_json_with_retries(
                messages=messages,
                model_name=ANALYSIS_MODEL_NAME,
                context=f"keyword analysis for {filename}",
            )
            return self._sanitize_keyword_analysis(parsed_json, keyword_sections, keyword_reasoning)
        except Exception as err:
            logger.error("Keyword mode analysis failed: %s", err, exc_info=True)
            return fallback_result

    def _create_fallback_analyses(self, sub_prompts_with_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback analyses for all sub-prompts when the main analysis fails."""
        return [self._create_fallback_analysis(item) for item in sub_prompts_with_contexts]

    def _create_fallback_analysis(self, sub_prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback analysis for a single sub-prompt."""
        sub_prompt = sub_prompt_data.get("sub_prompt", "Unknown sub-prompt")
        title = sub_prompt_data.get("title", "Unknown title")

        error_response = {
            "sub_prompt_analyzed": sub_prompt,
            "analysis_summary": "An error occurred while analyzing this sub-prompt.",
            "supporting_quotes": ["No relevant phrase found."],
            "analysis_context": "Analysis Error"
        }

        return {
            "title": title,
            "sub_prompt": sub_prompt,
            "analysis_json": json.dumps(error_response, indent=2)
        }

    def _sanitize_keyword_analysis(
        self,
        parsed_json: Dict[str, Any],
        keyword_sections: Dict[str, Dict[str, Any]],
        keyword_reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ensure keyword analysis output matches available occurrences."""

        sanitized_entries: List[Dict[str, Any]] = []

        parsed_lookup: Dict[str, Dict[str, Any]] = {}
        for item in parsed_json.get("keywords", []) or []:
            keyword = item.get("keyword")
            if isinstance(keyword, str) and keyword.strip():
                parsed_lookup[keyword.strip()] = item

        for section_key, section in keyword_sections.items():
            keyword = section.get("keyword")
            occurrences = section.get("occurrences", []) or []
            expected_by_id = {occ.get("id"): occ for occ in occurrences if isinstance(occ, dict) and occ.get("id")}

            parsed_item = parsed_lookup.get(keyword)
            sanitized_citations: List[Dict[str, Any]] = []
            used_occurrence_ids = set()

            if parsed_item:
                for citation in parsed_item.get("occurrence_citations", []) or []:
                    occurrence_id = citation.get("occurrence_id")
                    if occurrence_id not in expected_by_id:
                        continue
                    occ_data = expected_by_id[occurrence_id]
                    sanitized_citations.append({
                        "occurrence_id": occurrence_id,
                        "page_label": occ_data.get("page_label"),
                        "match_score": occ_data.get("match_score", 0.0),
                        "citation_text": citation.get("citation_text") or occ_data.get("snippet"),
                        "context_note": citation.get("context_note", ""),
                        "snippet": occ_data.get("snippet"),
                    })
                    used_occurrence_ids.add(occurrence_id)

            for occurrence_id, occ_data in expected_by_id.items():
                if occurrence_id in used_occurrence_ids:
                    continue
                sanitized_citations.append({
                    "occurrence_id": occurrence_id,
                    "page_label": occ_data.get("page_label"),
                    "match_score": occ_data.get("match_score", 0.0),
                    "citation_text": occ_data.get("snippet"),
                    "context_note": "Automatically generated citation based on provided snippet.",
                    "snippet": occ_data.get("snippet"),
                })

            sanitized_citations.sort(key=lambda entry: entry.get("occurrence_id", ""))

            analysis_summary = None
            if parsed_item and isinstance(parsed_item.get("analysis_summary"), str):
                analysis_summary = parsed_item["analysis_summary"].strip()

            if not analysis_summary:
                analysis_summary = self._build_keyword_summary(keyword, sanitized_citations)

            sanitized_entries.append({
                "section_key": section_key,
                "keyword": keyword,
                "analysis_summary": analysis_summary,
                "occurrence_citations": sanitized_citations,
                "total_occurrences": len(sanitized_citations),
            })

        return {
            "keywords": sanitized_entries,
            "total_occurrences": sum(entry["total_occurrences"] for entry in sanitized_entries),
            "keyword_reasoning": keyword_reasoning,
        }

    def _keyword_analysis_fallback(
        self,
        keyword_sections: Dict[str, Dict[str, Any]],
        keyword_reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        fallback_entries: List[Dict[str, Any]] = []
        for section_key, section in keyword_sections.items():
            keyword = section.get("keyword")
            occurrences = section.get("occurrences", []) or []
            citations = []
            for occ in occurrences:
                if not isinstance(occ, dict):
                    continue
                citations.append({
                    "occurrence_id": occ.get("id"),
                    "page_label": occ.get("page_label"),
                    "match_score": occ.get("match_score", 0.0),
                    "citation_text": occ.get("snippet"),
                    "context_note": "Automatically generated citation based on provided snippet.",
                    "snippet": occ.get("snippet"),
                })

            fallback_entries.append({
                "section_key": section_key,
                "keyword": keyword,
                "analysis_summary": self._build_keyword_summary(keyword, citations),
                "occurrence_citations": citations,
                "total_occurrences": len(citations),
            })

        return {
            "keywords": fallback_entries,
            "total_occurrences": sum(entry["total_occurrences"] for entry in fallback_entries),
            "keyword_reasoning": keyword_reasoning,
        }

    @staticmethod
    def _build_keyword_summary(keyword: Optional[str], citations: List[Dict[str, Any]]) -> str:
        keyword_label = keyword or "the keyword"
        count = len(citations)
        if not citations:
            return f"No occurrences of '{keyword_label}' were detected in the supplied snippets." if keyword else "No keyword occurrences were detected."

        pages = sorted({citation.get("page_label") for citation in citations if isinstance(citation.get("page_label"), (int, float))})
        if pages:
            page_part = "page" if len(pages) == 1 else "pages"
            page_numbers = ", ".join(str(int(p)) for p in pages)
            return f"Found {count} occurrence(s) of '{keyword_label}' on {page_part} {page_numbers}."
        return f"Found {count} occurrence(s) of '{keyword_label}'."