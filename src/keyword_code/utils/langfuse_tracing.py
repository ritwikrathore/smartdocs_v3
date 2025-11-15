"""Utility helpers for Langfuse tracing integration.

These helpers centralize all interactions with the Langfuse Python SDK and
provide graceful fallbacks when the SDK (or required credentials) are not
available.  Use the context managers defined here to wrap long-running
operations, LLM generations, and reranker calls so that the resulting traces are
consistent across the application.
"""

from __future__ import annotations

import logging
import os
from contextlib import ExitStack, contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

try:
    from langfuse import get_client, propagate_attributes

    _LANGFUSE_IMPORT_ERROR: Optional[Exception] = None
except Exception as import_err:  # pragma: no cover - defensive guard
    get_client = None  # type: ignore
    propagate_attributes = None  # type: ignore
    _LANGFUSE_IMPORT_ERROR = import_err


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_tracing_enabled() -> bool:
    """Return True when Langfuse tracing should be active."""

    if get_client is None:
        if _LANGFUSE_IMPORT_ERROR and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Langfuse SDK import failed: %s", _LANGFUSE_IMPORT_ERROR)
        return False

    if not _is_truthy(os.getenv("LANGFUSE_TRACING_ENABLED", "true")):
        return False

    # Require both keys to avoid partial configuration errors.
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.debug("Langfuse tracing disabled: missing API keys in environment")
        return False

    return True


@lru_cache(maxsize=1)
def get_langfuse_client_cached():
    """Return a cached Langfuse client or None when tracing is disabled."""

    if not is_tracing_enabled():
        return None

    try:
        client = get_client()  # type: ignore[misc]
        return client
    except Exception as err:  # pragma: no cover - network/env specific
        logger.warning("Failed to initialize Langfuse client: %s", err)
        return None


def _resolve_client():
    client = get_langfuse_client_cached()
    if client is None and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Langfuse client unavailable; tracing will be a no-op")
    return client


def _normalize_tags(tags: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not tags:
        return None
    normalized: List[str] = []
    for tag in tags:
        if isinstance(tag, str) and tag.strip():
            normalized.append(tag.strip())
    return normalized or None


@contextmanager
def start_trace(
    name: str,
    *,
    input_data: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
):
    """Create a root Langfuse span representing a full document run."""

    client = _resolve_client()
    if client is None:
        yield None
        return

    propagate_kwargs: Dict[str, Any] = {}
    trace_update_kwargs: Dict[str, Any] = {}

    if user_id:
        propagate_kwargs["user_id"] = user_id
        trace_update_kwargs["user_id"] = user_id
    if session_id:
        propagate_kwargs["session_id"] = session_id
        trace_update_kwargs["session_id"] = session_id
    normalized_tags = _normalize_tags(tags)
    if normalized_tags:
        propagate_kwargs["tags"] = normalized_tags
        trace_update_kwargs["tags"] = normalized_tags
    if version:
        propagate_kwargs["version"] = version
        trace_update_kwargs["version"] = version
    if trace_metadata:
        propagate_kwargs["metadata"] = trace_metadata
        trace_update_kwargs["metadata"] = trace_metadata

    with client.start_as_current_span(name=name, input=input_data, metadata=metadata) as span:
        with ExitStack() as exit_stack:
            if propagate_kwargs and propagate_attributes is not None:
                exit_stack.enter_context(propagate_attributes(**propagate_kwargs))

            if trace_update_kwargs:
                try:
                    span.update_trace(**trace_update_kwargs)
                except Exception as err:
                    logger.debug("Unable to update Langfuse trace metadata: %s", err)

            yield span


@contextmanager
def start_span(
    name: str,
    *,
    input_data: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Start a Langfuse span for non-LLM operations (e.g., reranker)."""

    client = _resolve_client()
    if client is None:
        yield None
        return

    with client.start_as_current_span(name=name, input=input_data, metadata=metadata) as span:
        yield span


@contextmanager
def start_generation(
    name: str,
    *,
    input_data: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
):
    """Start a Langfuse generation span for LLM calls."""

    client = _resolve_client()
    if client is None:
        yield None
        return

    with client.start_as_current_generation(
        name=name,
        input=input_data,
        metadata=metadata,
        model=model,
    ) as generation:
        yield generation


def update_current_trace(**kwargs: Any) -> None:
    client = _resolve_client()
    if client is None:
        return
    try:
        client.update_current_trace(**kwargs)
    except Exception as err:
        logger.debug("Failed to update current Langfuse trace: %s", err)


def update_current_span(**kwargs: Any) -> None:
    client = _resolve_client()
    if client is None:
        return
    try:
        client.update_current_span(**kwargs)
    except Exception as err:
        logger.debug("Failed to update current Langfuse span: %s", err)


def update_current_generation(**kwargs: Any) -> None:
    client = _resolve_client()
    if client is None:
        return
    try:
        client.update_current_generation(**kwargs)
    except Exception as err:
        logger.debug("Failed to update current Langfuse generation: %s", err)


def set_generation_output(
    generation: Any,
    *,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> None:
    if generation is None:
        return

    payload: Dict[str, Any] = {}
    if output is not None:
        payload["output"] = output
    if metadata is not None:
        payload["metadata"] = metadata
    if usage is not None:
        payload["usage_details"] = usage

    if not payload:
        return

    try:
        generation.update(**payload)
    except Exception as err:
        logger.debug("Failed to update Langfuse generation output: %s", err)


def set_span_output(
    span: Any,
    *,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if span is None:
        return

    payload: Dict[str, Any] = {}
    if output is not None:
        payload["output"] = output
    if metadata is not None:
        payload["metadata"] = metadata

    if not payload:
        return

    try:
        span.update(**payload)
    except Exception as err:
        logger.debug("Failed to update Langfuse span output: %s", err)


def record_generation_error(generation: Any, error: Exception, *, metadata: Optional[Dict[str, Any]] = None) -> None:
    if generation is None:
        return
    try:
        update_payload: Dict[str, Any] = {
            "level": "ERROR",
            "status_message": str(error),
        }
        if metadata is not None:
            update_payload["metadata"] = metadata
        generation.update(**update_payload)
    except Exception as err:
        logger.debug("Failed to record Langfuse generation error: %s", err)


def record_span_error(span: Any, error: Exception, *, metadata: Optional[Dict[str, Any]] = None) -> None:
    if span is None:
        return
    try:
        update_payload: Dict[str, Any] = {
            "level": "ERROR",
            "status_message": str(error),
        }
        if metadata is not None:
            update_payload["metadata"] = metadata
        span.update(**update_payload)
    except Exception as err:
        logger.debug("Failed to record Langfuse span error: %s", err)


@contextmanager
def optional_context(manager):
    """Return a context manager when provided, otherwise a nullcontext."""

    if manager is None:
        with nullcontext() as ctx:
            yield ctx
    else:
        with manager as ctx:
            yield ctx
