"""
Memory monitoring utilities for the application.
"""

import os
import psutil
import gc
import logging
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from ..config import logger

# Default memory thresholds (in percentage)
DEFAULT_WARNING_THRESHOLD = 75.0  # 75% memory usage triggers a warning
DEFAULT_CRITICAL_THRESHOLD = 90.0  # 90% memory usage triggers cleanup

# Global flag to track if memory monitoring is enabled
_MEMORY_MONITORING_ENABLED = True


def enable_memory_monitoring():
    """Enable memory monitoring."""
    global _MEMORY_MONITORING_ENABLED
    _MEMORY_MONITORING_ENABLED = True


def disable_memory_monitoring():
    """Disable memory monitoring."""
    global _MEMORY_MONITORING_ENABLED
    _MEMORY_MONITORING_ENABLED = False


def is_memory_monitoring_enabled() -> bool:
    """Check if memory monitoring is enabled."""
    return _MEMORY_MONITORING_ENABLED


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage information.
    
    Returns:
        Dict with memory usage information:
            - percent: Percentage of memory used
            - used: Memory used in bytes
            - available: Memory available in bytes
            - total: Total memory in bytes
    """
    try:
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss  # Resident Set Size in bytes
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        return {
            "percent": system_memory.percent,
            "used": system_memory.used,
            "available": system_memory.available,
            "total": system_memory.total,
            "process_memory": process_memory,
            "process_percent": (process_memory / system_memory.total) * 100
        }
    except Exception as e:
        logger.error(f"Error getting memory usage: {str(e)}")
        return {
            "percent": 0,
            "used": 0,
            "available": 0,
            "total": 0,
            "process_memory": 0,
            "process_percent": 0
        }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to a human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def check_memory_usage(
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
) -> Tuple[bool, str]:
    """
    Check memory usage and return status.
    
    Args:
        warning_threshold: Percentage threshold for warning
        critical_threshold: Percentage threshold for critical
        
    Returns:
        Tuple of (is_critical, message)
    """
    if not _MEMORY_MONITORING_ENABLED:
        return False, "Memory monitoring disabled"
    
    memory_info = get_memory_usage()
    percent = memory_info["percent"]
    process_percent = memory_info["process_percent"]
    
    message = (
        f"Memory usage: {percent:.1f}% of system memory, "
        f"Process: {format_bytes(memory_info['process_memory'])} "
        f"({process_percent:.1f}% of total)"
    )
    
    if percent >= critical_threshold:
        return True, f"CRITICAL: {message}"
    elif percent >= warning_threshold:
        return False, f"WARNING: {message}"
    else:
        return False, f"OK: {message}"


def cleanup_memory(force: bool = False) -> Dict[str, Any]:
    """
    Perform memory cleanup operations.
    
    Args:
        force: Force cleanup even if memory usage is below thresholds
        
    Returns:
        Dict with cleanup results
    """
    if not _MEMORY_MONITORING_ENABLED and not force:
        return {"status": "skipped", "reason": "Memory monitoring disabled"}
    
    before_memory = get_memory_usage()
    is_critical, _ = check_memory_usage()
    
    if not is_critical and not force:
        return {
            "status": "skipped", 
            "reason": "Memory usage below critical threshold",
            "before": before_memory
        }
    
    # Perform cleanup operations
    cleanup_operations = []
    
    # 1. Run garbage collection
    gc.collect()
    cleanup_operations.append("Garbage collection")
    
    # 2. Clear any large objects in session state that can be regenerated
    large_objects_cleared = _clear_large_session_objects()
    if large_objects_cleared:
        cleanup_operations.append(f"Cleared {large_objects_cleared} large session objects")
    
    # Get memory usage after cleanup
    after_memory = get_memory_usage()
    memory_freed = before_memory["used"] - after_memory["used"]
    
    return {
        "status": "completed",
        "operations": cleanup_operations,
        "before": before_memory,
        "after": after_memory,
        "freed_bytes": memory_freed,
        "freed_formatted": format_bytes(memory_freed)
    }


def _clear_large_session_objects() -> int:
    """
    Clear large objects from session state that can be regenerated.
    
    Returns:
        Number of objects cleared
    """
    if not hasattr(st, "session_state"):
        return 0
    
    # List of keys for large objects that can be safely cleared
    large_object_keys = [
        "chunk_embeddings",
        "original_bytes"
    ]
    
    cleared_count = 0
    
    # Check preprocessed_data for large objects
    if "preprocessed_data" in st.session_state and isinstance(st.session_state.preprocessed_data, dict):
        for filename, data in st.session_state.preprocessed_data.items():
            if isinstance(data, dict):
                for key in large_object_keys:
                    if key in data:
                        data[key] = None
                        cleared_count += 1
    
    return cleared_count


def monitor_memory_usage(
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    auto_cleanup: bool = True
) -> Dict[str, Any]:
    """
    Monitor memory usage and perform cleanup if necessary.
    
    Args:
        warning_threshold: Percentage threshold for warning
        critical_threshold: Percentage threshold for critical
        auto_cleanup: Whether to automatically clean up memory if critical
        
    Returns:
        Dict with monitoring results
    """
    if not _MEMORY_MONITORING_ENABLED:
        return {"status": "skipped", "reason": "Memory monitoring disabled"}
    
    is_critical, message = check_memory_usage(warning_threshold, critical_threshold)
    result = {
        "status": "critical" if is_critical else "ok",
        "message": message,
        "memory_info": get_memory_usage()
    }
    
    if is_critical and auto_cleanup:
        cleanup_result = cleanup_memory(force=True)
        result["cleanup"] = cleanup_result
    
    return result
