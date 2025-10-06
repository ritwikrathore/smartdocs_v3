"""
File management utilities for handling temporary files and cleanup.
"""

import os
import shutil
import tempfile
import atexit
import logging
import time
from pathlib import Path
from typing import Optional, List, Set, Dict, Any
import streamlit as st
from ..config import logger

# Global registry of temporary files and directories
_TEMP_FILES: Set[str] = set()
_TEMP_DIRS: Set[str] = set()

# Create tmp directory if it doesn't exist
TMP_DIR = Path("tmp")
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True)
    logger.info(f"Created temporary directory: {TMP_DIR.absolute()}")

# Track last access time for each session
_SESSION_LAST_ACCESS: Dict[str, float] = {}
# Session timeout in seconds (90 minutes)
SESSION_TIMEOUT = 5400
# File age threshold for cleanup (24 hours)
FILE_AGE_THRESHOLD = 86400


def get_temp_dir() -> Path:
    """Get the temporary directory path."""
    return TMP_DIR


def create_temp_file(prefix: str = "", suffix: str = "", delete: bool = True) -> str:
    """
    Create a temporary file and return its path.

    Args:
        prefix: Prefix for the temporary file name
        suffix: Suffix for the temporary file name
        delete: Whether to register the file for deletion

    Returns:
        Path to the temporary file
    """
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=TMP_DIR)
    os.close(fd)

    if delete:
        register_temp_file(temp_path)

    return temp_path


def create_temp_dir(prefix: str = "", suffix: str = "", delete: bool = True) -> str:
    """
    Create a temporary directory and return its path.

    Args:
        prefix: Prefix for the temporary directory name
        suffix: Suffix for the temporary directory name
        delete: Whether to register the directory for deletion

    Returns:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=TMP_DIR)

    if delete:
        register_temp_dir(temp_dir)

    return temp_dir


def register_temp_file(file_path: str) -> None:
    """
    Register a temporary file for deletion.

    Args:
        file_path: Path to the temporary file
    """
    _TEMP_FILES.add(file_path)
    logger.debug(f"Registered temporary file: {file_path}")


def register_temp_dir(dir_path: str) -> None:
    """
    Register a temporary directory for deletion.

    Args:
        dir_path: Path to the temporary directory
    """
    _TEMP_DIRS.add(dir_path)
    logger.debug(f"Registered temporary directory: {dir_path}")


def remove_temp_file(file_path: str) -> bool:
    """
    Remove a temporary file.

    Args:
        file_path: Path to the temporary file

    Returns:
        True if the file was removed, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")

        _TEMP_FILES.discard(file_path)
        return True
    except Exception as e:
        logger.error(f"Error removing temporary file {file_path}: {str(e)}")
        return False


def remove_temp_dir(dir_path: str) -> bool:
    """
    Remove a temporary directory.

    Args:
        dir_path: Path to the temporary directory

    Returns:
        True if the directory was removed, False otherwise
    """
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.debug(f"Removed temporary directory: {dir_path}")

        _TEMP_DIRS.discard(dir_path)
        return True
    except Exception as e:
        logger.error(f"Error removing temporary directory {dir_path}: {str(e)}")
        return False


def cleanup_session_files(session_id: Optional[str] = None) -> None:
    """
    Clean up temporary files for a specific session or all sessions.

    Args:
        session_id: Session ID to clean up, or None to clean up all sessions
    """
    if session_id:
        # Clean up files for a specific session
        session_prefix = f"session_{session_id}_"
        files_to_remove = [f for f in _TEMP_FILES if os.path.basename(f).startswith(session_prefix)]
        dirs_to_remove = [d for d in _TEMP_DIRS if os.path.basename(d).startswith(session_prefix)]

        for file_path in files_to_remove:
            remove_temp_file(file_path)

        for dir_path in dirs_to_remove:
            remove_temp_dir(dir_path)

        logger.info(f"Cleaned up {len(files_to_remove)} files and {len(dirs_to_remove)} directories for session {session_id}")
    else:
        # Clean up all temporary files and directories
        cleanup_all_temp_files()


def cleanup_all_temp_files() -> None:
    """Clean up all temporary files and directories."""
    # Make a copy of the sets to avoid modification during iteration
    temp_files = _TEMP_FILES.copy()
    temp_dirs = _TEMP_DIRS.copy()

    for file_path in temp_files:
        remove_temp_file(file_path)

    for dir_path in temp_dirs:
        remove_temp_dir(dir_path)

    logger.info(f"Cleaned up {len(temp_files)} temporary files and {len(temp_dirs)} temporary directories")


def update_session_access(session_id: str) -> None:
    """
    Update the last access time for a session.

    Args:
        session_id: Session ID
    """
    _SESSION_LAST_ACCESS[session_id] = time.time()


def cleanup_expired_sessions() -> None:
    """Clean up files for expired sessions."""
    current_time = time.time()
    expired_sessions = []

    for session_id, last_access in _SESSION_LAST_ACCESS.items():
        if current_time - last_access > SESSION_TIMEOUT:
            cleanup_session_files(session_id)
            expired_sessions.append(session_id)

    # Remove expired sessions from the dictionary
    for session_id in expired_sessions:
        _SESSION_LAST_ACCESS.pop(session_id, None)

    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


def get_session_id() -> str:
    """
    Get the current Streamlit session ID.

    Returns:
        Session ID or a default value if not available
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
        return "default_session"
    except Exception:
        return "default_session"


def create_session_temp_file(prefix: str = "", suffix: str = "", delete: bool = True) -> str:
    """
    Create a temporary file for the current session.

    Args:
        prefix: Prefix for the temporary file name
        suffix: Suffix for the temporary file name
        delete: Whether to register the file for deletion

    Returns:
        Path to the temporary file
    """
    session_id = get_session_id()
    session_prefix = f"session_{session_id}_{prefix}"
    update_session_access(session_id)
    return create_temp_file(prefix=session_prefix, suffix=suffix, delete=delete)


def cleanup_old_files():
    """
    Clean up files in the temporary directory that are older than the threshold,
    regardless of whether they're registered in our tracking system.
    This helps catch any files that might have been missed by the regular cleanup.
    """
    if not TMP_DIR.exists():
        return

    current_time = time.time()
    old_files_count = 0
    old_dirs_count = 0

    try:
        # Clean up old files
        for file_path in TMP_DIR.glob('*'):
            try:
                if file_path.is_file():
                    # Check file age
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > FILE_AGE_THRESHOLD:
                        os.remove(file_path)
                        old_files_count += 1
                        # Also remove from tracking if it's there
                        _TEMP_FILES.discard(str(file_path))
                elif file_path.is_dir():
                    # Check directory age (based on modification time)
                    dir_age = current_time - os.path.getmtime(file_path)
                    if dir_age > FILE_AGE_THRESHOLD:
                        shutil.rmtree(file_path, ignore_errors=True)
                        old_dirs_count += 1
                        # Also remove from tracking if it's there
                        _TEMP_DIRS.discard(str(file_path))
            except Exception as e:
                logger.error(f"Error cleaning up old file/directory {file_path}: {str(e)}")

        if old_files_count > 0 or old_dirs_count > 0:
            logger.info(f"Cleaned up {old_files_count} old files and {old_dirs_count} old directories based on age")
    except Exception as e:
        logger.error(f"Error during old files cleanup: {str(e)}")


# Register cleanup function to run at exit
@atexit.register
def _cleanup_at_exit():
    """Clean up all temporary files and directories when the application exits."""
    # First clean up registered files
    cleanup_all_temp_files()

    # Then clean up any old files that might have been missed
    cleanup_old_files()

    # Also try to remove the tmp directory itself if it's empty
    try:
        if TMP_DIR.exists() and not os.listdir(TMP_DIR):
            TMP_DIR.rmdir()
            logger.info(f"Removed empty temporary directory: {TMP_DIR.absolute()}")
    except Exception as e:
        logger.error(f"Error removing temporary directory {TMP_DIR}: {str(e)}")
