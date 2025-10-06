"""
General utility functions for the keyword_code package.
"""

import base64
import re
import threading
from typing import Optional


def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison: lowercase, strip, whitespace."""
    if not text:
        return ""
    text = str(text)
    text = text.lower()  # Case-insensitive matching
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()


def remove_markdown_formatting(text: Optional[str]) -> str:
    """Removes common markdown formatting and standardizes quotation marks."""
    if not text:
        return ""
    text = str(text)
    # Basic bold, italics, code
    text = re.sub(r"\*(\*|_)(.*?)\1\*?", r"\2", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Basic headings, blockquotes, lists
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Standardize quotation marks - replace all types of quotes with a space
    # This helps with verification when quotes are inconsistently used
    text = re.sub(r'[\'"""'']', ' ', text)
    return text.strip()


def get_base64_encoded_image(image_path: str) -> Optional[str]:
    """Get base64 encoded image."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        from ..config import logger
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None


# Thread-safe counter class
class Counter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

    def value(self):
        with self._lock:
            return self._value
