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


def render_limited_markdown(text: Optional[str]) -> str:
    """
    Converts limited Markdown formatting to HTML for display in analysis sections.

    Supports:
    - Bullet points (unordered lists using *, -, or +)
    - Numbered lists (ordered lists using 1., 2., etc.)
    - Bold text (**text** or __text__)
    - Italic text (*text* or _text_)

    Does NOT support:
    - Headers (#, ##, etc.)
    - Code blocks (```)
    - Tables
    - Other complex Markdown elements

    Args:
        text: The text containing limited Markdown formatting

    Returns:
        HTML string with Markdown converted to HTML tags
    """
    if not text:
        return ""

    text = str(text)

    # Escape HTML special characters first to prevent XSS
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Split text into lines for list processing FIRST (before processing bold/italic)
    lines = text.split('\n')
    processed_lines = []
    in_ul = False
    in_ol = False

    def process_inline_formatting(line_text: str) -> str:
        """Process bold and italic formatting in a line of text."""
        # Process bold text (**text** or __text__)
        line_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line_text)
        line_text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', line_text)

        # Process italic text (*text* or _text_)
        # Make sure we don't match bold patterns or list markers
        line_text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', line_text)
        line_text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', line_text)

        return line_text

    for line in lines:
        stripped = line.strip()

        # Check for unordered list items (*, -, +)
        ul_match = re.match(r'^([\*\-\+])\s+(.+)$', stripped)
        if ul_match:
            if not in_ul:
                processed_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_ul = True
            if in_ol:
                processed_lines.append('</ol>')
                in_ol = False
            # Process inline formatting in the list item content
            list_content = process_inline_formatting(ul_match.group(2))
            processed_lines.append(f'<li>{list_content}</li>')
            continue

        # Check for ordered list items (1., 2., etc.)
        ol_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if ol_match:
            if not in_ol:
                processed_lines.append('<ol style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_ol = True
            if in_ul:
                processed_lines.append('</ul>')
                in_ul = False
            # Process inline formatting in the list item content
            list_content = process_inline_formatting(ol_match.group(2))
            processed_lines.append(f'<li>{list_content}</li>')
            continue

        # Not a list item - close any open lists
        if in_ul:
            processed_lines.append('</ul>')
            in_ul = False
        if in_ol:
            processed_lines.append('</ol>')
            in_ol = False

        # Add the line with inline formatting processed
        if stripped:  # Only add non-empty lines
            processed_lines.append(process_inline_formatting(line))
        else:
            processed_lines.append('<br>')  # Preserve blank lines as line breaks

    # Close any remaining open lists
    if in_ul:
        processed_lines.append('</ul>')
    if in_ol:
        processed_lines.append('</ol>')

    # Join lines back together
    result = '\n'.join(processed_lines)

    return result


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
