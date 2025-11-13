"""
General utility functions for the keyword_code package.
"""

import base64
import re
import threading
from typing import List, Optional


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
    - Tables (Markdown tables that include header and alignment rows)

    Does NOT support:
    - Headers (#, ##, etc.)
    - Code blocks (```)
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

    lines = text.split('\n')
    processed_lines: List[str] = []
    in_ul = False
    in_ol = False

    def process_inline_formatting(line_text: str) -> str:
        """Process bold and italic formatting in a line of text."""
        line_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line_text)
        line_text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', line_text)
        line_text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', line_text)
        line_text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', line_text)
        return line_text

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            processed_lines.append('</ul>')
            in_ul = False
        if in_ol:
            processed_lines.append('</ol>')
            in_ol = False

    def split_table_row(row_text: str) -> List[str]:
        row_text = row_text.strip()
        if row_text.startswith("|"):
            row_text = row_text[1:]
        if row_text.endswith("|"):
            row_text = row_text[:-1]
        return [cell.strip() for cell in row_text.split("|")]

    def is_table_separator(row_text: str) -> bool:
        stripped = row_text.strip()
        if not stripped or "|" not in stripped:
            return False
        stripped = stripped.strip("|")
        segments = stripped.split("|")
        if not segments:
            return False
        for segment in segments:
            segment = segment.strip()
            if not segment:
                return False
            if not re.fullmatch(r":?-{3,}:?", segment):
                return False
        return True

    def is_table_header(row_text: str) -> bool:
        stripped = row_text.strip()
        if "|" not in stripped:
            return False
        cells = split_table_row(stripped)
        return len(cells) >= 2

    def is_table_row(row_text: str) -> bool:
        stripped = row_text.strip()
        return bool(stripped) and "|" in stripped

    def infer_alignment(spec: str) -> str:
        spec = spec.strip()
        left = spec.startswith(":")
        right = spec.endswith(":")
        if left and right:
            return "center"
        if right:
            return "right"
        return "left"

    def convert_markdown_table(table_lines: List[str]) -> str:
        header_cells = split_table_row(table_lines[0]) if table_lines else []
        if not header_cells:
            return ""
        alignment_cells = split_table_row(table_lines[1]) if len(table_lines) > 1 else []
        alignments = [infer_alignment(cell) for cell in alignment_cells]
        header_cells = [process_inline_formatting(cell) for cell in header_cells]
        target_len = len(header_cells)
        if len(alignments) < target_len:
            alignments.extend(["left"] * (target_len - len(alignments)))
        elif len(alignments) > target_len:
            alignments = alignments[:target_len]

        body_rows: List[List[str]] = []
        for raw_row in table_lines[2:]:
            if is_table_separator(raw_row):
                continue
            if not is_table_row(raw_row):
                continue
            cells = split_table_row(raw_row)
            if len(cells) < target_len:
                cells.extend([""] * (target_len - len(cells)))
            elif len(cells) > target_len:
                cells = cells[:target_len]
            body_rows.append([process_inline_formatting(cell) for cell in cells])

        table_parts: List[str] = [
            '<div class="analysis-table-wrapper" style="overflow-x:auto; margin: 0.75rem 0;">',
            '<table style="border-collapse: collapse; width: 100%; font-size: 0.95rem;">',
            '<thead><tr>'
        ]
        for idx, header in enumerate(header_cells):
            alignment = alignments[idx] if idx < len(alignments) else "left"
            table_parts.append(
                f'<th style="border: 1px solid #d6d6d6; padding: 0.5rem; background-color: #f0f4ff; text-align: {alignment};">{header}</th>'
            )
        table_parts.append('</tr></thead>')
        table_parts.append('<tbody>')
        for row in body_rows:
            table_parts.append('<tr>')
            for idx, cell in enumerate(row):
                alignment = alignments[idx] if idx < len(alignments) else "left"
                table_parts.append(
                    f'<td style="border: 1px solid #e0e0e0; padding: 0.45rem; text-align: {alignment};">{cell}</td>'
                )
            table_parts.append('</tr>')
        table_parts.append('</tbody></table></div>')
        return ''.join(table_parts)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped and is_table_header(line) and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            close_lists()
            table_block = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and is_table_row(lines[i]):
                if not lines[i].strip():
                    break
                table_block.append(lines[i])
                i += 1
            table_html = convert_markdown_table(table_block)
            if table_html:
                processed_lines.append(table_html)
            continue

        ul_match = re.match(r'^([\*\-\+])\s+(.+)$', stripped)
        if ul_match:
            if not in_ul:
                close_lists()
                processed_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_ul = True
            if in_ol:
                processed_lines.append('</ol>')
                in_ol = False
            list_content = process_inline_formatting(ul_match.group(2))
            processed_lines.append(f'<li>{list_content}</li>')
            i += 1
            continue

        ol_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if ol_match:
            if not in_ol:
                close_lists()
                processed_lines.append('<ol style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_ol = True
            if in_ul:
                processed_lines.append('</ul>')
                in_ul = False
            list_content = process_inline_formatting(ol_match.group(2))
            processed_lines.append(f'<li>{list_content}</li>')
            i += 1
            continue

        close_lists()
        if stripped:
            processed_lines.append(process_inline_formatting(line))
        else:
            processed_lines.append('<br>')
        i += 1

    close_lists()
    return '\n'.join(processed_lines)


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
