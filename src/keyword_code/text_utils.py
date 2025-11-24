"""Lightweight text helpers used by both chunking and verification modules."""

from typing import Optional, Any, Union, List, Dict, Tuple
import re
import json
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, strip, and squash whitespace for consistent comparisons."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", str(text).lower())
    return normalized.strip()


def sanitize_json_string(json_str: str) -> str:
    """Escape control characters that appear unescaped within JSON strings."""
    sanitized_chars: List[str] = []
    in_string = False
    escape_next = False
    
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
                continue
            if ch == "\r":
                sanitized_chars.append("\\r")
                continue
            if ch == "\t":
                sanitized_chars.append("\\t")
                continue
            if code_point < 0x20:
                sanitized_chars.append(f"\\u{code_point:04x}")
                continue
                
            sanitized_chars.append(ch)
        else:
            if ch == '"':
                in_string = True
            sanitized_chars.append(ch)
            
    return "".join(sanitized_chars)


def clean_and_parse_json(text: str, default: Any = None) -> Any:
    """
    Robustly parse JSON from a string, handling markdown blocks and extra text.
    
    Args:
        text: The string containing JSON
        default: Value to return if parsing fails (default: None)
        
    Returns:
        Parsed JSON object (dict or list) or default value
    """
    if not text:
        return default

    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()

    # 2. Remove markdown code blocks
    # Extract content inside the first code block
    match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        block_content = match.group(1).strip()
        try:
            return json.loads(block_content)
        except json.JSONDecodeError:
            # If parsing content of code block fails, continue to other methods
            # using the content of the block as the new text
            cleaned = block_content

    # 3. Try to find JSON object or array
    # Find first { or [
    start_obj = cleaned.find('{')
    start_arr = cleaned.find('[')
    
    start = -1
    end = -1
    
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        start = start_obj
        end = cleaned.rfind('}')
    elif start_arr != -1:
        start = start_arr
        end = cleaned.rfind(']')
        
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try sanitizing control characters
            try:
                sanitized = sanitize_json_string(candidate)
                return json.loads(sanitized)
            except json.JSONDecodeError:
                pass
            
        # 4. Fallback: Try to fix common issues (like single quotes)
        # This is risky but sometimes helpful for LLM outputs
        try:
            # Replace single quotes with double quotes, but be careful about contractions
            # This is a simple heuristic and might break valid strings containing single quotes
            # So we only try it if everything else failed
            fixed = candidate.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
            
    return default
