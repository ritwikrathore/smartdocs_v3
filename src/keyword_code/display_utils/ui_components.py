"""
UI components and utilities for display.
"""

import base64
from pathlib import Path
from typing import Optional
from ..config import logger


def get_base64_encoded_image(image_path: Path) -> Optional[str]:
    """Get base64 encoded image."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None


# --- Define check_img for use in UI elements ---
check_img = "✅"  # Default to emoji
try:
    # Path to the assets directory from src/keyword_code/display_utils/
    assets_path = Path(__file__).parent.parent / "assets"
    correct_png_path = assets_path / "correct.png"
    logger.info(f"Attempting to load check icon from: {correct_png_path}")
    if correct_png_path.is_file():
        check_base64 = get_base64_encoded_image(correct_png_path)
        if check_base64:
            check_img = f'<img src="data:image/png;base64,{check_base64}" style="width: 18px; height: 18px; vertical-align: middle; margin-right: 5px;" alt="✓">'
        else:
            logger.warning(f"Failed to encode check icon: {correct_png_path}")
    else:
        logger.warning(f"Check icon not found at: {correct_png_path}")
except Exception as img_e:
    logger.warning(f"Could not load check icon, using emoji fallback: {img_e}")

