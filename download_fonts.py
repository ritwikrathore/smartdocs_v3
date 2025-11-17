"""
Download Material Icons font files for offline/airgapped use.
Run this script to download the required font files.
"""
import urllib.request
import os

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

FONTS = {
    "MaterialIcons-Regular.woff2": "https://fonts.gstatic.com/s/materialicons/v140/flUhRq6tzZclQEJ-Vdg-IuiaDsNc.woff2",
    "MaterialIconsOutlined-Regular.woff2": "https://fonts.gstatic.com/s/materialiconsoutlined/v109/gok-H7zzDkdnRel8-DQ6KAXJ69wP1tGnf4ZGhUce.woff2"
}

for filename, url in FONTS.items():
    filepath = os.path.join(FONT_DIR, filename)
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Successfully downloaded {filename}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

print(f"\nFont files saved to: {FONT_DIR}")
