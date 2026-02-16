# Material Icons Font Files

This directory contains locally hosted Material Icons font files to support airgapped deployments.

## Font Files Required

Download the following files from Google Fonts and place them in this directory:

1. `MaterialIcons-Regular.woff2` - Main font file
2. `MaterialIconsOutlined-Regular.woff2` - Outlined variant (if needed)

## How to Obtain Font Files

### Option 1: Direct Download
Visit https://github.com/google/material-design-icons/tree/master/font and download:
- `MaterialIcons-Regular.woff2`
- `MaterialIconsOutlined-Regular.woff2`

### Option 2: Google Fonts Download
1. Visit https://fonts.google.com/icons
2. Download the Material Icons font package
3. Extract the `.woff2` files to this directory

## Usage

The application automatically loads these fonts via CSS in `utils/ui_helpers.py` when available.
If files are missing, the app will fall back to Google CDN (which won't work in airgapped environments).
