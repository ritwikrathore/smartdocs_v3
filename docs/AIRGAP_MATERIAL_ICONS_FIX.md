# Material Icons Fix for Airgapped Azure Deployment

## Problem
In airgapped Azure environments, Streamlit's Material Icons fail to load from Google's CDN, causing:
- Expander arrow icons appearing as text (e.g., `keyboard_double_arrow_right`)
- Missing UI icons throughout the application

## Solution Implemented
We've implemented a **local Material Symbols Rounded font** solution that embeds the font files directly into the application CSS, eliminating external CDN dependencies.

**IMPORTANT**: Streamlit 1.44+ uses **"Material Symbols Rounded"** font (not "Material Icons").

**UPDATE for Streamlit 1.49+**: Streamlit now bundles the Material Symbols Rounded font locally (no Google CDN). However, the font may still fail to load in restricted environments due to CSP policies, file access issues, or other infrastructure constraints. The CSS failsafes remain critical.

## Files Added/Modified

### 1. Font Files (Already Downloaded)
```
src/keyword_code/assets/fonts/
├── MaterialSymbolsRounded.woff2         # PRIMARY: Material Symbols Rounded (for Streamlit 1.44+)
├── MaterialIcons-Regular.woff2          # Legacy fallback: Material Icons font
├── MaterialIconsOutlined-Regular.woff2  # Legacy fallback: Outlined variant
└── README.md                            # Documentation
```

### 2. Modified Code
- **`src/keyword_code/utils/ui_helpers.py`**: Updated `apply_ui_styling()` function to:
  - Load **Material Symbols Rounded** font (primary, for Streamlit 1.44+)
  - Load legacy Material Icons fonts (fallback)
  - Base64-encode fonts for embedding in CSS
  - Inject `@font-face` rules with embedded font data
  - Apply font styling to `[data-testid="stIconMaterial"]` elements
  - **Include CSS failsafes** to hide icon text if fonts fail to render

## How It Works
1. On app startup, `apply_ui_styling()` reads the `.woff2` font files
2. Fonts are base64-encoded and embedded directly in CSS via `data:` URIs
3. CSS rules override Streamlit's default font loading behavior
4. All Material Symbols render from local embedded fonts, not CDN
5. **CSS failsafe**: If fonts fail to load, CSS hides the icon text (e.g., `keyboard_double_arrow_right`) by:
   - Setting `max-width: 24px` and `overflow: hidden` on icon elements
   - Targeting expander arrows specifically for aggressive suppression

## Testing Locally
Run the test script to verify icons display correctly:
```bash
python -m streamlit run test_material_icons.py
```

Expected behavior:
- ✅ Expander arrows display as icons (not text)
- ✅ All UI icons render properly
- ✅ No console errors about missing fonts

## Deployment to Azure

### Step 1: Ensure Font Files Are Included
Verify these files exist in your deployment package:
```
src/keyword_code/assets/fonts/MaterialIcons-Regular.woff2
src/keyword_code/assets/fonts/MaterialIconsOutlined-Regular.woff2
```

### Step 2: No Additional Configuration Needed
The fix is automatic when the app loads. The `apply_ui_styling()` function is called in:
- `pages/1_📄_CNT_space.py` (line 33)
- `src/keyword_code/app.py` (line 1062)

### Step 3: Verify in Azure
After deployment, check browser console (F12) for:
- ❌ **Before fix**: `Failed to load resource: net::ERR_INTERNET_DISCONNECTED` for fonts.gstatic.com
- ✅ **After fix**: No font-related errors

## Fallback Behavior (Multi-Layer Defense)

### Layer 0: Streamlit Bundled Font (1.49+)
- Streamlit 1.49+ bundles `MaterialSymbols-Rounded.woff2` locally
- No Google CDN dependency
- May still fail in restricted environments

### Layer 1: Our Embedded Font (Material Symbols Rounded)
- Additional embedded font as base64 data URI
- Provides redundancy if Streamlit's bundled font fails

### Layer 2: Legacy Fonts (Material Icons)
- Fallback for older Streamlit versions (<1.44)
- Also embedded as base64 data URIs

### Layer 3: CSS Text Hiding (CRITICAL)
If all fonts fail to load or render:
- CSS rules constrain icon elements: `max-width: 28px` + `overflow: hidden`
- Prevents text like `keyboard_double_arrow_right` from being visible
- Targets `[data-testid="stIconMaterial"]` elements

### Layer 4: Aggressive Expander Fix
- Expander arrows are the most visible failure case
- Extra CSS rules target `.stExpander`, `details summary`, `[class*="stExpander"]`
- Forces fixed width of 24px to completely hide any icon text

### What You'll See
| Scenario | Result |
|----------|--------|
| Streamlit 1.49+ bundled font works | ✅ Normal icons (no patch needed) |
| Bundled font fails, our embedded works | ✅ Normal icons |
| All fonts fail | ✅ Empty space (text hidden by CSS) |
| Legacy Streamlit (<1.44) | ✅ Fallback to Material Icons font |

### Logging
- Success: `Successfully loaded Material Symbols Rounded font (X KB encoded)`
- Warning: `Material Symbols Rounded font file not found at: {path}`
- Fallback active: `No Material fonts loaded - fallback text hiding applied`

### Toast Notifications & Error Codes
When the CNT space page loads, a toast appears briefly showing the UI patch status. These codes help diagnose issues in production without checking logs.

#### Success Codes (2xx)
| Code | Toast Message | Meaning |
|------|---------------|---------|
| UI-200 | Primary font loaded | Material Symbols Rounded loaded successfully (ideal state) |
| UI-201 | Legacy font fallback | Material Icons loaded as fallback (older Streamlit) |
| UI-202 | Font not readable | Font file exists but cannot be read (permission issue) |
| UI-203 | CSS failsafe active | No fonts available, CSS text-hiding is active |
| UI-204 | Font check error | Error occurred while checking font status |

#### Error Codes (1xx)
| Code | Toast Message | Meaning | Action |
|------|---------------|---------|--------|
| UI-100 | Styling error | Unexpected error in CSS generation | Check logs for stack trace |
| UI-101 | CSS generation failed | `_get_ui_css()` returned empty | Check font paths and file integrity |
| UI-102 | Font file missing | Font file not found at expected path | Verify font files in deployment |
| UI-103 | Font file permission denied | Cannot read font file (access denied) | Check file permissions in Azure |
| UI-104 | Memory error loading fonts | Out of memory during base64 encoding | Reduce font file size or increase memory |

#### Expected Behavior
- **Normal operation**: You should see `UI-200: Primary font loaded`
- **Fallback working**: `UI-201` or `UI-203` means fallback is active but icons should still work
- **Errors (1xx)**: Indicates a problem that needs investigation

#### Log Correlation
All codes are also logged with prefix `UI Patch Error` or `UI Patch Warning` for log searching:
```bash
grep "UI Patch" logs/app_*.log
```

### Saved Prompts (Pills) Toast Codes
When a user clicks on a saved prompt pill, a toast notification appears confirming the action.

#### Success Codes (2xx)
| Code | Toast Message | Meaning |
|------|---------------|---------|
| PILL-200 | Loaded '{pill_name}' | Saved prompt successfully loaded into text area |
| PILL-201 | (logged only) | Prompt selection cleared (no toast shown) |

#### Error Codes (1xx)
| Code | Toast Message | Meaning | Action |
|------|---------------|---------|--------|
| PILL-100 | Failed to load prompt | Unexpected error loading saved prompt | Check logs for stack trace |
| PILL-101 | Prompt not found | Pill label not found in suggestion list | Verify prompt configuration |

#### Expected Behavior
- **Normal operation**: You should see `PILL-200: Loaded '{prompt_name}'` when clicking a pill
- **Errors (1xx)**: Indicates a configuration or code issue that needs investigation

#### Log Correlation
All pill codes are also logged for searching:
```bash
grep "PILL-" logs/app_*.log
```

## File Size Impact
- MaterialSymbolsRounded.woff2: ~291 KB (primary font)
- MaterialIcons-Regular.woff2: ~125 KB (legacy fallback)
- MaterialIconsOutlined-Regular.woff2: ~152 KB (legacy fallback)
- Total: ~568 KB added to deployment

Note: All three fonts are base64-encoded, adding ~33% overhead when embedded in CSS.

## Additional Airgap Considerations

### Langfuse Tracing (Optional Observability Tool)
Langfuse is an optional observability tool used in development. **It is NOT required for production.**

To disable Langfuse tracing in your Azure deployment:

**Option 1: Environment Variable (Recommended)**
```bash
ENABLE_LANGFUSE_TRACING=false
```

**Option 2: Remove Langfuse Credentials**
Simply omit these environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL`

The application will automatically detect the absence of Langfuse and continue running normally with no errors.

**What happens when Langfuse is disabled:**
- ✅ Application runs normally
- ✅ No errors or warnings about missing Langfuse
- ✅ All functionality works as expected
- ℹ️ Observability traces are not collected (only affects debugging/monitoring)

### Content Security Policy (CSP)
If you have CSP headers configured, ensure they allow:
```
font-src 'self' data:;
```

### Streamlit Configuration
Create `.streamlit/config.toml` in project root if needed:
```toml
[server]
enableCORS = false
enableXsrfProtection = true
enableWebsocketCompression = true

[browser]
gatherUsageStats = false
```

### Other Potential CDN Dependencies
If you encounter other missing resources in Azure, check for:
1. **Google Fonts** - Any custom fonts loaded from fonts.googleapis.com
2. **Streamlit Assets** - Ensure `streamlit` package includes all static assets
3. **Python Package Indices** - Verify all pip packages are pre-installed (no runtime downloads)

## Troubleshooting

### Icons Still Display as Text (e.g., `keyboard_double_arrow_right`)
This should no longer happen due to CSS failsafes, but if it does:

1. **Verify the new font file exists**: Check that `MaterialSymbolsRounded.woff2` is in the deployment
2. **Clear Streamlit cache**: The CSS is cached - restart the app or clear cache
3. **Check browser console (F12)** for JavaScript errors
4. **Verify file permissions** in Azure App Service
5. **Check file paths** are correct (Linux vs Windows path separators)
6. **Ensure `apply_ui_styling()` is called** before rendering UI

### If CSS Failsafe Shows Empty Space Instead of Icons
This means the font didn't load but the failsafe is working. To debug:
1. Check application logs for font loading messages
2. Verify font file exists: `ls -la src/keyword_code/assets/fonts/`
3. Check file permissions are readable

### Font File Load Errors
Check application logs for:
```
Failed to load local Material Icons font: [error message]
```

Common causes:
- File permissions in Azure App Service
- Incorrect relative path resolution
- Files not included in deployment package

### Performance Issues
Base64 encoding fonts adds ~33% size overhead and loads on every page. If this causes issues:
- Consider serving fonts as static files instead of embedded data URIs
- Add browser caching headers for font files

## Verification Checklist
- [ ] `MaterialSymbolsRounded.woff2` exists in `src/keyword_code/assets/fonts/`
- [ ] Legacy font files exist (MaterialIcons-Regular.woff2, MaterialIconsOutlined-Regular.woff2)
- [ ] Test app shows icons correctly locally
- [ ] Deployment package includes all font files
- [ ] Azure app shows icons correctly (not text)
- [ ] If icons fail, text is hidden (not showing `keyboard_double_arrow_right`)
- [ ] Browser console has no font errors
- [ ] Application logs show successful font loading

## Contact
If issues persist after deployment, provide:
1. Browser console screenshot (F12 → Console tab)
2. Application log excerpt showing `apply_ui_styling()` execution
3. Screenshot of the icon display issue

---
**Original Implementation Date**: November 17, 2025
**Fix Updated**: January 19, 2026 - Changed to Material Symbols Rounded font + added CSS failsafes
**Verified For**: Streamlit 1.49.0 (bundled font + our embedded font + CSS failsafes)
**Modified Files**: `src/keyword_code/utils/ui_helpers.py`, `pages/1_📄_CNT_space.py`
**Added Files**: `MaterialSymbolsRounded.woff2` in `src/keyword_code/assets/fonts/`
**Root Cause**: Original fix used "Material Icons" font, but Streamlit 1.44+ uses "Material Symbols Rounded"
