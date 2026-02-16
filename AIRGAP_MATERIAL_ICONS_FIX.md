# Material Icons Fix for Airgapped Azure Deployment

## Problem
In airgapped Azure environments, Streamlit's Material Icons fail to load from Google's CDN, causing:
- Expander arrow icons appearing as text (e.g., `keyboard_double_arrow_right`)
- Missing UI icons throughout the application

## Solution Implemented
We've implemented a **local Material Icons font** solution that embeds the font files directly into the application CSS, eliminating external CDN dependencies.

## Files Added/Modified

### 1. Font Files (Already Downloaded)
```
src/keyword_code/assets/fonts/
├── MaterialIcons-Regular.woff2          # Main Material Icons font
├── MaterialIconsOutlined-Regular.woff2  # Outlined variant
└── README.md                            # Documentation
```

### 2. Modified Code
- **`src/keyword_code/utils/ui_helpers.py`**: Updated `apply_ui_styling()` function to:
  - Load font files from local disk
  - Base64-encode fonts for embedding in CSS
  - Inject `@font-face` rules with embedded font data
  - Apply Material Icons styling to Streamlit icon elements

## How It Works
1. On app startup, `apply_ui_styling()` reads the `.woff2` font files
2. Fonts are base64-encoded and embedded directly in CSS via `data:` URIs
3. CSS rules override Streamlit's default font loading behavior
4. All Material Icons render from local embedded fonts, not CDN

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

## Fallback Behavior
If font files are missing or fail to load:
- A warning is logged: `Failed to load local Material Icons font: {error}`
- App continues to run, but icons may display as text
- No application crash occurs

## File Size Impact
- MaterialIcons-Regular.woff2: ~49 KB
- MaterialIconsOutlined-Regular.woff2: ~38 KB
- Total: ~87 KB added to deployment

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

### Icons Still Display as Text
1. Check browser console (F12) for JavaScript errors
2. Verify font files exist and are readable in deployment
3. Check file paths are correct (Linux vs Windows path separators)
4. Ensure `apply_ui_styling()` is called before rendering UI

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
- [ ] Font files exist in `src/keyword_code/assets/fonts/`
- [ ] Test app shows icons correctly locally
- [ ] Deployment includes font files in package
- [ ] Azure app shows icons correctly (not text)
- [ ] Browser console has no font errors
- [ ] Application logs show no font loading warnings

## Contact
If issues persist after deployment, provide:
1. Browser console screenshot (F12 → Console tab)
2. Application log excerpt showing `apply_ui_styling()` execution
3. Screenshot of the icon display issue

---
**Implementation Date**: November 17, 2025  
**Modified Files**: `src/keyword_code/utils/ui_helpers.py`  
**Added Files**: Font files in `src/keyword_code/assets/fonts/`
