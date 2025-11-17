# Azure Production Environment Variables

This document lists the environment variables needed for deploying SmartDocs to Azure in an airgapped environment.

## Required Environment Variables

### Databricks (Required)
```bash
DATABRICKS_API_KEY=<your-databricks-api-key>
```

## Optional Configuration

### Langfuse Tracing (Development/Debugging Only)
**For production, set to false or omit entirely:**
```bash
ENABLE_LANGFUSE_TRACING=false
```

If you want Langfuse enabled (not recommended for airgapped prod), you also need:
```bash
LANGFUSE_PUBLIC_KEY=<your-public-key>
LANGFUSE_SECRET_KEY=<your-secret-key>
LANGFUSE_BASE_URL=<your-langfuse-instance-url>
```

### Chunking Strategy
**Default is true (adaptive chunker enabled). Only set if you need legacy chunker:**
```bash
USE_ADAPTIVE_SENTENCE_CHUNKER=true  # Default, can be omitted
```

### Worker Configuration
```bash
MAX_WORKERS=3                # Default: 3
ENABLE_PARALLEL=true         # Default: true
```

### Logging Configuration
```bash
# Set to "true" to enable file-based logging (logs/app_*.log)
# Default: false (console only)
ENABLE_APP_LOGGING=false

# Set to "true" to enable detailed RAG interaction logs
# Default: true
ENABLE_INTERACTION_LOGGING=true

# Set to "true" to enable highlight matching debug logs
# Default: true  
ENABLE_HIGHLIGHT_DEBUG_LOGGING=true
```

### LLM Configuration
```bash
LLM_MAX_RETRIES=3           # Default: 3
```

### RAG Configuration (Advanced)
```bash
# Adaptive chunker bounds (only used if USE_ADAPTIVE_SENTENCE_CHUNKER=true)
ADAPTIVE_CHUNK_MIN_CHARS=450
ADAPTIVE_CHUNK_MAX_CHARS=900
ADAPTIVE_CHUNK_MIN_SENTENCES=3
ADAPTIVE_CHUNK_MAX_SENTENCES=8
ADAPTIVE_CHUNK_OVERLAP_SENTENCES=2
```

## Minimal Production Configuration

For a basic airgapped Azure production deployment, you only need:

```bash
# Required
DATABRICKS_API_KEY=<your-key>

# Recommended for airgapped environments
ENABLE_LANGFUSE_TRACING=false
```

All other settings use sensible defaults defined in `src/keyword_code/config.py`.

## How Defaults Work

1. **Environment variables take precedence** over config defaults
2. **If environment variable is not set**, the default from `config.py` is used
3. **Langfuse gracefully degrades** when disabled - no errors, just no tracing

## Testing Configuration

To verify your configuration works locally before Azure deployment:

```bash
# Create a test .env file with your Azure settings
cp .env .env.azure-test

# Edit .env.azure-test to match Azure config
# Then test locally:
python -m streamlit run Home.py
```

## Troubleshooting

### Issue: Langfuse errors in logs
**Solution:** Set `ENABLE_LANGFUSE_TRACING=false` or remove Langfuse keys

### Issue: Icons showing as text (e.g., "keyboard_double_arrow_right")
**Solution:** Ensure Material Icons font files are included in deployment package:
- `src/keyword_code/assets/fonts/MaterialIcons-Regular.woff2`
- `src/keyword_code/assets/fonts/MaterialIconsOutlined-Regular.woff2`

### Issue: Chunking behavior different from local
**Solution:** Check `USE_ADAPTIVE_SENTENCE_CHUNKER` setting matches between environments

---
**Last Updated:** November 17, 2025
