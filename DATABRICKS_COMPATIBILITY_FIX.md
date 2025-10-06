# Databricks Compatibility Fix

## Issues Encountered

### Issue 1: JSON Schema Constraints

The initial implementation used Pydantic-AI's structured output feature, which automatically generates JSON schemas with constraints. Databricks LLM endpoints don't support certain JSON schema features.

**Error Messages:**
```
pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: databricks-llama-4-maverick,
body: {'error_code': 'BAD_REQUEST', 'message': 'Invalid JSON schema - number types do not support minimum\n'}
```

```
pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: databricks-llama-4-maverick,
body: {'error_code': 'BAD_REQUEST', 'message': 'Invalid JSON schema - the "additionalProperties" keyword must be False or not specified\n'}
```

### Issue 2: Markdown-Wrapped JSON Responses

Even after bypassing Pydantic-AI's structured output, the LLM was returning JSON wrapped in markdown code blocks:

```
```json
{
  "query": "...",
  "expected_fact_types": [...]
}
```
```

This caused JSON parsing errors:
```
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

## Root Cause

When using Pydantic-AI's `Agent` with `output_type=<PydanticModel>`, it automatically generates a JSON schema from the Pydantic model and sends it to the LLM as part of the structured output request. Pydantic models with field constraints like:

```python
confidence: float = Field(ge=0.0, le=1.0)  # ge/le generate minimum/maximum in JSON schema
```

Generate JSON schemas with `minimum` and `maximum` constraints that Databricks doesn't support.

## Solutions

### Solution 1: Bypass Pydantic-AI Structured Output

Instead of using Pydantic-AI's structured output feature, we now:

1. **Use Databricks LLM client directly** - Bypass Pydantic-AI's structured output
2. **Request JSON in the prompt** - Ask the LLM to respond with JSON
3. **Parse JSON manually** - Parse the response and construct Pydantic models manually
4. **Remove field constraints** - Remove `ge`, `le`, `gt`, `lt` from Field definitions
5. **Validate in code** - Use `@field_validator` decorators for validation instead

### Solution 2: Strip Markdown Formatting

Created a helper function to strip markdown code blocks from JSON responses:

```python
def strip_markdown_json(response: str) -> str:
    """Strip markdown code block formatting from JSON responses."""
    response_clean = response.strip()

    # Remove opening markdown fence
    if response_clean.startswith("```json"):
        response_clean = response_clean[7:]
    elif response_clean.startswith("```"):
        response_clean = response_clean[3:]

    # Remove closing markdown fence
    if response_clean.endswith("```"):
        response_clean = response_clean[:-3]

    return response_clean.strip()
```

This function is called before JSON parsing to handle LLMs that wrap responses in markdown.

## Changes Made

### 1. Removed Field Constraints

**Before:**
```python
confidence: float = Field(
    description="Confidence score (0-1)",
    ge=0.0,  # ❌ Generates "minimum" in JSON schema
    le=1.0   # ❌ Generates "maximum" in JSON schema
)
```

**After:**
```python
confidence: float = Field(
    description="Confidence score (0-1)"
)

@field_validator('confidence')
@classmethod
def validate_confidence_range(cls, v: float) -> float:
    """Ensure confidence is within valid range."""
    if v < 0.0 or v > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
    return v
```

### 2. Modified FactTypeIdentificationAgent

**Before:**
```python
result = await self.agent.run(prompt, message_history=[])
return result.output  # Pydantic-AI handles structured output
```

**After:**
```python
# Use Databricks client directly
messages = [
    {"role": "system", "content": "You are an expert. Respond only with valid JSON. Do not wrap the JSON in markdown code blocks."},
    {"role": "user", "content": prompt}
]

response = self.databricks_client.get_completion(messages, max_tokens=1000)

# Strip markdown formatting if present
response_clean = strip_markdown_json(response)

# Parse JSON manually
response_json = json.loads(response_clean)

# Construct Pydantic model manually
return FactTypeAnalysis(
    query=response_json.get("query", query),
    expected_fact_types=[FactType(ft) for ft in response_json.get("expected_fact_types", [])],
    reasoning=response_json.get("reasoning", ""),
    confidence=float(response_json.get("confidence", 0.5)),
    extraction_hints=response_json.get("extraction_hints", [])
)
```

### 3. Modified FactExtractionAgent

Same approach as FactTypeIdentificationAgent:
- Use Databricks client directly
- Request JSON in prompt with detailed schema description
- Parse JSON response manually
- Construct Pydantic models from parsed data
- Handle JSON parsing errors with retry logic

### 4. Enhanced Error Handling

Added specific handling for JSON parsing errors:

```python
except json.JSONDecodeError as e:
    logger.error(f"JSON parsing error: {e}")
    logger.error(f"Response was: {response}")
    # Retry with enhanced prompt
    attempt += 1
```

## Benefits of This Approach

1. **✅ Databricks Compatible** - No JSON schema constraints
2. **✅ More Control** - Full control over JSON parsing and error handling
3. **✅ Better Error Messages** - Can log the actual LLM response when parsing fails
4. **✅ Flexible** - Can handle variations in JSON format
5. **✅ Retry-Friendly** - Can adjust prompts based on parsing errors

## Testing

### Quick Test
```bash
python test_fact_extraction_fix.py
```

### Full Test Suite
```bash
python test_custom_fact_extraction.py
```

## Prompt Engineering

The prompts now explicitly request JSON format:

```python
prompt = f"""
...your instructions...

Respond with a JSON object containing:
- query: the original query
- expected_fact_types: list of fact type strings
- reasoning: explanation
- confidence: number between 0.0 and 1.0
- extraction_hints: list of hints

Example response:
{{
    "query": "What is the interest rate?",
    "expected_fact_types": ["percentage"],
    "reasoning": "Query asks for a rate",
    "confidence": 0.95,
    "extraction_hints": ["Look for percentage values"]
}}

Respond ONLY with the JSON object, no other text.
"""
```

## Validation Strategy

Validation now happens in two places:

1. **Pydantic Validators** - Run when constructing models from parsed JSON
   ```python
   @field_validator('confidence')
   @classmethod
   def validate_confidence_range(cls, v: float) -> float:
       if v < 0.0 or v > 1.0:
           raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
       return v
   ```

2. **Quality Validation** - Additional checks after extraction
   ```python
   if len(fact.fact_name) < 2:
       validation_errors.append(f"Fact name too short: {fact.fact_name}")
       continue
   ```

## Backward Compatibility

All public interfaces remain unchanged:
- `FactExtractionService.extract_facts()` - Same signature and return type
- `FactExtractionService.extract_facts_from_text()` - Same signature and return type
- All Pydantic models - Same structure and fields

The changes are internal implementation details that don't affect the API.

## Performance Impact

Minimal performance impact:
- **Before**: Pydantic-AI structured output (1 LLM call + schema validation)
- **After**: Direct LLM call + JSON parsing + Pydantic validation (1 LLM call + manual parsing)

The overhead of manual JSON parsing is negligible compared to the LLM call time.

## Future Considerations

### If Databricks Adds JSON Schema Support

If Databricks adds support for JSON schema constraints in the future, we can:
1. Keep the current implementation (it works fine)
2. Or switch back to Pydantic-AI structured output for cleaner code

### Alternative Approaches Considered

1. **Use `json_mode='json'` in Pydantic-AI** - Not available in current version
2. **Custom Pydantic-AI model wrapper** - Too complex, not worth it
3. **Fork Pydantic-AI** - Overkill for this issue
4. **Use different LLM** - Not an option (Databricks is required)

## Conclusion

The fix successfully resolves the Databricks compatibility issue while maintaining:
- ✅ All functionality
- ✅ Backward compatibility
- ✅ Code quality
- ✅ Error handling
- ✅ Validation
- ✅ Performance

The system is now production-ready and fully compatible with Databricks LLM endpoints.

