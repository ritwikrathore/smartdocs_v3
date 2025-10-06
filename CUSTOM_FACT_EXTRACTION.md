# Custom LLM-Based Fact Extraction System

## Overview

This document describes the custom LLM-based fact extraction system that replaces the LangExtract library. The new system uses Pydantic-AI agents to intelligently identify fact types from user queries and extract structured facts from analysis text.

## Architecture

### Two-Step Process

1. **Fact Type Identification**: Analyzes the user's query to determine what types of facts should be extracted
2. **Fact Extraction**: Extracts facts from the analysis text based on the identified types

### Key Components

#### 1. Pydantic Models (`src/keyword_code/agents/fact_extraction_agent.py`)

- **FactType**: Enum defining supported fact types (definition, percentage, amount, date, entity, etc.)
- **FactTypeAnalysis**: Analysis result from the type identification agent
- **ExtractedFact**: A single extracted fact with validation
- **FactExtractionResult**: Complete extraction result with metadata and errors

#### 2. Pydantic-AI Agents

- **FactTypeIdentificationAgent**: Analyzes queries to identify expected fact types
- **FactExtractionAgent**: Extracts facts from analysis text with retry logic

#### 3. Orchestration Service (`src/keyword_code/services/fact_extraction_service.py`)

- **FactExtractionService**: Coordinates the two-step process and provides backward-compatible interfaces

## Supported Fact Types

| Fact Type | Description | Example Query |
|-----------|-------------|---------------|
| DEFINITION | Explanations and meanings | "What is a business day?" |
| PERCENTAGE | Percentage values | "What is the interest rate?" |
| AMOUNT | Numeric amounts without currency | "How many days?" |
| CURRENCY_AMOUNT | Monetary amounts with currency | "What is the loan amount?" |
| DATE | Dates or time periods | "When is the maturity date?" |
| ENTITY | Names of people/organizations | "Who is the borrower?" |
| RATE | Rates or ratios | "What is the exchange rate?" |
| DURATION | Time durations | "What is the loan term?" |
| BOOLEAN | Yes/no values | "Is it secured?" |
| LIST | Lists of items | "What are the conditions?" |
| OTHER | Any other type | - |

## Key Features

### 1. Intelligent Fact Type Detection

The system analyzes the user's query to automatically determine what types of facts to extract:

```python
query = "What are the loan amounts for Facility A and Facility B?"
# System identifies: CURRENCY_AMOUNT type
# Extraction hints: "Look for multiple loan amounts"
```

### 2. Multiple Instance Extraction

The system can extract multiple instances of the same fact type:

```python
# Query: "What are the loan amounts?"
# Analysis: "Facility A: $500M, Facility B: $300M"
# Extracts:
#   - Fact: "Facility A Loan Amount", Value: "$500 million"
#   - Fact: "Facility B Loan Amount", Value: "$300 million"
```

### 3. Comprehensive Validation

Pydantic validators ensure data quality:

- **Fact names**: Must be descriptive (min 2 chars)
- **Fact values**: Cannot be empty
- **Type-specific validation**:
  - Percentages must contain numeric data
  - Amounts must contain numeric data
  - Dates must contain numeric data
- **Confidence scores**: Must be between 0.0 and 1.0

### 4. Intelligent Retry Logic

The system automatically retries failed extractions with enhanced prompts:

- **Attempt 1**: Standard extraction
- **Attempt 2**: Enhanced prompt with previous error context
- **Attempt 3**: Final attempt with all validation feedback

Each retry includes:
- Previous error messages
- Validation error details
- Specific guidance for improvement

### 5. Backward Compatibility

The service maintains compatibility with the old LangExtract interface:

```python
# Old interface (still works)
extracted_facts = service.extract_facts_from_text(
    text=analysis_text,
    context="Legal/Financial Analysis"
)

# New interface (recommended)
result = service.extract_facts(
    query="What is the interest rate?",
    analysis_text=analysis_text,
    context="Loan Agreement Analysis"
)
```

## Usage Examples

### Basic Usage

```python
from src.keyword_code.services.fact_extraction_service import FactExtractionService

# Initialize the service
service = FactExtractionService()

# Extract facts
result = service.extract_facts(
    query="What is the interest rate?",
    analysis_text="The loan has an interest rate of 5.5% per annum.",
    context="Loan Agreement"
)

# Access results
for fact in result.extracted_facts:
    print(f"{fact.fact_name}: {fact.fact_value}")
    print(f"Type: {fact.fact_type.value}, Confidence: {fact.confidence}")
```

### Batch Processing

```python
# Extract from multiple analyses
analyses = [
    {
        "analysis_summary": "The loan amount is $500 million...",
        "sub_prompt_analyzed": "What is the loan amount?",
        "analysis_context": "Facility A"
    },
    # ... more analyses
]

results = service.extract_facts_from_multiple_analyses(analyses)
```

### Excel Export Format

```python
# Extract fact-definition pairs for Excel export
rows = service.extract_fact_definitions_from_text(
    text=analysis_text,
    section_name="Key Terms",
    filename="loan_agreement.pdf"
)

# Each row has: Filename, Section, Fact, Definition
# Can be directly exported to Excel with pandas
import pandas as pd
df = pd.DataFrame(rows)
df.to_excel("facts.xlsx", index=False)
```

## UI Integration

### Generate Facts Button

The "Generate Facts" button in the UI triggers the new extraction system:

```python
# In display.py
from src.keyword_code.services.fact_extraction_service import FactExtractionService

fact_service = FactExtractionService()
rows = fact_service.extract_fact_definitions_for_results(results_with_real_analysis)

# Export to Excel (Fact/Definition columns only)
df = pd.DataFrame([{"Fact": r["Fact"], "Definition": r["Definition"]} for r in rows])
```

### Inline Section Facts

Facts are automatically extracted for each analysis section:

```python
# In display.py - per section
fact_service = FactExtractionService()
extracted_facts = fact_service.extract_facts_from_text(
    text=section_data.get("Analysis", ""),
    context=f"Legal/Financial Analysis - Section: {section_key}",
    section_name=section_key,
    filename=result.get("filename", "Unknown")
)
```

## Configuration

### Environment Variables

- **DATABRICKS_API_KEY**: Required for LLM access
- **DATABRICKS_BASE_URL**: Databricks serving endpoint URL (default configured)
- **DATABRICKS_LLM_MODEL**: Model to use (default: "databricks-llama-4-maverick")

### Retry Configuration

```python
# Adjust max retries (default: 2)
result = service.extract_facts(
    query=query,
    analysis_text=text,
    max_retries=3  # Try up to 4 times total
)
```

## Error Handling

The system provides comprehensive error reporting:

```python
result = service.extract_facts(query, text)

if result.errors:
    print("Extraction errors:")
    for error in result.errors:
        print(f"  - {error}")

# Check metadata for details
metadata = result.extraction_metadata
print(f"Attempts: {metadata.get('attempts')}")
print(f"Validation errors: {metadata.get('validation_errors')}")
```

## Testing

Run the test suite to validate the system:

```bash
python test_custom_fact_extraction.py
```

Tests cover:
- Definition extraction
- Percentage extraction
- Multiple amounts extraction
- Date extraction
- Excel export format validation

## Migration from LangExtract

### What Changed

1. **Import statements**:
   ```python
   # Old
   from src.keyword_code.langextract_integration.fact_extractor import FactExtractor
   
   # New
   from src.keyword_code.services.fact_extraction_service import FactExtractionService
   ```

2. **Initialization**:
   ```python
   # Old
   fact_extractor = FactExtractor()
   
   # New
   fact_service = FactExtractionService()
   ```

3. **Method calls** (backward compatible):
   ```python
   # Both work the same way
   result = fact_service.extract_facts_from_text(text, context)
   ```

### What Stayed the Same

- Return format for `extract_facts_from_text()`
- Return format for `extract_fact_definitions_from_text()`
- Excel export format (Fact/Definition columns)
- UI display format

## Performance Considerations

- **Async support**: Use `extract_facts_async()` for better performance in async contexts
- **Batch processing**: Process multiple sections in parallel when possible
- **Caching**: The Databricks LLM client is cached via Streamlit's `@st.cache_resource`

## Troubleshooting

### No facts extracted

1. Check that DATABRICKS_API_KEY is set
2. Verify the analysis text contains extractable information
3. Check logs for validation errors
4. Try increasing max_retries

### Low-quality extractions

1. Review the query - make it more specific
2. Check the analysis text quality
3. Review extraction_metadata for validation errors
4. Consider adjusting the system prompts in the agent classes

### Validation errors

1. Check that numeric facts contain actual numbers
2. Ensure fact names are descriptive (min 2 chars)
3. Verify confidence scores are between 0.0 and 1.0
4. Review the validation_errors in extraction_metadata

## Future Enhancements

Potential improvements:
- Add more fact types (e.g., PHONE_NUMBER, EMAIL, ADDRESS)
- Implement caching for repeated queries
- Add support for custom fact type definitions
- Enhance validation rules per fact type
- Add confidence threshold filtering
- Implement fact deduplication across documents

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Review the test suite for usage examples
3. Examine the extraction_metadata for debugging info
4. Contact the CNT Automations team

