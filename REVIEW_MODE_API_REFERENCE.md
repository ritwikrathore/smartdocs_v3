# Review Mode API Reference

## Quick Start

### Ask Mode Decomposition
```python
from src.keyword_code.ai.analyzer import DocumentAnalyzer
from src.keyword_code.ai.decomposition import decompose_ask_mode_prompt

analyzer = DocumentAnalyzer()
user_query = "What is the loan amount and interest rate?"

# Returns List[Dict] with 'title', 'sub_prompt', and 'rag_params'
sub_prompts = await decompose_ask_mode_prompt(analyzer, user_query)

for sp in sub_prompts:
    print(f"Title: {sp['title']}")
    print(f"Sub-prompt: {sp['sub_prompt']}")
    print(f"BM25 weight: {sp['rag_params']['bm25_weight']}")
    print(f"Semantic weight: {sp['rag_params']['semantic_weight']}")
```

### Review Mode Decomposition
```python
from src.keyword_code.ai.analyzer import DocumentAnalyzer
from src.keyword_code.ai.decomposition import decompose_review_mode_prompt

analyzer = DocumentAnalyzer()
rules_text = """
Check that currency names use proper case (e.g., 'U.S. dollar' not 'U.S. Dollar').
Verify dates are in MM/DD/YYYY format.
"""

# Returns List[ReviewModeSubPrompt]
sub_prompts = await decompose_review_mode_prompt(analyzer, rules_text)

for sp in sub_prompts:
    print(f"Title: {sp.title}")
    print(f"Clarified rule: {sp.sub_prompt}")
    print(f"Validation type: {sp.validation_type}")
    print(f"Reasoning: {sp.validation_reasoning}")
    print(f"Extracted examples: {sp.extracted_examples}")
    print(f"Violation examples: {sp.violation_examples}")
    print(f"Compliance examples: {sp.compliance_examples}")
```

### Regex Pattern Generation
```python
from src.keyword_code.ai.analyzer import DocumentAnalyzer
from src.keyword_code.ai.decomposition import (
    decompose_review_mode_prompt,
    generate_regex_pattern
)

analyzer = DocumentAnalyzer()
rules_text = "Dates must be in MM/DD/YYYY format"

# Step 1: Decompose
sub_prompts = await decompose_review_mode_prompt(analyzer, rules_text)

# Step 2: Generate regex for regex-flagged rules
for sp in sub_prompts:
    if sp.validation_type == "regex":
        regex_result = await generate_regex_pattern(analyzer, sp)
        if regex_result:
            print(f"Pattern: {regex_result.regex_pattern}")
            print(f"Explanation: {regex_result.explanation}")
            print(f"Test matches: {regex_result.test_matches}")
            print(f"Test non-matches: {regex_result.test_non_matches}")
```

### Complete Review Mode Workflow (V2)
```python
from src.keyword_code.smartreview import propose_validation_from_rule_v2

rule_text = "Currency names must use proper case"
example_text = "U.S. dollar"  # Optional

# Single API call that handles decomposition and regex generation
proposal = await propose_validation_from_rule_v2(rule_text, example_text)

if proposal:
    print(f"Validation type: {proposal.validation_type}")
    print(f"Validator: {proposal.validator}")
    print(f"Clarified rule: {proposal.clarified_rule}")
    print(f"Extracted examples: {proposal.extracted_examples}")
```

## Data Structures

### ReviewModeSubPrompt
```python
class ReviewModeSubPrompt(BaseModel):
    title: str  # Concise title (max 5-6 words)
    sub_prompt: str  # Re-written/clarified rule text
    validation_type: Literal['regex', 'semantic']  # Validation approach
    validation_reasoning: str  # Why this approach was chosen
    extracted_examples: List[str]  # Examples from rule text
    violation_examples: List[str]  # What WOULD violate
    compliance_examples: List[str]  # What WOULD comply
```

### RegexGenerationResult
```python
class RegexGenerationResult(BaseModel):
    regex_pattern: str  # Python-compatible regex pattern
    explanation: str  # What the regex matches
    test_matches: List[str]  # Should match these
    test_non_matches: List[str]  # Should NOT match these
```

### ProposedValidation
```python
class ProposedValidation(BaseModel):
    explanation: str  # User-friendly explanation
    validation_type: Literal['regex', 'semantic']
    validator: str  # Regex pattern or semantic prompt
    example_finding: str  # Example from document
    clarified_rule: str  # Rewritten rule
    extracted_examples: List[str]  # Examples to ignore
```

## Function Reference

### decompose_ask_mode_prompt
```python
async def decompose_ask_mode_prompt(
    analyzer: DocumentAnalyzer,
    user_prompt: str
) -> List[Dict[str, str]]
```
**Purpose:** Decompose Ask Mode queries into sub-prompts with RAG parameters

**Parameters:**
- `analyzer`: DocumentAnalyzer instance
- `user_prompt`: User's question/query

**Returns:** List of dicts with keys:
- `title`: Concise title (str)
- `sub_prompt`: Full sub-prompt text (str)
- `rag_params`: Dict with `bm25_weight`, `semantic_weight`, `reasoning`

**Example:**
```python
[
    {
        "title": "Loan Amount",
        "sub_prompt": "What is the loan amount?",
        "rag_params": {
            "bm25_weight": 0.6,
            "semantic_weight": 0.4,
            "reasoning": "Numerical data requires keyword precision"
        }
    }
]
```

### decompose_review_mode_prompt
```python
async def decompose_review_mode_prompt(
    analyzer: DocumentAnalyzer,
    user_prompt: str
) -> List[ReviewModeSubPrompt]
```
**Purpose:** Decompose Review Mode rules into structured sub-prompts

**Parameters:**
- `analyzer`: DocumentAnalyzer instance
- `user_prompt`: Validation rule(s) text

**Returns:** List of `ReviewModeSubPrompt` objects

**Key Features:**
- No document text included (saves tokens)
- Determines validation type (regex vs semantic)
- Extracts examples from rule text
- Generates violation/compliance examples
- Clarifies ambiguous rules

### generate_regex_pattern
```python
async def generate_regex_pattern(
    analyzer: DocumentAnalyzer,
    sub_prompt: ReviewModeSubPrompt
) -> Optional[RegexGenerationResult]
```
**Purpose:** Generate regex pattern for regex-flagged rules

**Parameters:**
- `analyzer`: DocumentAnalyzer instance
- `sub_prompt`: ReviewModeSubPrompt with validation_type='regex'

**Returns:** `RegexGenerationResult` or None on failure

**Key Features:**
- Comprehensive regex guidelines in system prompt
- Handles word boundaries and decimal numbers
- Validates generated pattern (compiles it)
- Provides test cases

### propose_validation_from_rule_v2
```python
async def propose_validation_from_rule_v2(
    rule_text: str,
    example_text: str = ""
) -> Optional[ProposedValidation]
```
**Purpose:** Complete workflow for creating validation rule (V2 - no document text)

**Parameters:**
- `rule_text`: The validation rule in plain text
- `example_text`: Optional example (default: "")

**Returns:** `ProposedValidation` object or None

**Workflow:**
1. Calls `decompose_review_mode_prompt` (no document text)
2. For regex: calls `generate_regex_pattern`
3. For semantic: uses clarified rule as validator
4. Returns `ProposedValidation` compatible with UI

### propose_validation_from_rule (Legacy)
```python
async def propose_validation_from_rule(
    rule_text: str,
    example_text: str,
    doc_chunks: List[DocumentChunk]
) -> Optional[ProposedValidation]
```
**Purpose:** Legacy version with document text (for interactive UI)

**Parameters:**
- `rule_text`: The validation rule
- `example_text`: Optional example
- `doc_chunks`: Document chunks for context

**Returns:** `ProposedValidation` object or None

**Note:** Kept for backward compatibility. Use V2 for new code.

## Migration Examples

### Before (Legacy)
```python
from src.keyword_code.smartreview import propose_validation_from_rule

# Required document chunks
doc_chunks = extract_text_from_pdf(pdf_bytes)

# API call includes document text (wastes tokens)
proposal = await propose_validation_from_rule(
    rule_text="Check currency case",
    example_text="U.S. dollar",
    doc_chunks=doc_chunks
)
```

### After (V2)
```python
from src.keyword_code.smartreview import propose_validation_from_rule_v2

# No document chunks needed for decomposition
proposal = await propose_validation_from_rule_v2(
    rule_text="Check currency case",
    example_text="U.S. dollar"
)

# Document chunks only needed for validation execution
doc_chunks = extract_text_from_pdf(pdf_bytes)
results = await execute_validation_template(template, doc_chunks)
```

## Best Practices

1. **Use V2 for Automated Workflows**
   - Saves tokens by not including document text
   - Cleaner separation of concerns

2. **Use Legacy for Interactive UI**
   - Provides example findings from actual document
   - Better user experience for rule creation

3. **Handle Regex Generation Failures**
   ```python
   if sp.validation_type == "regex":
       regex_result = await generate_regex_pattern(analyzer, sp)
       if not regex_result:
           # Fallback to semantic
           sp.validation_type = "semantic"
   ```

4. **Validate Regex Patterns**
   ```python
   import re
   try:
       re.compile(regex_result.regex_pattern)
   except re.error as e:
       print(f"Invalid regex: {e}")
   ```

5. **Use Extracted Examples**
   ```python
   # Don't flag examples from rule text as violations
   rule = Rule(
       description=rule_text,
       validation_type=proposal.validation_type,
       validator=proposal.validator,
       extracted_examples=proposal.extracted_examples
   )
   ```

## Common Patterns

### Batch Processing Multiple Rules
```python
rules = [
    "Currency names must use proper case",
    "Dates must be in MM/DD/YYYY format",
    "Numbers over 1 billion must include decimal precision"
]

proposals = []
for rule_text in rules:
    proposal = await propose_validation_from_rule_v2(rule_text)
    if proposal:
        proposals.append(proposal)
```

### Creating Validation Template
```python
from src.keyword_code.smartreview import ValidationTemplate, Rule

rules = []
for proposal in proposals:
    rule = Rule(
        description=rule_text,
        validation_type=proposal.validation_type,
        validator=proposal.validator,
        clarified_rule=proposal.clarified_rule,
        extracted_examples=proposal.extracted_examples
    )
    rules.append(rule)

template = ValidationTemplate(
    name="My Validation Template",
    rules=rules
)
```

### Executing Validation
```python
from src.keyword_code.smartreview import execute_validation_template

doc_chunks = extract_text_from_pdf(pdf_bytes)
results = await execute_validation_template(template, doc_chunks)

for result in results:
    print(f"Page {result.page_num}: {result.finding}")
    print(f"Analysis: {result.analysis}")
```

