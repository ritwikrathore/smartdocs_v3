# Quick Start Guide - Custom Fact Extraction

## ✅ System Status

The custom LLM-based fact extraction system is **fully functional** and ready to use!

## 🚀 How to Use

### In the UI (Streamlit App)

1. **Start the app:**
   ```bash
   streamlit run src/keyword_code/app.py
   ```

2. **Upload and analyze a document:**
   - Upload your document (PDF, DOCX, etc.)
   - Run the analysis
   - Wait for results

3. **Generate facts:**
   - Scroll to the **"Fact Extraction (beta)"** expander
   - Click the **"Generate Facts"** button
   - Wait for extraction to complete (~2-4 seconds per section)
   - Preview the extracted facts
   - Download as Excel if needed

### Programmatically

```python
from src.keyword_code.services.fact_extraction_service import FactExtractionService

# Initialize service
service = FactExtractionService()

# Extract facts
result = service.extract_facts(
    query="What is the interest rate?",
    analysis_text="The interest rate is 5.5% per annum.",
    context="Loan Agreement"
)

# Access extracted facts
for fact in result.extracted_facts:
    print(f"{fact.fact_name}: {fact.fact_value}")
    print(f"  Type: {fact.fact_type}")
    print(f"  Confidence: {fact.confidence}")
```

## 📊 What Gets Extracted

The system intelligently identifies and extracts:

- **Definitions**: "What is a business day?"
- **Percentages**: "What is the interest rate?"
- **Amounts**: "What is the loan amount?"
- **Currency Amounts**: "What is the facility amount?"
- **Dates**: "When is the maturity date?"
- **Entities**: "Who is the borrower?"
- **Rates**: "What is the exchange rate?"
- **Durations**: "What is the loan term?"
- **Boolean**: "Is it secured?"
- **Lists**: "What are the conditions?"
- **Other**: Any other type of fact

## 🎯 Key Features

### Intelligent Type Detection
The system analyzes your query to determine what type of facts to extract:
- Query: "What is the interest rate?" → Extracts percentages/rates
- Query: "What are the loan amounts?" → Extracts currency amounts

### Multiple Instance Extraction
Can extract multiple instances separately:
- "Facility A: $500M, Facility B: $300M"
- Extracts as: "Facility A Loan Amount" and "Facility B Loan Amount"

### Quality Validation
- Type-specific validation (percentages must have numbers, etc.)
- Confidence scoring (0.0 to 1.0)
- Automatic retry with enhanced prompts if quality is low

### Excel Export
- Standard format: Filename, Section, Fact, Definition
- Compatible with existing workflows
- Easy to share and analyze

## 🔧 Configuration

### Environment Variables
```bash
# Required
DATABRICKS_API_KEY=your_api_key_here

# Optional (defaults provided)
DATABRICKS_BASE_URL=https://your-workspace.azuredatabricks.net
DATABRICKS_LLM_MODEL=databricks-llama-4-maverick
```

### Adjust Retry Attempts
```python
result = service.extract_facts(
    query=query,
    analysis_text=text,
    max_retries=3  # Default: 2
)
```

## 📝 Example Outputs

### Example 1: Definition
**Query:** "What is a business day?"
**Analysis:** "A Business Day is any day other than Saturday, Sunday, or a bank holiday."

**Extracted Facts:**
- **Fact:** Business Day
- **Definition:** Any day other than Saturday, Sunday, or a bank holiday
- **Type:** definition
- **Confidence:** 0.95

### Example 2: Multiple Amounts
**Query:** "What are the loan amounts for Facility A and Facility B?"
**Analysis:** "Facility A has a loan amount of $500 million. Facility B has a loan amount of $300 million."

**Extracted Facts:**
- **Fact:** Facility A Loan Amount
- **Definition:** $500 million
- **Type:** currency_amount
- **Confidence:** 0.95

- **Fact:** Facility B Loan Amount
- **Definition:** $300 million
- **Type:** currency_amount
- **Confidence:** 0.95

### Example 3: Percentage
**Query:** "What is the interest rate?"
**Analysis:** "The interest rate is 5.5% per annum."

**Extracted Facts:**
- **Fact:** Interest Rate
- **Definition:** 5.5% per annum
- **Type:** percentage
- **Confidence:** 0.95

## 🐛 Troubleshooting

### No Facts Extracted
**Possible Causes:**
- Analysis text is too short or unclear
- Query doesn't match the analysis content
- LLM couldn't identify relevant facts

**Solutions:**
- Check the analysis text quality
- Make the query more specific
- Check logs for error messages

### Low Quality Facts
**Possible Causes:**
- Analysis text is ambiguous
- Multiple interpretations possible

**Solutions:**
- Increase `max_retries` parameter
- Improve analysis text quality
- Check confidence scores

### JSON Parsing Errors
**Status:** ✅ Fixed!
- System now automatically strips markdown formatting
- Robust error handling with retries
- Should not occur in normal operation

### Databricks API Errors
**Possible Causes:**
- API key not set or invalid
- Network connectivity issues
- Rate limiting

**Solutions:**
- Verify `DATABRICKS_API_KEY` is set correctly
- Check network connection
- Wait and retry if rate limited

## 📚 Documentation

- **FINAL_FIX_SUMMARY.md** - Complete fix summary and status
- **CUSTOM_FACT_EXTRACTION.md** - Detailed technical documentation
- **DATABRICKS_COMPATIBILITY_FIX.md** - Compatibility issues and solutions
- **IMPLEMENTATION_SUMMARY.md** - Implementation details

## 🧪 Testing

### Quick Test
```bash
python test_fact_extraction_fix.py
```

### Full Test Suite
```bash
python test_custom_fact_extraction.py
```

### Expected Output
```
✓ PASSED: Import Test
✓ PASSED: Definition Extraction
✓ PASSED: Percentage Extraction
✓ PASSED: Multiple Amounts Extraction
✓ PASSED: Date Extraction
✓ PASSED: Excel Export Format

Total: 6/6 tests passed
🎉 All tests passed!
```

## 💡 Tips for Best Results

1. **Clear Queries**: Make queries specific and clear
   - Good: "What is the interest rate for Facility A?"
   - Bad: "Tell me about rates"

2. **Quality Analysis**: Ensure analysis text is comprehensive
   - Include relevant context
   - Use complete sentences
   - Avoid ambiguous language

3. **Review Confidence**: Check confidence scores
   - High (0.8-1.0): Reliable extraction
   - Medium (0.5-0.8): Review recommended
   - Low (0.0-0.5): Manual verification needed

4. **Use Batch Processing**: For multiple sections
   - More efficient than individual extractions
   - Consistent formatting
   - Easier to export

5. **Monitor Logs**: Check logs for insights
   - Extraction attempts
   - Validation errors
   - Performance metrics

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Facts appear in the preview table
- ✅ Confidence scores are high (>0.8)
- ✅ Fact names are descriptive and clear
- ✅ Fact values contain the actual data
- ✅ Excel export works correctly
- ✅ No errors in the logs

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs** - Most issues are logged with details
2. **Review documentation** - Check the docs listed above
3. **Run tests** - Validate your setup with the test suite
4. **Check environment** - Verify API keys and configuration
5. **Contact support** - Reach out to CNT Automations team

## 🚀 Next Steps

1. **Test with real documents** - Upload your documents and try it out
2. **Review extracted facts** - Check quality and accuracy
3. **Adjust as needed** - Tune retry settings if needed
4. **Provide feedback** - Share your experience with the team
5. **Explore advanced features** - Try batch processing, custom contexts, etc.

---

**Status**: Production Ready ✅
**Last Updated**: 2025-09-30
**Version**: 1.0

Happy fact extracting! 🎉

