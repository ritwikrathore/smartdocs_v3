#!/usr/bin/env python3
"""
Simple test to verify langextract is working with examples.
"""

import langextract as lx

def test_simple_extraction():
    """Test basic langextract functionality."""
    
    # Sample text
    text = "The loan amount is $50 million with an interest rate of 5.5% per annum. The maturity date is December 31, 2025."
    
    # Create examples
    examples = [
        {
            "text": "The investment is $25 million at 3% interest, due January 1, 2024.",
            "extracted_facts": [
                {
                    "text": "Investment amount is $25 million",
                    "category": "financial",
                    "attributes": {"amount": "$25 million", "type": "investment"}
                },
                {
                    "text": "Interest rate is 3%",
                    "category": "financial", 
                    "attributes": {"rate": "3%", "type": "interest"}
                },
                {
                    "text": "Due date is January 1, 2024",
                    "category": "temporal",
                    "attributes": {"date": "January 1, 2024", "type": "due_date"}
                }
            ]
        }
    ]
    
    try:
        print("Testing langextract with gemini...")
        result = lx.extract(
            text_or_documents=text,
            prompt_description="Extract financial amounts, rates, and dates from the text",
            examples=examples,
            model_id="gemini-2.5-flash"
        )
        
        print("✓ Extraction successful!")
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"Full result: {result}")

        if hasattr(result, 'extractions'):
            print(f"Number of extractions: {len(result.extractions)}")
            for i, extraction in enumerate(result.extractions[:3]):  # Show first 3
                print(f"  {i+1}. {extraction}")
        elif isinstance(result, dict) and 'extractions' in result:
            extractions = result['extractions']
            print(f"Number of extractions: {len(extractions)}")
            for i, extraction in enumerate(extractions[:3]):  # Show first 3
                print(f"  {i+1}. {extraction}")
        else:
            print(f"Result structure: {result}")

        return True
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_extraction()
    if success:
        print("\n🎉 LangExtract is working!")
    else:
        print("\n❌ LangExtract test failed")
