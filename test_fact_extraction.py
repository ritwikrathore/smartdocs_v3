#!/usr/bin/env python3
"""
Test script to verify fact extraction functionality.
"""

import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, 'src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fact_extractor_import():
    """Test if the fact extractor can be imported."""
    try:
        from keyword_code.langextract_integration.fact_extractor import FactExtractor
        logger.info("✓ FactExtractor imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import FactExtractor: {e}")
        return False

def test_langextract_databricksprovider_import():
    """Test if the databricks provider can be imported."""
    try:
        import langextract_databricksprovider
        logger.info("✓ langextract_databricksprovider imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import langextract_databricksprovider: {e}")
        return False

def test_fact_extraction_basic():
    """Test basic fact extraction functionality."""
    try:
        from keyword_code.langextract_integration.fact_extractor import FactExtractor
        
        # Create fact extractor
        fact_extractor = FactExtractor()
        logger.info("✓ FactExtractor created successfully")
        
        # Test with sample text
        sample_text = """
        The loan amount is $50 million with an interest rate of 5.5% per annum.
        The maturity date is December 31, 2025.
        The borrower is ABC Corporation located in New York.
        """
        
        # Check if API key is available
        if not os.environ.get('DATABRICKS_API_KEY'):
            logger.warning("⚠ DATABRICKS_API_KEY not set - fact extraction will fail")
            return False
        
        # Try fact extraction
        result = fact_extractor.extract_facts_from_text(
            text=sample_text,
            context="Test loan document analysis"
        )
        
        if result:
            logger.info("✓ Fact extraction completed successfully")
            logger.info(f"  - Extracted {len(result.get('extracted_facts', []))} facts")
            for fact in result.get('extracted_facts', [])[:3]:  # Show first 3 facts
                logger.info(f"  - {fact.get('category', 'unknown')}: {fact.get('text', 'no text')}")
            return True
        else:
            logger.warning("⚠ Fact extraction returned no results")
            return False
            
    except Exception as e:
        logger.error(f"✗ Fact extraction test failed: {e}")
        return False

def test_display_function_import():
    """Test if the display function can be imported."""
    try:
        from keyword_code.utils.display import display_section_facts_expander
        logger.info("✓ display_section_facts_expander imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import display_section_facts_expander: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting fact extraction tests...")
    
    tests = [
        ("Import FactExtractor", test_fact_extractor_import),
        ("Import langextract_databricksprovider", test_langextract_databricksprovider_import),
        ("Import display function", test_display_function_import),
        ("Basic fact extraction", test_fact_extraction_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        logger.info("🎉 All tests passed!")
    else:
        logger.warning("⚠ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
