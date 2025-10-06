#!/usr/bin/env python3
"""
Test the mock fact extraction functionality.
"""

import sys
import logging

# Add the src directory to the path
sys.path.insert(0, 'src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mock_fact_extraction():
    """Test the mock fact extraction."""
    try:
        from keyword_code.langextract_integration.fact_extractor import FactExtractor
        
        # Create fact extractor
        fact_extractor = FactExtractor()
        logger.info("✓ FactExtractor created successfully")
        
        # Test with sample text containing various patterns
        sample_text = """
        The loan amount is $50 million with an interest rate of 5.5% per annum.
        The maturity date is December 31, 2025.
        The borrower is ABC Corporation located in New York.
        The principal amount shall be repaid in quarterly installments.
        There is also a commitment fee of 0.25% on the undrawn amount.
        """
        
        # Try fact extraction
        result = fact_extractor.extract_facts_from_text(
            text=sample_text,
            context="Test loan document analysis"
        )
        
        if result:
            logger.info("✓ Fact extraction completed successfully")
            logger.info(f"  - Model used: {result.get('extraction_metadata', {}).get('model_used', 'unknown')}")
            logger.info(f"  - Total extractions: {result.get('extraction_metadata', {}).get('total_extractions', 0)}")
            
            facts = result.get('extracted_facts', [])
            logger.info(f"  - Extracted {len(facts)} facts:")
            
            for i, fact in enumerate(facts, 1):
                category = fact.get('category', 'unknown')
                text = fact.get('text', 'no text')
                attributes = fact.get('attributes', {})
                logger.info(f"    {i}. [{category}] {text}")
                if attributes:
                    logger.info(f"       Attributes: {attributes}")
            
            return True
        else:
            logger.warning("⚠ Fact extraction returned no results")
            return False
            
    except Exception as e:
        logger.error(f"✗ Fact extraction test failed: {e}")
        return False

def main():
    """Run the test."""
    logger.info("Testing mock fact extraction...")
    
    success = test_mock_fact_extraction()
    
    if success:
        logger.info("\n🎉 Mock fact extraction is working!")
    else:
        logger.warning("\n❌ Mock fact extraction test failed")

if __name__ == "__main__":
    main()
