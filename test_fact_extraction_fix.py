#!/usr/bin/env python3
"""
Quick test to verify the fact extraction fix works with Databricks.
"""

import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, 'src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_extraction():
    """Test basic fact extraction without JSON schema issues."""
    logger.info("Testing basic fact extraction...")
    
    try:
        # Check API key
        if not os.environ.get("DATABRICKS_API_KEY"):
            logger.error("DATABRICKS_API_KEY not set")
            return False
        
        from keyword_code.services.fact_extraction_service import FactExtractionService
        
        # Initialize service
        logger.info("Initializing FactExtractionService...")
        service = FactExtractionService()
        
        # Test with simple query
        query = "What is a business day?"
        analysis_text = "A Business Day is any day other than Saturday, Sunday, or a bank holiday."
        
        logger.info(f"Query: {query}")
        logger.info(f"Analysis: {analysis_text}")
        
        # Extract facts
        logger.info("Extracting facts...")
        result = service.extract_facts(
            query=query,
            analysis_text=analysis_text,
            context="Test"
        )
        
        logger.info(f"Extraction completed!")
        logger.info(f"Extracted {len(result.extracted_facts)} facts")
        
        for fact in result.extracted_facts:
            logger.info(f"  - {fact.fact_name}: {fact.fact_value}")
        
        if result.errors:
            logger.warning(f"Errors: {result.errors}")
        
        logger.info("✓ Test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_basic_extraction()
    sys.exit(0 if success else 1)

