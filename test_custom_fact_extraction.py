#!/usr/bin/env python3
"""
Test script for the custom LLM-based fact extraction system.
Tests various query types and validates the extraction results.
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, 'src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if required environment variables are set."""
    logger.info("Checking environment...")
    
    api_key = os.environ.get("DATABRICKS_API_KEY")
    if not api_key:
        logger.error("❌ DATABRICKS_API_KEY not found in environment variables")
        logger.info("Please set DATABRICKS_API_KEY in your .env file")
        return False
    
    logger.info(f"✓ DATABRICKS_API_KEY found (starts with: {api_key[:4]}...)")
    return True


def test_import():
    """Test if the fact extraction service can be imported."""
    logger.info("\n=== Testing Imports ===")
    
    try:
        from keyword_code.services.fact_extraction_service import FactExtractionService
        from keyword_code.agents.fact_extraction_agent import (
            FactType, FactTypeAnalysis, FactExtractionResult, ExtractedFact
        )
        logger.info("✓ Successfully imported FactExtractionService and related classes")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import: {e}")
        return False


def test_definition_extraction():
    """Test extraction of definitions."""
    logger.info("\n=== Testing Definition Extraction ===")
    
    try:
        from keyword_code.services.fact_extraction_service import FactExtractionService
        
        service = FactExtractionService()
        
        query = "What is a business day?"
        analysis_text = """
        A Business Day is defined as any day other than a Saturday, Sunday, or a day on which 
        banking institutions in New York are authorized or required by law to close. 
        Business Days are used to calculate payment dates and interest accrual periods.
        """
        
        logger.info(f"Query: {query}")
        logger.info(f"Analysis text length: {len(analysis_text)} chars")
        
        result = service.extract_facts(
            query=query,
            analysis_text=analysis_text,
            context="Test definition extraction"
        )
        
        logger.info(f"Extracted {len(result.extracted_facts)} facts")
        for fact in result.extracted_facts:
            logger.info(f"  - {fact.fact_name} ({fact.fact_type.value}): {fact.fact_value[:100]}...")
            logger.info(f"    Confidence: {fact.confidence}")
        
        if result.errors:
            logger.warning(f"Errors: {result.errors}")
        
        # Validate
        assert len(result.extracted_facts) > 0, "Should extract at least one fact"
        assert any("business day" in fact.fact_name.lower() for fact in result.extracted_facts), \
            "Should extract business day definition"
        
        logger.info("✓ Definition extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Definition extraction test failed: {e}", exc_info=True)
        return False


def test_percentage_extraction():
    """Test extraction of percentages."""
    logger.info("\n=== Testing Percentage Extraction ===")
    
    try:
        from keyword_code.services.fact_extraction_service import FactExtractionService
        
        service = FactExtractionService()
        
        query = "What is the interest rate?"
        analysis_text = """
        The loan agreement specifies an interest rate of 5.5% per annum, calculated on a 
        360-day year basis. Additionally, there is a commitment fee of 0.25% on the undrawn 
        portion of the facility.
        """
        
        logger.info(f"Query: {query}")
        logger.info(f"Analysis text length: {len(analysis_text)} chars")
        
        result = service.extract_facts(
            query=query,
            analysis_text=analysis_text,
            context="Test percentage extraction"
        )
        
        logger.info(f"Extracted {len(result.extracted_facts)} facts")
        for fact in result.extracted_facts:
            logger.info(f"  - {fact.fact_name} ({fact.fact_type.value}): {fact.fact_value}")
            logger.info(f"    Confidence: {fact.confidence}")
        
        if result.errors:
            logger.warning(f"Errors: {result.errors}")
        
        # Validate
        assert len(result.extracted_facts) > 0, "Should extract at least one fact"
        
        logger.info("✓ Percentage extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Percentage extraction test failed: {e}", exc_info=True)
        return False


def test_multiple_amounts_extraction():
    """Test extraction of multiple amounts (e.g., Loan A and Loan B)."""
    logger.info("\n=== Testing Multiple Amounts Extraction ===")
    
    try:
        from keyword_code.services.fact_extraction_service import FactExtractionService
        
        service = FactExtractionService()
        
        query = "What are the loan amounts for Facility A and Facility B?"
        analysis_text = """
        The financing structure consists of two facilities:
        - Facility A: $500 million for refinancing existing debt
        - Facility B: $300 million for greenfield development
        
        The total commitment under both facilities is $800 million.
        """
        
        logger.info(f"Query: {query}")
        logger.info(f"Analysis text length: {len(analysis_text)} chars")
        
        result = service.extract_facts(
            query=query,
            analysis_text=analysis_text,
            context="Test multiple amounts extraction"
        )
        
        logger.info(f"Extracted {len(result.extracted_facts)} facts")
        for fact in result.extracted_facts:
            logger.info(f"  - {fact.fact_name} ({fact.fact_type.value}): {fact.fact_value}")
            logger.info(f"    Confidence: {fact.confidence}")
        
        if result.errors:
            logger.warning(f"Errors: {result.errors}")
        
        # Validate
        assert len(result.extracted_facts) >= 2, "Should extract at least two separate amounts"
        
        logger.info("✓ Multiple amounts extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiple amounts extraction test failed: {e}", exc_info=True)
        return False


def test_date_extraction():
    """Test extraction of dates."""
    logger.info("\n=== Testing Date Extraction ===")
    
    try:
        from keyword_code.services.fact_extraction_service import FactExtractionService
        
        service = FactExtractionService()
        
        query = "When is the maturity date?"
        analysis_text = """
        The loan has a maturity date of December 31, 2030. The first payment is due on 
        March 31, 2025, with subsequent quarterly payments thereafter.
        """
        
        logger.info(f"Query: {query}")
        logger.info(f"Analysis text length: {len(analysis_text)} chars")
        
        result = service.extract_facts(
            query=query,
            analysis_text=analysis_text,
            context="Test date extraction"
        )
        
        logger.info(f"Extracted {len(result.extracted_facts)} facts")
        for fact in result.extracted_facts:
            logger.info(f"  - {fact.fact_name} ({fact.fact_type.value}): {fact.fact_value}")
            logger.info(f"    Confidence: {fact.confidence}")
        
        if result.errors:
            logger.warning(f"Errors: {result.errors}")
        
        # Validate
        assert len(result.extracted_facts) > 0, "Should extract at least one fact"
        
        logger.info("✓ Date extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Date extraction test failed: {e}", exc_info=True)
        return False


def test_excel_export_format():
    """Test that the export format matches requirements (Fact/Definition columns)."""
    logger.info("\n=== Testing Excel Export Format ===")
    
    try:
        from keyword_code.services.fact_extraction_service import FactExtractionService
        
        service = FactExtractionService()
        
        query = "What are the key terms?"
        analysis_text = """
        The loan amount is $500 million with an interest rate of 5.5% per annum.
        The maturity date is December 31, 2030.
        """
        
        # Test the fact-definition extraction method
        rows = service.extract_fact_definitions_from_text(
            text=analysis_text,
            context="Test export format",
            section_name="Key Terms",
            filename="test_document.pdf"
        )
        
        logger.info(f"Extracted {len(rows)} fact-definition pairs")
        
        # Validate format
        for row in rows:
            logger.info(f"  Row: {row}")
            assert "Filename" in row, "Row should have Filename field"
            assert "Section" in row, "Row should have Section field"
            assert "Fact" in row, "Row should have Fact field"
            assert "Definition" in row, "Row should have Definition field"
            assert row["Fact"], "Fact should not be empty"
            assert row["Definition"], "Definition should not be empty"
        
        logger.info("✓ Excel export format test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Excel export format test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Custom LLM-Based Fact Extraction System - Test Suite")
    logger.info("=" * 80)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        return False
    
    # Run tests
    tests = [
        ("Import Test", test_import),
        ("Definition Extraction", test_definition_extraction),
        ("Percentage Extraction", test_percentage_extraction),
        ("Multiple Amounts Extraction", test_multiple_amounts_extraction),
        ("Date Extraction", test_date_extraction),
        ("Excel Export Format", test_excel_export_format),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        return True
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

