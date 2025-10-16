"""
Test script to verify conditional workflow execution for Ask vs Review modes.

This script tests that:
1. Ask mode runs full RAG workflow (decomposition, retrieval, analysis)
2. Review mode skips RAG workflow and returns early
3. Backward compatibility works (defaults to 'ask' mode)
"""

import sys
import logging
from unittest.mock import Mock, patch, MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ask_mode():
    """Test that Ask mode runs full RAG workflow."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Ask Mode - Full RAG Workflow")
    logger.info("="*60)
    
    # Mock the dependencies
    with patch('src.keyword_code.app.DocumentAnalyzer') as mock_analyzer, \
         patch('src.keyword_code.app.decompose_prompt') as mock_decompose, \
         patch('src.keyword_code.app.run_async') as mock_run_async, \
         patch('src.keyword_code.app.PDFProcessor') as mock_pdf_processor, \
         patch('src.keyword_code.app.embedding_model', MagicMock()):
        
        # Setup mocks
        mock_run_async.return_value = [
            {
                "title": "Test Question",
                "sub_prompt": "What is the test?",
                "rag_params": {"bm25_weight": 0.5, "semantic_weight": 0.5, "reasoning": "Test"}
            }
        ]
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.extract_structured_text_and_chunks.return_value = (
            [{"text": "Test chunk", "page": 1}],
            "Test full text"
        )
        mock_pdf_processor.return_value = mock_processor_instance
        
        # Import after patching
        from src.keyword_code.app import process_file_wrapper
        
        # Create test args with 'ask' mode
        test_args = (
            b"fake_pdf_bytes",  # uploaded_file_data
            "test.pdf",  # filename
            "What is the test?",  # user_prompt
            False,  # use_advanced_extraction
            None,  # preprocessed_data_for_file
            'ask'  # mode
        )
        
        try:
            result = process_file_wrapper(test_args)
            
            # Verify decompose_prompt was called
            if mock_run_async.called:
                logger.info("✅ PASS: decompose_prompt was called in Ask mode")
            else:
                logger.error("❌ FAIL: decompose_prompt was NOT called in Ask mode")
                return False
            
            # Verify result structure
            if isinstance(result, dict) and 'filename' in result:
                logger.info("✅ PASS: Result has correct structure")
            else:
                logger.error("❌ FAIL: Result structure is incorrect")
                return False
            
            logger.info("✅ TEST 1 PASSED: Ask mode runs full RAG workflow")
            return True
            
        except Exception as e:
            logger.error(f"❌ TEST 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_review_mode():
    """Test that Review mode skips RAG workflow."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Review Mode - Skip RAG Workflow")
    logger.info("="*60)
    
    # Mock the dependencies
    with patch('src.keyword_code.app.DocumentAnalyzer') as mock_analyzer, \
         patch('src.keyword_code.app.decompose_prompt') as mock_decompose, \
         patch('src.keyword_code.app.run_async') as mock_run_async, \
         patch('src.keyword_code.app.PDFProcessor') as mock_pdf_processor, \
         patch('src.keyword_code.app.embedding_model', MagicMock()):
        
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor_instance.extract_structured_text_and_chunks.return_value = (
            [{"text": "Test chunk", "page": 1}],
            "Test full text"
        )
        mock_pdf_processor.return_value = mock_processor_instance
        
        # Import after patching
        from src.keyword_code.app import process_file_wrapper
        
        # Create test args with 'review' mode
        test_args = (
            b"fake_pdf_bytes",  # uploaded_file_data
            "test.pdf",  # filename
            "Verify all billion values have decimal precision",  # user_prompt
            False,  # use_advanced_extraction
            None,  # preprocessed_data_for_file
            'review'  # mode
        )
        
        try:
            result = process_file_wrapper(test_args)
            
            # Verify decompose_prompt was NOT called
            if not mock_run_async.called:
                logger.info("✅ PASS: decompose_prompt was NOT called in Review mode")
            else:
                logger.error("❌ FAIL: decompose_prompt WAS called in Review mode (should be skipped)")
                return False
            
            # Verify result indicates review mode
            if result.get('mode') == 'review':
                logger.info("✅ PASS: Result indicates review mode")
            else:
                logger.error("❌ FAIL: Result does not indicate review mode")
                return False
            
            # Verify message
            if 'Review mode' in result.get('message', ''):
                logger.info("✅ PASS: Result contains review mode message")
            else:
                logger.error("❌ FAIL: Result does not contain review mode message")
                return False
            
            logger.info("✅ TEST 2 PASSED: Review mode skips RAG workflow")
            return True
            
        except Exception as e:
            logger.error(f"❌ TEST 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_backward_compatibility():
    """Test that old code without mode parameter defaults to 'ask' mode."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Backward Compatibility - Default to Ask Mode")
    logger.info("="*60)
    
    # Mock the dependencies
    with patch('src.keyword_code.app.DocumentAnalyzer') as mock_analyzer, \
         patch('src.keyword_code.app.decompose_prompt') as mock_decompose, \
         patch('src.keyword_code.app.run_async') as mock_run_async, \
         patch('src.keyword_code.app.PDFProcessor') as mock_pdf_processor, \
         patch('src.keyword_code.app.embedding_model', MagicMock()):
        
        # Setup mocks
        mock_run_async.return_value = [
            {
                "title": "Test Question",
                "sub_prompt": "What is the test?",
                "rag_params": {"bm25_weight": 0.5, "semantic_weight": 0.5, "reasoning": "Test"}
            }
        ]
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.extract_structured_text_and_chunks.return_value = (
            [{"text": "Test chunk", "page": 1}],
            "Test full text"
        )
        mock_pdf_processor.return_value = mock_processor_instance
        
        # Import after patching
        from src.keyword_code.app import process_file_wrapper
        
        # Create test args WITHOUT mode parameter (old format)
        test_args = (
            b"fake_pdf_bytes",  # uploaded_file_data
            "test.pdf",  # filename
            "What is the test?",  # user_prompt
            False,  # use_advanced_extraction
            None  # preprocessed_data_for_file
            # NO mode parameter - should default to 'ask'
        )
        
        try:
            result = process_file_wrapper(test_args)
            
            # Verify decompose_prompt was called (defaults to ask mode)
            if mock_run_async.called:
                logger.info("✅ PASS: decompose_prompt was called (defaulted to Ask mode)")
            else:
                logger.error("❌ FAIL: decompose_prompt was NOT called (should default to Ask mode)")
                return False
            
            # Verify result structure
            if isinstance(result, dict) and 'filename' in result:
                logger.info("✅ PASS: Result has correct structure")
            else:
                logger.error("❌ FAIL: Result structure is incorrect")
                return False
            
            logger.info("✅ TEST 3 PASSED: Backward compatibility works (defaults to Ask mode)")
            return True
            
        except Exception as e:
            logger.error(f"❌ TEST 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("CONDITIONAL WORKFLOW TEST SUITE")
    logger.info("="*60)
    
    results = []
    
    # Run tests
    results.append(("Ask Mode", test_ask_mode()))
    results.append(("Review Mode", test_review_mode()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.error("❌ SOME TESTS FAILED")
    logger.info("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

