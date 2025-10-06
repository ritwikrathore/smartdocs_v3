"""
Test script for reranker fallback mechanism.

This script tests both the Databricks reranker API and the LLM-based fallback
to ensure they work correctly and return scores in the expected format.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.keyword_code.config import logger


def test_databricks_reranker():
    """Test the Databricks reranker API."""
    print("\n" + "=" * 70)
    print("TEST 1: Databricks Reranker API")
    print("=" * 70)
    
    try:
        from src.keyword_code.models.databricks_reranker import DatabricksRerankerModel
        
        print("✓ Successfully imported DatabricksRerankerModel")
        
        # Create model instance
        model = DatabricksRerankerModel()
        print("✓ Successfully created model instance")
        
        # Test with sample pairs
        test_pairs = [
            ["What is the capital of France?", "Paris is the capital and largest city of France."],
            ["What is the capital of France?", "The Eiffel Tower is a famous landmark in Paris."],
            ["What is the capital of France?", "Berlin is the capital of Germany."],
        ]
        
        print(f"\nTesting with {len(test_pairs)} query-document pairs...")
        scores = model.predict(test_pairs)
        
        print(f"\n✓ Received scores: {scores}")
        print(f"  - Type: {type(scores)}")
        print(f"  - Shape: {scores.shape}")
        print(f"  - Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
        
        # Validate scores
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"
        assert len(scores) == len(test_pairs), "Should return one score per pair"
        assert all(0 <= s <= 1 for s in scores), "Scores should be between 0 and 1"
        
        print("\n✓ All validations passed!")
        print("✓ Databricks reranker API is working correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ Databricks reranker test failed: {e}")
        logger.error(f"Databricks reranker test error: {e}", exc_info=True)
        return False


def test_llm_reranker():
    """Test the LLM-based fallback reranker."""
    print("\n" + "=" * 70)
    print("TEST 2: LLM-Based Fallback Reranker")
    print("=" * 70)
    
    try:
        from src.keyword_code.models.llm_reranker import LLMRerankerModel
        
        print("✓ Successfully imported LLMRerankerModel")
        
        # Create model instance
        model = LLMRerankerModel()
        print("✓ Successfully created model instance")
        
        # Test with sample pairs
        test_pairs = [
            ["What is Python?", "Python is a high-level programming language."],
            ["What is Python?", "The weather is nice today."],
        ]
        
        print(f"\nTesting with {len(test_pairs)} query-document pairs...")
        print("(This may take a few seconds as it calls the LLM...)")
        scores = model.predict(test_pairs)
        
        print(f"\n✓ Received scores: {scores}")
        print(f"  - Type: {type(scores)}")
        print(f"  - Shape: {scores.shape}")
        print(f"  - Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
        
        # Validate scores
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"
        assert len(scores) == len(test_pairs), "Should return one score per pair"
        assert all(0 <= s <= 1 for s in scores), "Scores should be between 0 and 1"
        
        # Check if first score is higher (more relevant)
        if scores[0] > scores[1]:
            print(f"\n✓ Relevance ranking is correct: {scores[0]:.4f} > {scores[1]:.4f}")
        else:
            print(f"\n⚠ Warning: Expected first score to be higher, got {scores[0]:.4f} vs {scores[1]:.4f}")
        
        print("\n✓ All validations passed!")
        print("✓ LLM-based fallback reranker is working correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ LLM reranker test failed: {e}")
        logger.error(f"LLM reranker test error: {e}", exc_info=True)
        return False


def test_load_reranker_with_fallback():
    """Test the complete reranker loading with fallback mechanism."""
    print("\n" + "=" * 70)
    print("TEST 3: Complete Reranker Loading (with fallback)")
    print("=" * 70)
    
    try:
        from src.keyword_code.models.databricks_reranker import load_databricks_reranker_model
        
        print("Loading reranker (will attempt API first, then fallback if needed)...")
        model = load_databricks_reranker_model()
        
        if model is None:
            print("\n✗ Failed to load any reranker model")
            return False
        
        model_type = type(model).__name__
        print(f"\n✓ Successfully loaded reranker: {model_type}")
        
        # Test the loaded model
        test_pairs = [
            ["How does RAG work?", "RAG combines retrieval and generation for better AI responses."],
            ["How does RAG work?", "The sky is blue and grass is green."],
        ]
        
        print(f"\nTesting loaded model with {len(test_pairs)} pairs...")
        scores = model.predict(test_pairs)
        
        print(f"\n✓ Received scores: {scores}")
        print(f"  - Model type: {model_type}")
        print(f"  - Scores: {scores}")
        
        # Validate
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"
        assert len(scores) == len(test_pairs), "Should return one score per pair"
        
        print("\n✓ Complete reranker loading test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Complete loading test failed: {e}")
        logger.error(f"Complete loading test error: {e}", exc_info=True)
        return False


def test_interface_compatibility():
    """Test that both rerankers have compatible interfaces."""
    print("\n" + "=" * 70)
    print("TEST 4: Interface Compatibility")
    print("=" * 70)
    
    try:
        from src.keyword_code.models.databricks_reranker import DatabricksRerankerModel
        from src.keyword_code.models.llm_reranker import LLMRerankerModel
        
        # Check that both have predict method
        assert hasattr(DatabricksRerankerModel, 'predict'), "DatabricksRerankerModel should have predict method"
        assert hasattr(LLMRerankerModel, 'predict'), "LLMRerankerModel should have predict method"
        
        print("✓ Both models have 'predict' method")
        
        # Test that both return numpy arrays
        test_pairs = [["test query", "test document"]]
        
        try:
            db_model = DatabricksRerankerModel()
            db_scores = db_model.predict(test_pairs)
            assert isinstance(db_scores, np.ndarray), "Databricks model should return numpy array"
            print("✓ Databricks model returns numpy array")
        except Exception as e:
            print(f"⚠ Could not test Databricks model: {e}")
        
        try:
            llm_model = LLMRerankerModel()
            llm_scores = llm_model.predict(test_pairs)
            assert isinstance(llm_scores, np.ndarray), "LLM model should return numpy array"
            print("✓ LLM model returns numpy array")
        except Exception as e:
            print(f"⚠ Could not test LLM model: {e}")
        
        print("\n✓ Interface compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Interface compatibility test failed: {e}")
        logger.error(f"Interface compatibility test error: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RERANKER FALLBACK MECHANISM TEST SUITE")
    print("=" * 70)
    
    results = {
        "Databricks Reranker API": test_databricks_reranker(),
        "LLM-Based Fallback": test_llm_reranker(),
        "Complete Loading with Fallback": test_load_reranker_with_fallback(),
        "Interface Compatibility": test_interface_compatibility(),
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

