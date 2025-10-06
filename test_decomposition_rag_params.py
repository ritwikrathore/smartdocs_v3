"""
Test script to verify decomposition with RAG parameters.
"""
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from keyword_code.ai.analyzer import DocumentAnalyzer
from keyword_code.ai.decomposition import decompose_prompt
from keyword_code.config import logger

async def test_decomposition():
    """Test the decomposition with RAG parameters."""
    
    # Initialize analyzer
    analyzer = DocumentAnalyzer()
    
    # Test prompts
    test_prompts = [
        "What is the MT599 Swift message format and field 79 content?",
        "What are the interest rates and fees for this loan?",
        "Analyze the termination clause and liability limitations in the loan agreement.",
        "What is the loan amount, currency, and duration of the availability period?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {prompt}")
        print('='*80)
        
        try:
            result = await decompose_prompt(analyzer, prompt)
            
            print(f"\nDecomposed into {len(result)} sub-prompts:\n")
            
            for j, item in enumerate(result, 1):
                print(f"\n--- Sub-prompt {j} ---")
                print(f"Title: {item.get('title')}")
                print(f"Sub-prompt: {item.get('sub_prompt')}")
                
                rag_params = item.get('rag_params', {})
                print(f"\nRAG Parameters:")
                print(f"  BM25 Weight: {rag_params.get('bm25_weight', 'N/A')}")
                print(f"  Semantic Weight: {rag_params.get('semantic_weight', 'N/A')}")
                print(f"  Reasoning: {rag_params.get('reasoning', 'N/A')}")
                
                # Validate weights
                bm25 = rag_params.get('bm25_weight', 0)
                semantic = rag_params.get('semantic_weight', 0)
                weight_sum = bm25 + semantic
                
                if abs(weight_sum - 1.0) > 0.01:
                    print(f"  ⚠️  WARNING: Weights don't sum to 1.0 (sum={weight_sum:.3f})")
                else:
                    print(f"  ✓ Weights sum to 1.0")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Test failed for prompt: {prompt}", exc_info=True)

if __name__ == "__main__":
    print("Testing Decomposition with RAG Parameters")
    print("="*80)
    asyncio.run(test_decomposition())
    print("\n" + "="*80)
    print("Testing complete!")

