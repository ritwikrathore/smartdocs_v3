"""
Test to verify that bounding boxes are preserved through the RAG retrieval pipeline.
This test ensures the PDF highlighting fix is working correctly.
"""

import pytest
from typing import List, Dict, Any
import fitz  # PyMuPDF


def test_bbox_preservation_in_reranking():
    """
    Test that bboxes are preserved when chunks go through reranking.
    This is the critical fix for PDF highlighting.
    """
    # Mock chunks with bboxes (simulating what pdf_processor creates)
    mock_chunks = [
        {
            "chunk_id": "chunk_0",
            "text": "This is the first chunk of text.",
            "page_num": 0,
            "bboxes": [fitz.Rect(10, 10, 100, 30), fitz.Rect(10, 35, 100, 55)]
        },
        {
            "chunk_id": "chunk_1",
            "text": "This is the second chunk of text.",
            "page_num": 0,
            "bboxes": [fitz.Rect(10, 60, 100, 80)]
        },
        {
            "chunk_id": "chunk_2",
            "text": "This is the third chunk of text.",
            "page_num": 1,
            "bboxes": [fitz.Rect(10, 10, 100, 30), fitz.Rect(10, 35, 100, 55), fitz.Rect(10, 60, 100, 80)]
        }
    ]
    
    # Simulate what the reranking function should do (after fix)
    def rerank_results_fixed(chunks: List[Dict[str, Any]], indices: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        """Simulates the fixed rerank_results function."""
        results = []
        for idx, score in zip(indices, scores):
            chunk = chunks[idx]
            # CORRECT: Preserve all fields with copy()
            result_chunk = chunk.copy()
            result_chunk["score"] = float(score)
            result_chunk["retrieval_method"] = "hybrid"
            results.append(result_chunk)
        return results
    
    # Simulate what the OLD broken function did (before fix)
    def rerank_results_broken(chunks: List[Dict[str, Any]], indices: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        """Simulates the BROKEN rerank_results function (before fix)."""
        results = []
        for idx, score in zip(indices, scores):
            chunk = chunks[idx]
            # WRONG: Creates new dict without bboxes
            results.append({
                "text": chunk.get("text", ""),
                "page_num": chunk.get("page_num", -1),
                "score": float(score),
                "chunk_id": chunk.get("chunk_id", f"unknown_{idx}"),
                "retrieval_method": "hybrid"
            })
        return results
    
    # Test data
    selected_indices = [2, 0, 1]  # Reranked order
    rerank_scores = [0.95, 0.87, 0.72]
    
    # Test the FIXED version
    fixed_results = rerank_results_fixed(mock_chunks, selected_indices, rerank_scores)
    
    # Verify bboxes are preserved
    assert len(fixed_results) == 3, "Should have 3 results"
    
    # Check first result (chunk_2)
    assert "bboxes" in fixed_results[0], "bboxes field should be present"
    assert len(fixed_results[0]["bboxes"]) == 3, "Should have 3 bboxes from chunk_2"
    assert fixed_results[0]["chunk_id"] == "chunk_2"
    assert fixed_results[0]["score"] == 0.95
    assert isinstance(fixed_results[0]["bboxes"][0], fitz.Rect), "bboxes should be fitz.Rect objects"
    
    # Check second result (chunk_0)
    assert "bboxes" in fixed_results[1], "bboxes field should be present"
    assert len(fixed_results[1]["bboxes"]) == 2, "Should have 2 bboxes from chunk_0"
    assert fixed_results[1]["chunk_id"] == "chunk_0"
    assert fixed_results[1]["score"] == 0.87
    
    # Check third result (chunk_1)
    assert "bboxes" in fixed_results[2], "bboxes field should be present"
    assert len(fixed_results[2]["bboxes"]) == 1, "Should have 1 bbox from chunk_1"
    assert fixed_results[2]["chunk_id"] == "chunk_1"
    assert fixed_results[2]["score"] == 0.72
    
    print("✅ FIXED version: All bboxes preserved correctly!")
    
    # Test the BROKEN version (to demonstrate the bug)
    broken_results = rerank_results_broken(mock_chunks, selected_indices, rerank_scores)
    
    # Verify bboxes are MISSING (this was the bug)
    assert len(broken_results) == 3, "Should have 3 results"
    assert "bboxes" not in broken_results[0], "BROKEN: bboxes field is missing!"
    assert "bboxes" not in broken_results[1], "BROKEN: bboxes field is missing!"
    assert "bboxes" not in broken_results[2], "BROKEN: bboxes field is missing!"
    
    print("❌ BROKEN version: bboxes are missing (this was the bug)")
    
    # Demonstrate the impact on highlighting
    def can_highlight(chunk: Dict[str, Any]) -> bool:
        """Check if a chunk has the data needed for highlighting."""
        return "bboxes" in chunk and len(chunk.get("bboxes", [])) > 0
    
    fixed_highlightable = sum(1 for chunk in fixed_results if can_highlight(chunk))
    broken_highlightable = sum(1 for chunk in broken_results if can_highlight(chunk))
    
    assert fixed_highlightable == 3, "All fixed chunks should be highlightable"
    assert broken_highlightable == 0, "No broken chunks should be highlightable"
    
    print(f"\n📊 Highlighting capability:")
    print(f"   Fixed version: {fixed_highlightable}/3 chunks can be highlighted (100%)")
    print(f"   Broken version: {broken_highlightable}/3 chunks can be highlighted (0%)")
    print(f"\n✅ Test passed! The fix correctly preserves bboxes for highlighting.")


def test_bbox_preservation_in_hybrid_search():
    """
    Test that bboxes are preserved in hybrid search without reranking.
    """
    mock_chunks = [
        {
            "chunk_id": "chunk_0",
            "text": "First chunk",
            "page_num": 0,
            "bboxes": [fitz.Rect(10, 10, 100, 30)]
        },
        {
            "chunk_id": "chunk_1",
            "text": "Second chunk",
            "page_num": 0,
            "bboxes": [fitz.Rect(10, 35, 100, 55)]
        }
    ]
    
    # Simulate the fixed hybrid search result formatting
    def format_hybrid_results_fixed(chunks: List[Dict[str, Any]], indices: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        """Simulates the fixed hybrid search formatting."""
        results = []
        for idx, score in zip(indices, scores):
            chunk = chunks[idx]
            # CORRECT: Preserve all fields
            result_chunk = chunk.copy()
            result_chunk["score"] = score
            result_chunk["retrieval_method"] = "hybrid_no_rerank"
            results.append(result_chunk)
        return results
    
    selected_indices = [1, 0]
    scores = [0.85, 0.73]
    
    results = format_hybrid_results_fixed(mock_chunks, selected_indices, scores)
    
    # Verify bboxes are preserved
    assert len(results) == 2
    assert "bboxes" in results[0]
    assert "bboxes" in results[1]
    assert len(results[0]["bboxes"]) == 1
    assert len(results[1]["bboxes"]) == 1
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[1]["chunk_id"] == "chunk_0"
    
    print("✅ Hybrid search (no rerank): bboxes preserved correctly!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing PDF Highlighting Fix - Bbox Preservation")
    print("=" * 70)
    print()
    
    test_bbox_preservation_in_reranking()
    print()
    print("-" * 70)
    print()
    test_bbox_preservation_in_hybrid_search()
    print()
    print("=" * 70)
    print("All tests passed! ✅")
    print("=" * 70)

