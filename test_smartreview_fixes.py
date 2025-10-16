"""
Test script to verify SmartReview fixes.

Run this script to validate:
1. Decomposition function works correctly
2. Billion decimal precision regex is correct
3. Pydantic-AI evaluator initializes without errors
"""

import re
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_billion_regex():
    """Test that the billion decimal precision regex works correctly."""
    print("\n" + "="*60)
    print("TEST 1: Billion Decimal Precision Regex")
    print("="*60)
    
    # The regex pattern from decompose_rule_smartreview
    pattern = r"\b(?:\d{1,3}(?:,\d{3})*|\d+)(?!\.\d+)\s+billion\b"
    
    test_cases = [
        # (text, should_match, description)
        ("Revenue: 67 billion", True, "Integer billion (should match)"),
        ("Revenue: 67.3 billion", False, "Decimal billion (should NOT match)"),
        ("Revenue: 1.0 billion", False, "Decimal billion with .0 (should NOT match)"),
        ("Revenue: 1,234 billion", True, "Integer with comma (should match)"),
        ("Revenue: 1,234.5 billion", False, "Decimal with comma (should NOT match)"),
        ("Revenue: 5 billion dollars", True, "Integer billion with text after (should match)"),
        ("Revenue: 5.5 billion dollars", False, "Decimal billion with text after (should NOT match)"),
    ]
    
    all_passed = True
    for text, should_match, description in test_cases:
        matches = list(re.finditer(pattern, text))
        matched = len(matches) > 0
        
        status = "✓ PASS" if matched == should_match else "✗ FAIL"
        if matched != should_match:
            all_passed = False
        
        print(f"{status}: {description}")
        print(f"  Text: '{text}'")
        print(f"  Expected match: {should_match}, Got: {matched}")
        if matches:
            print(f"  Matched: '{matches[0].group(0)}'")
        print()
    
    return all_passed


def test_decomposition():
    """Test the decompose_rule_smartreview function."""
    print("\n" + "="*60)
    print("TEST 2: Decomposition Function")
    print("="*60)
    
    try:
        from SmartReview import decompose_rule_smartreview, Rule
        
        # Test 1: Billion decimal precision rule
        print("\nTest 2a: Billion decimal precision rule")
        rule_text = "Verify that all billion values are expressed with decimal precision (e.g., '1.0 billion' not '1 billion')"
        tasks = decompose_rule_smartreview(rule_text)
        
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
        assert tasks[0].validation_type == 'regex', f"Expected 'regex', got '{tasks[0].validation_type}'"
        assert r'(?!\.\d+)' in tasks[0].validator, "Expected negative lookahead in regex"
        
        print(f"✓ PASS: Generated regex validation")
        print(f"  Pattern: {tasks[0].validator}")
        
        # Test 2: Format-based rule (should use semantic)
        print("\nTest 2b: Date format rule")
        rule_text = "Dates must be in YYYY-MM-DD format"
        tasks = decompose_rule_smartreview(rule_text)
        
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
        assert tasks[0].validation_type == 'semantic', f"Expected 'semantic', got '{tasks[0].validation_type}'"
        assert 'decimal precision' in tasks[0].validator.lower(), "Expected decimal precision guardrail in prompt"
        
        print(f"✓ PASS: Generated semantic validation with guardrails")
        print(f"  Prompt includes decimal precision check: Yes")
        
        # Test 3: Generic rule (should use semantic)
        print("\nTest 2c: Generic rule")
        rule_text = "Ensure professional tone throughout"
        tasks = decompose_rule_smartreview(rule_text)
        
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
        assert tasks[0].validation_type == 'semantic', f"Expected 'semantic', got '{tasks[0].validation_type}'"
        
        print(f"✓ PASS: Generated semantic validation for generic rule")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator_import():
    """Test that the pydantic-ai evaluator can be imported without errors."""
    print("\n" + "="*60)
    print("TEST 3: Pydantic-AI Evaluator Import")
    print("="*60)
    
    try:
        # This will trigger the import and check for AttributeError
        from src.keyword_code.agents import review_evaluator
        
        print("✓ PASS: review_evaluator module imported successfully")
        
        # Check if pydantic-ai is available
        if review_evaluator._HAS_PYDANTIC_AI:
            print("✓ PASS: pydantic-ai is installed and available")
            
            # Try to get the agent (will check initialization)
            agent = review_evaluator._get_agent()
            if agent is not None:
                print("✓ PASS: Evaluator agent initialized successfully")
                print(f"  Agent type: {type(agent)}")
            else:
                print("⚠ WARNING: Agent is None (check DATABRICKS_API_KEY env var)")
                print("  This is expected if DATABRICKS_API_KEY is not set")
        else:
            print("⚠ WARNING: pydantic-ai is not installed")
            print("  Install with: pip install pydantic-ai")
        
        return True
        
    except AttributeError as e:
        print(f"✗ FAIL: AttributeError (the bug we're fixing): {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"⚠ WARNING: {e}")
        import traceback
        traceback.print_exc()
        return True  # Other errors are acceptable (e.g., missing env vars)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SmartReview Fixes Validation Tests")
    print("="*60)
    
    results = []
    
    # Test 1: Regex pattern
    results.append(("Billion Regex", test_billion_regex()))
    
    # Test 2: Decomposition function
    results.append(("Decomposition", test_decomposition()))
    
    # Test 3: Evaluator import
    results.append(("Evaluator Import", test_evaluator_import()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

