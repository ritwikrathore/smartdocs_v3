"""
Test the billion decimal precision regex pattern to ensure it works correctly.
"""

import re

# The regex pattern from decompose_rule_smartreview (UPDATED with negative lookbehind)
pattern = r"(?<!\.)(?<!\d\.)\b(?:\d{1,3}(?:,\d{3})*|\d+)(?!\.\d+)\s+billion\b"

test_cases = [
    # (text, should_match, description)
    ("Revenue: 67 billion", True, "Integer billion (VIOLATION - should match)"),
    ("Revenue: 67.3 billion", False, "Decimal billion (COMPLIANT - should NOT match)"),
    ("Revenue: 1.0 billion", False, "Decimal billion with .0 (COMPLIANT - should NOT match)"),
    ("Revenue: 1,234 billion", True, "Integer with comma (VIOLATION - should match)"),
    ("Revenue: 1,234.5 billion", False, "Decimal with comma (COMPLIANT - should NOT match)"),
    ("Revenue: 5 billion dollars", True, "Integer billion with text after (VIOLATION - should match)"),
    ("Revenue: 5.5 billion dollars", False, "Decimal billion with text after (COMPLIANT - should NOT match)"),
    ("Revenue: 4.4 billion", False, "Decimal 4.4 billion (COMPLIANT - should NOT match)"),
    ("Revenue: 5.5 billion in", False, "Decimal 5.5 billion (COMPLIANT - should NOT match)"),
    ("Total: 100 billion", True, "Integer 100 billion (VIOLATION - should match)"),
    ("Total: 100.0 billion", False, "Decimal 100.0 billion (COMPLIANT - should NOT match)"),
]

print("Testing billion decimal precision regex pattern")
print("=" * 80)
print(f"Pattern: {pattern}")
print("=" * 80)

all_passed = True
for text, should_match, description in test_cases:
    matches = list(re.finditer(pattern, text))
    matched = len(matches) > 0
    
    status = "✓ PASS" if matched == should_match else "✗ FAIL"
    if matched != should_match:
        all_passed = False
    
    print(f"\n{status}: {description}")
    print(f"  Text: '{text}'")
    print(f"  Expected match: {should_match}, Got: {matched}")
    if matches:
        print(f"  Matched: '{matches[0].group(0)}'")
        print(f"  Match position: {matches[0].start()}-{matches[0].end()}")

print("\n" + "=" * 80)
if all_passed:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed!")
print("=" * 80)

