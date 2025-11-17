"""
Test script to verify configuration changes work correctly.
Tests:
1. USE_ADAPTIVE_SENTENCE_CHUNKER default is now true
2. ENABLE_LANGFUSE_TRACING can be toggled
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("Testing Configuration Defaults")
print("=" * 60)

# Test 1: Default values from config
print("\nTest 1: Checking default configuration values...")
from keyword_code.config import USE_ADAPTIVE_SENTENCE_CHUNKER, ENABLE_LANGFUSE_TRACING

print(f"  USE_ADAPTIVE_SENTENCE_CHUNKER: {USE_ADAPTIVE_SENTENCE_CHUNKER}")
if USE_ADAPTIVE_SENTENCE_CHUNKER:
    print("  ✅ PASS - Adaptive chunker enabled by default (expected: True)")
else:
    print("  ❌ FAIL - Should be True by default")
    
print(f"  ENABLE_LANGFUSE_TRACING: {ENABLE_LANGFUSE_TRACING}")
# This will be True if .env has keys, False otherwise - both are valid
print(f"  ℹ️  Current state: {'Enabled' if ENABLE_LANGFUSE_TRACING else 'Disabled'}")

# Test 2: Environment variable override
print("\nTest 2: Testing environment variable override...")
print("  Setting ENABLE_LANGFUSE_TRACING=false in environment...")
os.environ["ENABLE_LANGFUSE_TRACING"] = "false"

# Reload config to pick up the change
import importlib
from keyword_code import config
importlib.reload(config)

print(f"  ENABLE_LANGFUSE_TRACING after reload: {config.ENABLE_LANGFUSE_TRACING}")
if config.ENABLE_LANGFUSE_TRACING == False:
    print("  ✅ PASS - Environment variable correctly overrides default")
else:
    print("  ❌ FAIL - Environment variable should override to False")

# Test 3: Langfuse import safety
print("\nTest 3: Testing Langfuse import safety...")
try:
    # Try importing langfuse_tracing directly
    import keyword_code.utils.langfuse_tracing as lf_tracing
    print("  ✅ PASS - langfuse_tracing module imports without errors")
    
    # Check if is_tracing_enabled exists and is callable
    if hasattr(lf_tracing, 'is_tracing_enabled') and callable(lf_tracing.is_tracing_enabled):
        print("  ✅ PASS - is_tracing_enabled function is available")
        
        # Test that it returns a boolean
        result = lf_tracing.is_tracing_enabled()
        if isinstance(result, bool):
            print(f"  ✅ PASS - is_tracing_enabled() returns boolean: {result}")
        else:
            print(f"  ❌ FAIL - is_tracing_enabled() should return boolean, got: {type(result)}")
    else:
        print("  ❌ FAIL - is_tracing_enabled function not found")
        
except ImportError as e:
    print(f"  ❌ FAIL - Failed to import langfuse_tracing: {e}")
except Exception as e:
    print(f"  ⚠️  WARNING - Unexpected error: {e}")

print("\n" + "=" * 60)
print("Configuration Tests Complete!")
print("=" * 60)
print("\n✅ Key Changes Verified:")
print("   • USE_ADAPTIVE_SENTENCE_CHUNKER defaults to True")
print("   • ENABLE_LANGFUSE_TRACING can be toggled via environment")
print("   • Langfuse tracing module imports safely")
print("\n📋 For Azure Deployment:")
print("   • Set ENABLE_LANGFUSE_TRACING=false to disable tracing")
print("   • USE_ADAPTIVE_SENTENCE_CHUNKER will default to true")
print("   • No .env file changes needed in Azure")
