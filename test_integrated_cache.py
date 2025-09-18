#!/usr/bin/env python3
"""
Test the integrated cache functionality in the main qwen3_80b.py script
"""

import subprocess
import time
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return output safely"""
    # Validate command is a simple Python script execution
    if not cmd.startswith("python "):
        raise ValueError(f"Only Python script execution allowed, got: {cmd}")

    # Check for dangerous shell characters
    dangerous_chars = [';', '&', '|', '`', '$', '>', '<', '(', ')', '{', '}']
    if any(char in cmd for char in dangerous_chars):
        raise ValueError(f"Command contains dangerous characters: {cmd}")

    print(f"\n🔧 Running: {cmd}")
    # Use list of args instead of shell=True for safety
    cmd_parts = cmd.split()
    result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=30)
    return result

def test_cache_integration():
    """Test the cache integration in main script"""

    print("="*60)
    print("Testing Cache Integration in qwen3_80b.py")
    print("="*60)

    # Check if cache exists
    cache_dir = Path.home() / ".cache/qwen3_fast_loader"

    print("\n1. Checking cache status...")
    result = run_command("python qwen3_80b.py --check")
    if result.returncode == 0:
        print("   ✅ Check command works")

    print("\n2. Testing cache clear...")
    result = run_command("python qwen3_80b.py --clear-cache")
    if result.returncode == 0:
        print("   ✅ Clear cache command works")
        if not cache_dir.exists():
            print("   ✅ Cache directory removed")

    print("\n3. Testing dry run with cache option...")
    result = run_command("python qwen3_80b.py --use-cache --dry-run")
    if result.returncode == 0:
        print("   ✅ Dry run with cache works")

    print("\n" + "="*60)
    print("Test Commands to Try:")
    print("="*60)

    print("\n# Test cache loading (will create cache on first run):")
    print("python qwen3_80b.py --use-cache --load-strategy no-gpu --temperature 0.7 --max-tokens 50")

    print("\n# Test cache with thinking mode:")
    print("python qwen3_80b.py --use-cache --thinking --load-strategy no-gpu")

    print("\n# Interactive with cache:")
    print("python qwen3_80b.py --use-cache --interactive --load-strategy no-gpu")

    print("\n# Clear and rebuild cache:")
    print("python qwen3_80b.py --clear-cache")
    print("python qwen3_80b.py --use-cache --rebuild-cache --load-strategy no-gpu")

    print("\n" + "="*60)
    print("Cache Performance Expectations:")
    print("="*60)
    print("First run with --use-cache:")
    print("  1. Normal loading (30-60 minutes)")
    print("  2. Cache creation (2-3 minutes)")
    print("  3. Total: ~30-35 minutes")
    print("\nSubsequent runs with --use-cache:")
    print("  1. Cache loading (<1 minute)")
    print("  2. Ready to use!")
    print("\n" + "="*60)

if __name__ == "__main__":
    test_cache_integration()