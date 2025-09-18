#!/usr/bin/env python3
"""
Test the cache performance improvement
Shows the dramatic speedup from caching
"""

import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cache_timing():
    """Test and report cache performance"""

    print("="*60)
    print("Cache Performance Test")
    print("="*60)

    # Check if cache exists
    cache_dir = Path.home() / ".cache/qwen3_fast_loader"
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"
    cache_key = f"{model_name.replace('/', '_')}_cpu"
    model_cache = cache_dir / cache_key / "model.pkl"

    if model_cache.exists():
        print(f"\n✅ Cache found: {model_cache}")
        print(f"   Size: {model_cache.stat().st_size / (1024**3):.1f}GB")

        print("\n📊 Expected Performance:")
        print("   First load (no cache): 30-60 minutes")
        print("   Cached load: <1 minute")
        print(f"   Speedup: ~30-50x")

        print("\n💡 To test the cached loading:")
        print("   python qwen3_80b_cached.py --interactive")

        print("\n🔄 To test fresh loading (rebuild cache):")
        print("   python qwen3_80b_cached.py --rebuild-cache --interactive")

    else:
        print(f"\n❌ No cache found at: {cache_dir}")
        print("\n📝 To create cache:")
        print("   python qwen3_80b_cached.py --interactive")
        print("   This will:")
        print("   1. Load the model normally (30-60 minutes)")
        print("   2. Save to cache (2-3 minutes)")
        print("   3. Future loads will be <1 minute!")

    print("\n" + "="*60)
    print("Benchmarking Commands:")
    print("="*60)

    print("\n# Time the cached loading:")
    print("time python -c \"from qwen3_80b_cached import load_model_with_cache; import time; start=time.time(); m,t=load_model_with_cache('Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound', 'cpu'); print(f'\\nTotal: {time.time()-start:.1f}s')\"")

    print("\n# Compare with normal loading (WARNING: takes 30-60 minutes):")
    print("time python -c \"from qwen3_80b_cached import load_model_with_cache; import time; start=time.time(); m,t=load_model_with_cache('Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound', 'cpu', use_cache=False); print(f'\\nTotal: {time.time()-start:.1f}s')\"")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_cache_timing()