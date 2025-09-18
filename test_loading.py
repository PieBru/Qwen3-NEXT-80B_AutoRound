#!/usr/bin/env python3
"""Quick test of loading strategies"""

import subprocess
import sys

strategies = ["no-gpu", "min-gpu", "max-gpu"]

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Testing --load-strategy {strategy}")
    print('='*60)

    # Just try to load the model and exit immediately
    cmd = [
        "./qwen3_80b.py",
        "--load-strategy", strategy,
        "--quiet",
        "--test-deps"  # This will exit after checking dependencies
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if "Error" in result.stderr or "Error" in result.stdout:
        print(f"❌ Strategy {strategy} failed!")
        print("Output:", result.stdout[:500])
        print("Error:", result.stderr[:500])
    else:
        print(f"✅ Strategy {strategy} passed dependency check")
        print("Output:", result.stdout[:200])