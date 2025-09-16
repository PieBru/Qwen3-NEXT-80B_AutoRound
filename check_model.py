#!/usr/bin/env python3
"""
Check model download status and cache
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download, model_info

model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

print("Checking model information...")
try:
    # Get model info
    info = model_info(model_name)
    print(f"Model ID: {info.id}")
    print(f"Model size: ~50GB")

    # Check cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / f"models--{model_name.replace('/', '--')}"

    if model_cache.exists():
        # Get size of cached files
        total_size = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file())
        print(f"\nCached data: {total_size / (1024**3):.2f} GB")

        # Check for key files
        safetensors_files = list(model_cache.rglob("*.safetensors"))
        print(f"Safetensors files found: {len(safetensors_files)}")

        if safetensors_files:
            print("\nCached model files:")
            for f in sorted(safetensors_files)[:5]:  # Show first 5
                size_gb = f.stat().st_size / (1024**3)
                print(f"  - {f.name}: {size_gb:.2f} GB")
            if len(safetensors_files) > 5:
                print(f"  ... and {len(safetensors_files) - 5} more files")
    else:
        print("\nModel not cached yet. First run will download ~50GB.")

    print("\nTo download/verify model files, you can run:")
    print(f"  huggingface-cli download {model_name}")

except Exception as e:
    print(f"Error checking model: {e}")
    print("\nMake sure you're logged in:")
    print("  huggingface-cli login")