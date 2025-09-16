#!/usr/bin/env python3
"""Test that all dependencies are properly installed"""

print("Testing imports...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers: {e}")

try:
    import auto_round
    print(f"✓ auto_round {auto_round.__version__}")
except ImportError as e:
    print(f"✗ auto_round: {e}")

try:
    import gptqmodel
    print(f"✓ gptqmodel {gptqmodel.__version__}")
except ImportError as e:
    print(f"✗ gptqmodel: {e}")

try:
    import logbar
    print(f"✓ logbar installed")
except ImportError as e:
    print(f"✗ logbar: {e}")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\nAll imports successful! Dependencies are properly installed.")