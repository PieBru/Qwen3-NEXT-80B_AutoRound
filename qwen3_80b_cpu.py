#!/usr/bin/env python3
"""
CPU-only version for Qwen3-Next-80B AutoRound model
Specifically designed to work around quantization library CUDA requirements
"""

import os
import sys
import time
import torch
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Force CPU mode and disable CUDA completely
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHON_GIL"] = "0"

def main():
    print("=" * 60)
    print("Qwen3-Next-80B CPU-Only Mode")
    print("=" * 60)

    # Check if model is cached
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"
    model_cache_name = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{model_cache_name}"

    if not model_path.exists():
        print("\n❌ Model not cached. Please run the main script first to download:")
        print("   python qwen3_80b.py --check")
        return 1

    # Find snapshot
    snapshots = model_path / "snapshots"
    snapshot_dirs = list(snapshots.iterdir())
    if not snapshot_dirs:
        print("❌ No snapshot found")
        return 1

    snapshot_path = snapshot_dirs[0]
    print(f"\n📂 Model found: {snapshot_path}")

    print("\n⚠️  IMPORTANT: CPU-only mode for this quantized model has limitations:")
    print("   • The GPTQModel library has internal CUDA kernels")
    print("   • Full CPU-only inference may not work correctly")
    print("   • Consider using a different quantization format for CPU")
    print()
    print("Alternative solutions:")
    print("1. Use a GPU with 30GB+ VRAM")
    print("2. Use GGUF format models with llama.cpp for CPU inference")
    print("3. Use unquantized models with bitsandbytes for mixed precision")
    print()

    # Try loading anyway
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("⏳ Attempting to load model on CPU...")
    print("   This may fail due to quantization library limitations")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=True
        )
        print("   ✓ Tokenizer loaded")

        # Try loading model without device_map
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot_path),
            dtype=torch.float32,  # Use FP32 for CPU
            device_map=None,  # No device mapping
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            torch_dtype=torch.float32
        )

        print("   ✓ Model loaded")

        # Test generation
        test_input = "What is 2+2?"
        print(f"\nTest prompt: {test_input}")

        inputs = tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

        print("\n✅ Model works on CPU!")

    except Exception as e:
        print(f"\n❌ Failed to load model on CPU: {e}")
        print("\nThis is expected - the quantized model requires CUDA.")
        print("Please use one of the alternative solutions listed above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())