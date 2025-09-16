#!/usr/bin/env python3
"""
Simplified Qwen3-Next-80B runner with better loading strategy
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# Set environment for better performance
os.environ['PYTHON_GIL'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("=" * 60)
    print(f"Loading Qwen3-Next-80B (Simplified)")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Detect hardware
    print("\n2. Hardware detection:")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = "cuda"
    else:
        print("   CPU mode (no GPU detected)")
        device = "cpu"

    # Load model with simplified settings
    print("\n3. Loading model (this WILL take time on first run)...")
    print("   Model is 80GB - be patient!")
    print("   Using 4-bit quantization...")
    start_time = time.time()

    try:
        if device == "cuda":
            # GPU mode - let it auto-distribute
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # CPU mode - simpler loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        print(f"   ✓ Model loaded in {time.time() - start_time:.2f}s")

        # Quick test
        print("\n4. Quick test...")
        prompt = "What is 2+2?"
        print(f"   Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print("   Generating...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Time: {time.time() - start_time:.2f}s")
        print(f"   Response: {response}")

        print("\n✓ Success! Model is working.")
        print("\nEntering interactive mode (type 'quit' to exit)...")
        print("-" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Time: {time.time() - start_time:.2f}s")
            print(f"\nResponse: {response}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough RAM (64GB+ for CPU mode)")
        print("2. Try closing other applications")
        print("3. If on GPU, check VRAM with: nvidia-smi")
        print("4. The model is 80GB - first load takes a LONG time")
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())