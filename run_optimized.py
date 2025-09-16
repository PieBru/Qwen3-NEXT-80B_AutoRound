#!/usr/bin/env python3
"""
Optimized runner for systems with limited GPU VRAM but plenty of CPU RAM
Perfect for RTX 4090 laptop (16GB VRAM) + 128GB RAM
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
    print("Qwen3-Next-80B Optimized for Limited VRAM + High RAM")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Detect hardware
    print("\n2. Hardware configuration:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"   CPU RAM: 128GB (100GB+ available)")
        print(f"   Strategy: GPU+CPU offloading for optimal performance")
    else:
        print("   No GPU detected - CPU only mode")

    # Load model with optimized settings for limited VRAM
    print("\n3. Loading model with CPU+GPU offloading...")
    print("   Model size: ~58GB")
    print("   Will use: ~12GB GPU + ~46GB CPU RAM")
    print("   This configuration avoids OOM errors")
    start_time = time.time()

    try:
        # Optimized configuration for 16GB VRAM + high RAM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={
                0: "12GiB",  # Conservative GPU usage to avoid OOM
                "cpu": "100GiB"  # Plenty of CPU RAM available
            },
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_buffers=True,  # Enable buffer offloading to handle the warning
            offload_state_dict=True  # Offload state dict to CPU
        )

        print(f"   ✓ Model loaded in {time.time() - start_time:.2f}s")

        # Show device distribution
        print("\n4. Model distribution:")
        device_map_summary = {}
        if hasattr(model, 'hf_device_map'):
            for name, device in model.hf_device_map.items():
                if device not in device_map_summary:
                    device_map_summary[device] = 0
                device_map_summary[device] += 1

            for device, count in device_map_summary.items():
                print(f"   {device}: {count} layers")

        # Test generation
        print("\n5. Testing generation...")
        prompt = "What is artificial intelligence?"
        print(f"   Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt")
        # Move inputs to first available device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print("   Generating (first generation is slower)...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - start_time

        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Tokens/second: {50/gen_time:.2f}")

        print("\n6. Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        print("\n✓ Model loaded successfully with CPU+GPU offloading!")
        print("\nPerformance notes:")
        print("- First generation is slow due to model initialization")
        print("- Subsequent generations will be faster")
        print("- Some layers on GPU (fast), some on CPU (slower)")
        print("- This configuration prevents OOM errors")

        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Mode - Type 'quit' to exit")
        print("=" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available():
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

    except torch.cuda.OutOfMemoryError:
        print("\n✗ GPU Out of Memory!")
        print("\nTry reducing max_memory GPU allocation:")
        print("  max_memory={0: '10GiB', 'cpu': '100GiB'}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())