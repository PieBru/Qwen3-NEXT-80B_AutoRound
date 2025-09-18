#!/usr/bin/env python3
"""
Low-memory version that forces cleanup before IPEX optimization
"""

import os
import sys
import gc
import time
import torch
import psutil

# Disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    print("🧹 Forcing memory cleanup...")
    gc.collect()
    gc.collect()  # Run twice
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(2)  # Give OS time to reclaim

def main():
    print("="*60)
    print("Low-Memory IPEX Optimization Strategy")
    print("="*60)

    # Check initial memory
    print(f"\n📊 Initial memory: {get_memory_usage():.1f}GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("\n⏳ Loading model...")
    start = time.time()

    # Load with aggressive low memory settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Critical flag
        device_map=None,  # Don't use device_map for CPU
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    load_time = time.time() - start
    mem_after_load = get_memory_usage()
    print(f"✅ Model loaded in {load_time/60:.1f} minutes")
    print(f"📊 Memory after load: {mem_after_load:.1f}GB")

    # CRITICAL: Clean up before IPEX
    print("\n🔧 Preparing for IPEX optimization...")
    aggressive_cleanup()
    mem_after_cleanup = get_memory_usage()
    print(f"📊 Memory after cleanup: {mem_after_cleanup:.1f}GB")

    # Check if we have enough headroom
    mem_available = psutil.virtual_memory().available / (1024**3)
    swap_free = psutil.swap_memory().free / (1024**3)
    total_available = mem_available + swap_free

    print(f"📊 Available: {mem_available:.1f}GB RAM + {swap_free:.1f}GB swap = {total_available:.1f}GB total")

    if total_available < 40:
        print("\n⚠️  WARNING: Not enough memory for IPEX optimization!")
        print("   Need at least 40GB free (RAM + swap)")
        print("   Skipping IPEX optimization...")
    else:
        try:
            import intel_extension_for_pytorch as ipex

            print(f"\n🚀 Applying IPEX optimization...")
            print(f"   This will temporarily use ~40GB more memory")

            # Set process priority lower to avoid system freeze
            os.nice(10)

            start = time.time()
            model = ipex.optimize(model, dtype=torch.float16)
            ipex_time = time.time() - start

            mem_after_ipex = get_memory_usage()
            print(f"✅ IPEX optimization complete in {ipex_time:.1f}s")
            print(f"📊 Memory after IPEX: {mem_after_ipex:.1f}GB")

        except MemoryError:
            print("\n❌ Out of memory during IPEX optimization")
            print("   Add more swap space or use --no-ipex")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ IPEX optimization failed: {e}")
            print("   Continuing without IPEX...")

    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60)

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            inputs = tokenizer(user_input, return_tensors="pt")

            print("\nGenerating...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(response[len(user_input):].strip())

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()