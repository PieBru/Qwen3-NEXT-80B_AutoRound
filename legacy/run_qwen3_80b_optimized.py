#!/usr/bin/env python3
"""
Optimized script for running Qwen3-Next-80B with proper memory management
Uses local cache and optimizes for limited VRAM
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import os

def main():
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("=" * 60)
    print(f"Loading model: {model_name}")
    print("=" * 60)

    # Set environment variables for multi-threading BEFORE importing transformers
    print("\n🔧 Configuring multi-core environment...")
    cpu_count = os.cpu_count()

    # Enable all CPU cores for parallel processing
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)

    # PyTorch threading
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    print(f"   CPU threads available: {cpu_count}")
    print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"   PyTorch threads: {torch.get_num_threads()}")

    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Python GIL-free mode
    os.environ["PYTHON_GIL"] = "0"

    # Force using local cache
    local_cache = os.path.expanduser("~/.cache/huggingface/hub")
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use offline mode to avoid re-downloading

    print(f"\n📂 Using cached model from: {local_cache}")
    print("   (Offline mode enabled - no downloads)")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True  # Force using local files
    )
    print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Clear memory before model loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model with optimized memory settings
    print("\n2. Loading model with optimized memory settings...")
    print("   Note: Using 4-bit quantization")
    start_time = time.time()

    try:
        # Check system memory
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)
        available_ram = psutil.virtual_memory().available / (1024**3)
        print(f"   System RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

        # Check GPU status
        if torch.cuda.is_available():
            print(f"   GPU detected: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU VRAM: {gpu_memory:.1f} GB total, {allocated:.1f} GB already allocated")

            # Conservative memory settings
            device_map = "auto"

            # Dynamic RAM allocation based on available memory
            cpu_ram_limit = int(available_ram * 0.9)  # Use 90% of available RAM
            max_memory = {
                0: "14GiB",      # Maximum GPU memory to use (leaves headroom)
                "cpu": f"{cpu_ram_limit}GiB"  # Dynamically set based on available RAM
            }

            print(f"   Memory limits: GPU max=14GiB, CPU RAM max={cpu_ram_limit}GiB")
            print("   Note: Most layers will offload to CPU RAM - slower but stable")

        else:
            print("   No GPU detected - will use CPU")
            device_map = "cpu"
            max_memory = None

        print("\n   Loading model shards...")
        print(f"   Using {cpu_count} CPU threads for parallel loading")
        print("   This may take several minutes due to the 80B parameter size")
        print("   Using offload_buffers=True to handle memory constraints")
        print("   Monitor CPU usage with 'htop' - should show multiple cores active")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,  # Using float16 consistently
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,  # Force using local files
            offload_buffers=True,   # Enable buffer offloading for large models
            offload_folder="/tmp"   # Use /tmp for offloading if needed
        )

        print(f"   ✓ Model loaded in {time.time() - start_time:.2f}s")

        # Print device map to understand distribution
        if hasattr(model, 'hf_device_map'):
            print("\n3. Model distribution:")
            devices = {}
            for name, device in model.hf_device_map.items():
                if device not in devices:
                    devices[device] = 0
                devices[device] += 1
            for device, count in devices.items():
                print(f"   - {device}: {count} layers")

        # Memory status after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n   GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        # Simple inference test
        print("\n4. Testing inference...")
        prompt = "What is 2+2?"

        print(f"   Prompt: '{prompt}'")
        print("   Generating response (may be slow with CPU offloading)...")

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available() and model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - start_time

        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Tokens/second: {20/gen_time:.2f}")

        print("\n5. Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        print("\n✓ Model successfully loaded and tested!")
        print("\n⚠️  Note: Inference is slower due to CPU offloading")
        print(f"    This is necessary to run 80B model on {gpu_memory:.0f}GB VRAM")

        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive mode - Type 'quit' to exit")
        print("Note: Responses will be slow due to model size")
        print("=" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available() and model.device.type == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating response (please wait)...")
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
            print("\nResponse:")
            print(response[len(user_input):].strip())  # Only show the generated part

    except Exception as e:
        print(f"\n✗ Error loading or running model: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Suggestions:")
        print("   1. Close other applications to free up RAM")
        print("   2. Try reducing max_memory GPU allocation further")
        print("   3. Ensure you have at least 64GB system RAM available")
        print("   4. Consider using the CPU-only version if GPU issues persist")
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())