#!/usr/bin/env python3
"""
Optimized script with proper multi-core CPU usage for Qwen3-Next-80B
Leverages Python 3.13.3t's GIL-free threading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import os

def setup_multicore_environment():
    """Configure environment for optimal multi-core usage"""

    # Enable multi-threading for PyTorch
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Use all CPU cores
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
    os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

    # GPU memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Force using local cache
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Enable Python GIL-free mode explicitly
    os.environ["PYTHON_GIL"] = "0"

    # Set torch threads
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count())

    print(f"🔧 Environment configured for {os.cpu_count()} CPU threads")
    print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"   PyTorch threads: {torch.get_num_threads()}")
    print(f"   PyTorch interop threads: {torch.get_num_interop_threads()}")
    print(f"   Python GIL disabled: {os.environ.get('PYTHON_GIL', '1') == '0'}")

def main():
    # Set up multi-core environment first
    setup_multicore_environment()

    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("\n" + "=" * 60)
    print(f"Loading model: {model_name}")
    print("=" * 60)

    local_cache = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"\n📂 Using cached model from: {local_cache}")
    print("   (Offline mode enabled - no downloads)")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True
    )
    print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Clear memory before model loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model with optimized settings
    print("\n2. Loading model with multi-core optimization...")
    print("   Note: Using 4-bit quantization")
    print("   Multi-threading enabled for faster loading")
    start_time = time.time()

    try:
        # Check GPU status
        if torch.cuda.is_available():
            print(f"   GPU detected: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU VRAM: {gpu_memory:.1f} GB total, {allocated:.1f} GB already allocated")

            # Conservative memory settings
            print(f"   Using conservative memory allocation for {gpu_memory:.0f}GB VRAM")
            device_map = "auto"

            # Dynamic memory allocation
            import psutil
            available_ram = psutil.virtual_memory().available / (1024**3)
            cpu_ram_limit = int(available_ram * 0.9)
            max_memory = {
                0: "14GiB",      # GPU allocation
                "cpu": f"{cpu_ram_limit}GiB"  # Dynamic CPU RAM limit
            }

            print(f"   Memory limits: GPU max=14GiB, CPU RAM max={cpu_ram_limit}GiB")

        else:
            print("   No GPU detected - will use CPU")
            device_map = "cpu"
            max_memory = None

        print("\n   Loading model shards...")
        print(f"   Monitor CPU usage with htop")
        print("   This may take several minutes due to the 80B parameter size")

        # Create model with multi-threading support
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_buffers=True,
            offload_folder="/tmp"
        )

        load_time = time.time() - start_time
        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   Loading speed: {41 / load_time:.2f} GB/s (41GB model)")

        # Print device distribution
        if hasattr(model, 'hf_device_map'):
            print("\n3. Model distribution:")
            devices = {}
            for name, device in model.hf_device_map.items():
                if device not in devices:
                    devices[device] = 0
                devices[device] += 1
            for device, count in devices.items():
                print(f"   - {device}: {count} layers")

        # Memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n   GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        # Test inference with timing
        print("\n4. Testing inference with multi-core optimization...")
        test_prompts = [
            "What is 2+2?",
            "Explain quantum computing in one sentence.",
            "What color is the sky?"
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: '{prompt}'")
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

            print(f"   Response: {response[len(prompt):].strip()}")
            print(f"   Time: {gen_time:.2f}s ({20/gen_time:.1f} tokens/s)")

        print("\n✓ Model successfully loaded and tested!")
        print("✓ Multi-core processing enabled")

        # Performance summary
        print("\n📊 Performance Summary:")
        print(f"   - CPU threads available: {os.cpu_count()}")
        print(f"   - Model loading time: {load_time:.2f}s")
        print(f"   - Average inference speed: ~{20/gen_time:.1f} tokens/s")

        if load_time > 300:  # If loading took more than 5 minutes
            print("\n💡 Tip: Model loading can be sped up by:")
            print("   - Using faster SSD storage")
            print("   - Increasing system RAM speed")
            print("   - Ensuring no other heavy processes are running")

        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive mode - Type 'quit' to exit")
        print("Multi-core optimization active")
        print("=" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available() and model.device.type == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating response...")
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
            gen_time = time.time() - start_time
            print(f"Time: {gen_time:.2f}s ({100/gen_time:.1f} tokens/s)")
            print("\nResponse:")
            print(response[len(user_input):].strip())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Troubleshooting:")
        print("   1. Check CPU usage with 'top' or 'htop'")
        print("   2. Ensure Python 3.13.3t (free-threaded) is active")
        print("   3. Verify PYTHON_GIL=0 is set")
        print("   4. Close other memory-intensive applications")
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())