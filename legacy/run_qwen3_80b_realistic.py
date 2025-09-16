#!/usr/bin/env python3
"""
Realistic optimized loader for Qwen3-Next-80B
Acknowledges single-threaded loading limitation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import os
from pathlib import Path

def main():
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("=" * 60)
    print("Qwen3-Next-80B Optimized Loader")
    print("=" * 60)

    # Reality check
    print("\n⚠️  Important Information:")
    print("   - Model loading is single-threaded (transformers limitation)")
    print("   - Expect 20-30 minutes for initial load")
    print("   - Multi-threading only helps during inference")
    print("   - Consider keeping model in memory between runs")

    # Set environment for inference optimization (not loading)
    cpu_count = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use local cache
    os.environ["PYTHON_GIL"] = "0"

    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    print(f"\n🔧 System Configuration:")
    print(f"   CPU: {cpu_count} logical threads available")
    print(f"   Note: Only 1 thread used during loading (library limitation)")
    print(f"   All threads available during inference")

    # Cache path
    cache_dir = Path.home() / ".cache/huggingface/hub"
    local_model_path = cache_dir / "models--Intel--Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound" / "snapshots" / "23ee65d2680f86eb4f6f5b1ef654682daefb8b0a"

    if not local_model_path.exists():
        # Try with model name if direct path doesn't exist
        print(f"\n📂 Model will load from Hugging Face cache")
        local_model_path = model_name
    else:
        print(f"\n📂 Using local cache: {local_model_path}")
        # Check model size
        total_size = sum(f.stat().st_size for f in local_model_path.glob("*.safetensors"))
        print(f"   Model size on disk: {total_size / (1024**3):.1f}GB")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_model_path),
        local_files_only=True if local_model_path != model_name else False,
        trust_remote_code=True
    )
    print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # System memory detection
    import psutil
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)

    # GPU detection
    print("\n2. Preparing for model load...")
    print(f"   System RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f}GB")

        # Conservative allocation for stability
        device_map = "auto"
        # Use 90% of available RAM, leave some for system
        cpu_ram_limit = int(available_ram * 0.9)
        max_memory = {
            0: "14GiB",      # Maximum GPU memory to use (leaving headroom)
            "cpu": f"{cpu_ram_limit}GiB"  # Dynamically set based on available RAM
        }
        print(f"   Memory limits: GPU max=14GiB, CPU RAM max={cpu_ram_limit}GiB")
        print(f"   (Using 90% of available RAM, keeping some for system)")
    else:
        print("   No GPU detected - CPU only mode")
        device_map = "cpu"
        max_memory = None

    # Loading with progress indication
    print("\n3. Loading model (single-threaded, please be patient)...")
    print("   Progress: Loading checkpoint shards...")
    print("   💡 The slow loading is due to:")
    print("      - Single-threaded safetensors loader (transformers limitation)")
    print("      - On-the-fly quantization/dequantization of 4-bit weights")
    print("      - Memory mapping 80B parameters across GPU/CPU")
    print("   💡 Monitor with 'htop' - you'll see 1 thread at 100%")
    print("   💡 This is normal and expected behavior")
    print("\n   ⚠️  You may see a warning about '28GB buffer for offloaded layers'")
    print("      This is expected - we're already using offload_buffers=True")
    print("      The model will work correctly with CPU offloading")

    start_time = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True if local_model_path != model_name else False,
            offload_buffers=True,
            offload_folder="/tmp"
        )

        load_time = time.time() - start_time
        print(f"\n   ✓ Model loaded successfully in {load_time:.1f}s ({load_time/60:.1f} minutes)")

        # Show memory distribution
        if hasattr(model, 'hf_device_map'):
            print("\n4. Model layer distribution:")
            devices = {}
            for name, device in model.hf_device_map.items():
                devices[device] = devices.get(device, 0) + 1

            # Show summary only
            total_layers = sum(devices.values())
            for device, count in sorted(devices.items())[:3]:
                percentage = (count / total_layers) * 100
                print(f"   - {device}: {count} layers ({percentage:.1f}%)")

        # Memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n   GPU memory used: {allocated:.1f}GB")

        # Test inference (this will use multiple threads)
        print("\n5. Testing inference (multi-threaded)...")
        test_prompt = "What is machine learning?"

        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        print(f"   Prompt: '{test_prompt}'")
        print("   Generating (watch htop - should show multiple threads active)...")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - start_time

        print(f"\n   Response: {response[len(test_prompt):].strip()[:100]}...")
        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Speed: {50/gen_time:.1f} tokens/s")

        print("\n✅ Model loaded and ready!")
        print("   Note: Inference IS multi-threaded (unlike loading)")

        # Performance tips
        print("\n💡 Performance Tips:")
        print("   1. Loading is slow due to single-threaded limitation")
        print("   2. Consider using model servers (vLLM, TGI) for production")
        print("   3. Keep model in memory if doing multiple inferences")
        print("   4. Inference speed will be much better than loading speed")

        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Mode - Type 'quit' to exit")
        print("Inference uses all CPU threads for better performance")
        print("=" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

            print("\nGenerating response (multi-threaded)...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_time = time.time() - start_time

            print(f"Time: {gen_time:.2f}s ({150/gen_time:.1f} tokens/s)")
            print("\nResponse:")
            print(response[len(user_input):].strip())

    except KeyboardInterrupt:
        print("\n\n⚠️  Loading interrupted by user")
        print("   This is expected if loading was taking too long")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Troubleshooting:")
        print("   1. Ensure you have enough RAM (64GB+ recommended)")
        print("   2. Close other memory-intensive applications")
        print("   3. Try CPU-only mode if GPU issues persist")
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())