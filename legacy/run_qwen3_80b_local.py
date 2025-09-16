#!/usr/bin/env python3
"""
Load Qwen3-Next-80B directly from local cache without any downloading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import os
from pathlib import Path

def setup_environment():
    """Configure environment for local loading and multi-core usage"""

    # Multi-threading settings for 32 threads
    os.environ["OMP_NUM_THREADS"] = "32"
    os.environ["MKL_NUM_THREADS"] = "32"
    os.environ["OPENBLAS_NUM_THREADS"] = "32"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
    os.environ["NUMEXPR_NUM_THREADS"] = "32"

    # GPU memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Disable all network access for Hugging Face
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Python GIL-free mode
    os.environ["PYTHON_GIL"] = "0"

    # Set torch threads
    torch.set_num_threads(32)
    torch.set_num_interop_threads(32)

    print(f"🔧 Environment configured:")
    print(f"   - CPU threads: 32")
    print(f"   - Offline mode: ENABLED")
    print(f"   - Network access: DISABLED")

def main():
    setup_environment()

    # Use the exact local path to the cached model
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = "models--Intel--Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"
    snapshot_hash = "23ee65d2680f86eb4f6f5b1ef654682daefb8b0a"

    # Full local path to the model files
    local_model_path = cache_dir / model_cache_name / "snapshots" / snapshot_hash

    print("\n" + "=" * 60)
    print("Loading Qwen3-Next-80B from LOCAL cache")
    print("=" * 60)
    print(f"\n📂 Using LOCAL files from:")
    print(f"   {local_model_path}")

    # Verify the path exists
    if not local_model_path.exists():
        print(f"\n❌ Error: Local model path does not exist!")
        print(f"   Expected: {local_model_path}")
        return 1

    # List model files to confirm
    model_files = list(local_model_path.glob("model-*.safetensors"))
    print(f"\n✓ Found {len(model_files)} model shards locally:")
    for f in sorted(model_files)[:3]:
        print(f"   - {f.name}")
    if len(model_files) > 3:
        print(f"   ... and {len(model_files) - 3} more")

    # Load tokenizer from local path
    print("\n1. Loading tokenizer from local files...")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_path),  # Use the local path directly
            local_files_only=True,
            trust_remote_code=True
        )
        print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Error loading tokenizer: {e}")
        return 1

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model from local path
    print("\n2. Loading model from LOCAL cache (NO DOWNLOADING)...")
    print("   This loads from disk, not internet")
    print("   Watch disk I/O, not network activity")
    start_time = time.time()

    try:
        # GPU configuration
        if torch.cuda.is_available():
            print(f"\n   GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   VRAM: {gpu_memory:.1f} GB")

            device_map = "auto"
            max_memory = {
                0: "14GiB",     # Conservative GPU allocation
                "cpu": "100GiB"  # Use system RAM for overflow
            }
            print(f"   Memory plan: GPU=14GiB, CPU=100GiB")
        else:
            device_map = "cpu"
            max_memory = None
            print("   Using CPU only")

        print("\n   Loading model shards from disk...")
        print("   Progress will show as shards are loaded from LOCAL storage")

        # Load model from the exact local path
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),  # Use the local path directly
            dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,  # Enforce local only
            offload_buffers=True,
            offload_folder="/tmp"
        )

        load_time = time.time() - start_time
        print(f"\n   ✓ Model loaded in {load_time:.2f}s")
        print(f"   ✓ Loading speed: {41 / (load_time/60):.1f} GB/min")

        # Show device distribution
        if hasattr(model, 'hf_device_map'):
            print("\n3. Model layer distribution:")
            devices = {}
            for name, device in model.hf_device_map.items():
                devices[device] = devices.get(device, 0) + 1
            for device, count in sorted(devices.items()):
                print(f"   - {device}: {count} layers")

        # Memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n   GPU memory used: {allocated:.1f}/{gpu_memory:.1f} GB")

        # Quick test
        print("\n4. Testing inference...")
        prompt = "What is 2+2?"
        print(f"   Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

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

        print(f"   Response: {response}")
        print(f"   Speed: {20/gen_time:.1f} tokens/s")

        print("\n✅ SUCCESS! Model loaded from LOCAL cache")
        print("   No downloading occurred - purely local loading")

        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Mode (type 'quit' to exit)")
        print("=" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

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
            print("Response:")
            print(response[len(user_input):].strip())

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Debug info:")
        print(f"   - Check path exists: {local_model_path}")
        print(f"   - Check all shards present (should be 9)")
        print(f"   - Run: ls -la {local_model_path}")
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())