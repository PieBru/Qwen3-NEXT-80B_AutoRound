#!/usr/bin/env python3
"""
Qwen3-Next-80B Hybrid GPU/CPU Runner
Optimized for systems with limited VRAM
"""

import torch
import time
import gc
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig
import warnings
import traceback

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

def print_system_info():
    """Print detailed system information."""
    print("\n🔧 System Information:")

    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    print(f"   CPU: {cpu_count} logical threads")

    # RAM info
    ram = psutil.virtual_memory()
    print(f"   RAM: {ram.total / (1024**3):.1f}GB total, {ram.available / (1024**3):.1f}GB available")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {vram:.1f}GB total, {vram_free:.1f}GB free")
    else:
        print("   GPU: Not available")

    print()

def estimate_layer_distribution(model_size_gb=40, vram_gb=14, layer_count=80):
    """Estimate how many layers can fit in GPU."""
    bytes_per_layer = (model_size_gb * 1024**3) / layer_count
    vram_bytes = vram_gb * 1024**3

    # Reserve some VRAM for operations
    usable_vram = vram_bytes * 0.9
    layers_in_gpu = int(usable_vram / bytes_per_layer)

    return min(layers_in_gpu, layer_count)

def main():
    try:
        print("="*60)
        print("Qwen3-Next-80B Hybrid GPU/CPU Mode")
        print("="*60)

        print_system_info()

        if not torch.cuda.is_available():
            print("❌ No CUDA GPU detected. Please use run_qwen3_80b_cpu.py instead.")
            return

        # Get available VRAM
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)

        print("📊 Memory Analysis:")
        print(f"   Model size: ~40GB (4-bit quantized)")
        print(f"   Available VRAM: {vram_free:.1f}GB")

        # Estimate distribution
        gpu_memory_gb = min(12, int(vram_free * 0.9))  # Conservative allocation
        layers_on_gpu = estimate_layer_distribution(vram_gb=gpu_memory_gb)

        print(f"   GPU allocation: {gpu_memory_gb}GB")
        print(f"   Estimated layers on GPU: {layers_on_gpu}/80")
        print(f"   Remaining layers on CPU: {80 - layers_on_gpu}")
        print()

        # Check RAM
        ram_available = psutil.virtual_memory().available / (1024**3)
        ram_needed = 30  # Approximate RAM needed for CPU layers

        if ram_available < ram_needed:
            print(f"⚠️  Warning: Only {ram_available:.1f}GB RAM available")
            print(f"   Recommended: {ram_needed}GB+ for CPU layers")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return

        # Load tokenizer
        print("1. Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.1f}s\n")

        # Configure hybrid loading
        print("2. Configuring hybrid GPU/CPU mode...")

        # Create custom device map
        device_map = "auto"
        max_memory = {
            0: f"{gpu_memory_gb}GiB",  # GPU
            "cpu": f"{int(ram_available * 0.8)}GiB"  # CPU
        }

        print(f"   Memory allocation:")
        print(f"   - GPU: {max_memory[0]}")
        print(f"   - CPU: {max_memory['cpu']}")

        # Load model with offload buffers enabled
        print("\n3. Loading model (hybrid mode)...")
        print("   This will distribute layers between GPU and CPU")
        print("   Progress: Loading checkpoint shards...")

        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            offload_buffers=True,  # Important for handling offloaded layers
            offload_folder="offload",  # Temporary folder for offloaded weights
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start_time
        print(f"\n   ✓ Model loaded successfully in {load_time/60:.1f} minutes")

        # Check actual distribution
        if hasattr(model, 'hf_device_map'):
            gpu_layers = sum(1 for d in model.hf_device_map.values() if d == 0)
            cpu_layers = sum(1 for d in model.hf_device_map.values() if d == 'cpu')
            print(f"   Actual distribution: {gpu_layers} layers on GPU, {cpu_layers} on CPU")

        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"   GPU memory used: {gpu_memory_used:.1f}GB")

        ram_used = psutil.Process().memory_info().rss / (1024**3)
        print(f"   RAM used: {ram_used:.1f}GB")

        # Test generation
        print("\n" + "="*60)
        print("Testing Model Generation")
        print("="*60)

        test_prompt = "What is the capital of France?"
        print(f"\nTest prompt: {test_prompt}")

        messages = [{"role": "user", "content": test_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")

        # Move inputs to first available device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print("Generating response (hybrid mode)...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        gen_time = time.time() - start_time
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(f"\nResponse: {response}")
        print(f"Generation time: {gen_time:.1f}s")

        # Interactive mode
        print("\n" + "="*60)
        print("Interactive Chat Mode (Hybrid GPU/CPU)")
        print("Type 'quit' or 'exit' to end the conversation")
        print("="*60 + "\n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages = [{"role": "user", "content": user_input}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt")

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating (hybrid mode)...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            gen_time = time.time() - start_time
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()

            print(f"\nAssistant: {response}")
            print(f"[Generated in {gen_time:.1f}s]")

            # Memory monitoring
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024**3)
                print(f"[GPU: {gpu_mem:.1f}GB used]")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU out of memory error")
            print("\n💡 Solutions:")
            print("   1. Use run_qwen3_80b_cpu.py for CPU-only mode")
            print("   2. Reduce GPU memory allocation in this script")
            print("   3. Close other GPU applications")
            torch.cuda.empty_cache()
        else:
            print(f"\n❌ Runtime Error: {e}")
            traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Try run_qwen3_80b_cpu.py for CPU-only mode")
        print("   2. Ensure you have sufficient RAM (50GB+)")
        print("   3. Update transformers: pip install --upgrade transformers")

if __name__ == "__main__":
    main()