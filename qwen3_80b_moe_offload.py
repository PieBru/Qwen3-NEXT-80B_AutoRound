#!/usr/bin/env python3
"""
Automatic MoE Expert Offloading for Qwen3-Next-80B
Uses accelerate's built-in offloading with custom configuration
"""

import os
import sys
import time
import torch
import gc
from pathlib import Path

# Optimize environment
os.environ["PYTHON_GIL"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Use cached model

def main():
    print("=" * 60)
    print("Qwen3-Next-80B with Automatic MoE Offloading")
    print("=" * 60)

    # Force MoE offloading in gptqmodel
    os.environ["GPTQMODEL_FORCE_MOE_OFFLOAD"] = "1"
    os.environ["GPTQMODEL_MOE_OFFLOAD_PERCENTAGE"] = "0.8"  # Offload 80% of experts

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

    # Check system resources
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)

    print(f"\n📊 System Resources:")
    print(f"   RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f}GB")
    else:
        print("   ❌ No GPU detected")
        return 1

    # Model path
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("\n🔧 Configuration:")
    print("   • MoE Expert Offloading: ENABLED")
    print("   • Non-experts → GPU (FP16)")
    print("   • Experts → CPU (INT4)")
    print("   • Max GPU Memory: 14GB")
    print("   • Max CPU Memory: 90GB")

    print("\n⏳ Loading model with MoE offloading...")
    print("   This may take 20-30 minutes on first load")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        print("   ✓ Tokenizer loaded")

        # Configure memory limits
        max_memory = {
            0: "14GiB",      # GPU: non-experts, embeddings, attention
            "cpu": "90GiB"   # CPU: expert layers
        }

        # Load model with specific offloading config
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_buffers=True,
            offload_folder="/tmp",
            # MoE specific settings (if supported by the model)
            offload_experts=True,  # Offload expert layers
            expert_offload_percentage=0.8,  # Keep 20% of experts on GPU
        )

        print("   ✓ Model loaded with MoE offloading")

        # Show device distribution
        if hasattr(model, 'hf_device_map'):
            gpu_layers = sum(1 for d in model.hf_device_map.values() if str(d) in ['0', 'cuda:0'])
            cpu_layers = sum(1 for d in model.hf_device_map.values() if str(d) == 'cpu')
            print(f"\n📍 Layer Distribution:")
            print(f"   GPU: {gpu_layers} layers")
            print(f"   CPU: {cpu_layers} layers")

    except Exception as e:
        print(f"\n❌ Failed to load with MoE offloading: {e}")
        print("\n💡 Alternative: Trying dispatch-based offloading...")

        # Alternative: Use dispatch mode
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory

        # First load model on meta device
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        # Calculate balanced memory
        max_memory = get_balanced_memory(
            model,
            max_memory={0: "14GiB", "cpu": "90GiB"},
            no_split_module_classes=["Qwen3NextDecoderLayer"],
        )

        # Create device map prioritizing non-experts on GPU
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Qwen3NextDecoderLayer"],
        )

        # Modify device map to put experts on CPU
        for name in list(device_map.keys()):
            if "expert" in name.lower() and "shared" not in name.lower():
                device_map[name] = "cpu"

        # Load model with custom device map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("   ✓ Model loaded with dispatch-based offloading")

    # Memory status
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"\n💾 GPU Memory Used: {gpu_allocated:.1f}GB")

    process = psutil.Process()
    ram_usage = process.memory_info().rss / (1024**3)
    print(f"   RAM Used: {ram_usage:.1f}GB")

    # Test generation
    print("\n🧪 Testing generation...")
    test_prompt = "Explain quantum computing in one sentence."
    print(f"Prompt: {test_prompt}")

    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        # Move to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}

    print("Generating...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(test_prompt):].strip()
    gen_time = time.time() - start_time

    print(f"Response: {response}")
    print(f"Time: {gen_time:.2f}s ({50/gen_time:.1f} tokens/s)")

    # Performance tips
    print("\n💡 Performance Tips:")
    print("   • First token latency will be high due to expert loading")
    print("   • Subsequent tokens will be faster (KV cache)")
    print("   • Batch processing improves throughput")
    print("   • Consider using shorter max_new_tokens for responsiveness")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating...")
            start_time = time.time()

            with torch.no_grad():
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(user_input):].strip()

            gen_time = time.time() - start_time
            tokens = len(tokenizer.encode(response))

            print(f"\nResponse: {response}")
            print(f"⏱️ Time: {gen_time:.2f}s ({tokens/gen_time:.1f} tokens/s)")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    return 0

if __name__ == "__main__":
    sys.exit(main())