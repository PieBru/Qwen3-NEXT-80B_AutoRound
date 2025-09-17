#!/usr/bin/env python3
"""
Qwen3-Next-80B loader WITHOUT GPTQModel
Uses standard transformers loading for better CPU/hybrid compatibility
"""

import os
import sys
import time
import torch
import warnings
from pathlib import Path
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# CRITICAL: Prevent GPTQModel from auto-importing
os.environ["NO_GPTQMODEL"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Optimize environment
os.environ["PYTHON_GIL"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def disable_gptqmodel():
    """Prevent GPTQModel from interfering with model loading"""
    import sys

    # Create a fake gptqmodel module that does nothing
    class FakeGPTQModel:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    # Override the import
    sys.modules['gptqmodel'] = FakeGPTQModel()
    sys.modules['GPTQModel'] = FakeGPTQModel()

    print("✓ GPTQModel disabled - using standard transformers loading")

def load_model_without_gptq(args):
    """Load model using pure transformers without GPTQModel"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import psutil

    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    # Check if cached
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{model_cache_name}"

    if model_path.exists():
        snapshots = model_path / "snapshots"
        snapshot_dirs = list(snapshots.iterdir())
        if snapshot_dirs:
            model_path = snapshot_dirs[0]
            print(f"📂 Using cached model: {model_path}")
            model_name = str(model_path)
            local_only = True
    else:
        print(f"📂 Will download model: {model_name}")
        local_only = False

    # System info
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)

    print(f"\n📊 System Resources:")
    print(f"   RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

    if torch.cuda.is_available() and not args.cpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f}GB")
        device_map = "auto"

        # Configure memory limits for hybrid loading
        max_memory = {
            0: f"{int(gpu_memory * 0.85)}GiB",  # Use 85% of GPU
            "cpu": f"{int(available_ram * 0.75)}GiB"  # Use 75% of RAM
        }
    else:
        print("   Mode: CPU-only")
        device_map = "cpu"
        max_memory = None

    print("\n⏳ Loading model WITHOUT GPTQModel...")
    print("   This enables proper CPU offloading and MoE distribution")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_only
    )
    print("   ✓ Tokenizer loaded")

    # Model loading options
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_only,
        "low_cpu_mem_usage": True,
    }

    if device_map == "auto":
        # For GPU: Use mixed precision and offloading
        load_kwargs.update({
            "device_map": device_map,
            "max_memory": max_memory,
            "torch_dtype": torch.float16,
            "offload_buffers": True,
            "offload_folder": "/tmp",
        })

        if args.load_in_8bit:
            print("   Using 8-bit quantization (bitsandbytes)")
            load_kwargs["load_in_8bit"] = True
            load_kwargs.pop("torch_dtype", None)
        elif args.load_in_4bit:
            print("   Using 4-bit quantization (bitsandbytes)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs.pop("torch_dtype", None)
    else:
        # CPU-only mode
        load_kwargs.update({
            "device_map": None,  # Let model handle placement
            "torch_dtype": torch.float32,  # Full precision for CPU
        })

    try:
        # Load model
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        load_time = time.time() - start_time
        print(f"\n✅ Model loaded in {load_time:.1f}s ({load_time/60:.1f} minutes)")

        # Show device distribution
        if hasattr(model, 'hf_device_map'):
            print("\n📍 Layer Distribution:")
            devices = {}
            expert_on_cpu = 0
            non_expert_on_gpu = 0

            for name, device in model.hf_device_map.items():
                device_str = str(device)
                if "expert" in name.lower():
                    if device_str == "cpu":
                        expert_on_cpu += 1
                elif device_str in ["0", "cuda:0"]:
                    non_expert_on_gpu += 1

                devices[device_str] = devices.get(device_str, 0) + 1

            for device, count in sorted(devices.items()):
                print(f"   {device}: {count} layers")

            if expert_on_cpu > 0:
                print(f"\n   ✓ MoE Offloading Active:")
                print(f"     Experts on CPU: {expert_on_cpu}")
                print(f"     Non-experts on GPU: {non_expert_on_gpu}")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")

        if "out of memory" in str(e).lower():
            print("\n💡 Try these options:")
            print("   --load-in-8bit    : Use 8-bit quantization")
            print("   --load-in-4bit    : Use 4-bit quantization")
            print("   --cpu             : Force CPU-only mode")

        raise

def interactive_mode(model, tokenizer, args):
    """Interactive chat mode"""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Tokenize
            inputs = tokenizer(user_input, return_tensors="pt")

            # Move to model device
            if hasattr(model, 'device'):
                model_device = model.device
            else:
                try:
                    model_device = next(model.parameters()).device
                except:
                    model_device = torch.device('cpu')

            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            print("\nGenerating...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(user_input):].strip()

            gen_time = time.time() - start_time
            tokens = len(tokenizer.encode(response)) - len(tokenizer.encode(user_input))

            print(f"\nResponse: {response}")
            print(f"⏱️  Time: {gen_time:.2f}s ({tokens/gen_time:.1f} tokens/s)")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Next-80B WITHOUT GPTQModel - Better CPU/Hybrid Support"
    )

    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use 8-bit quantization (bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization (bitsandbytes)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens to generate (default: 150)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--test", action="store_true",
                        help="Run test prompt and exit")

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-Next-80B Loader (NO GPTQModel)")
    print("=" * 60)

    # Disable GPTQModel
    disable_gptqmodel()

    # Load model
    try:
        model, tokenizer = load_model_without_gptq(args)
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return 1

    # Test mode
    if args.test:
        print("\n🧪 Running test prompt...")
        test_prompt = "What is the capital of France?"
        print(f"Prompt: {test_prompt}")

        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move to device
        if torch.cuda.is_available() and not args.cpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        return 0

    # Interactive mode
    interactive_mode(model, tokenizer, args)
    return 0

if __name__ == "__main__":
    sys.exit(main())