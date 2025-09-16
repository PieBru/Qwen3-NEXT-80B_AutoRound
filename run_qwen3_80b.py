#!/usr/bin/env python3
"""
Unified Qwen3-Next-80B Runner with AutoRound Quantization
Supports CPU-only, GPU-only, and hybrid GPU/CPU modes
"""

import torch
import time
import gc
import psutil
import argparse
import sys
import warnings
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

def print_system_info(verbose=True):
    """Print detailed system information."""
    if not verbose:
        return

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

def check_model_cache():
    """Check if model is cached locally."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = MODEL_ID.replace("/", "--")
    model_path = cache_dir / f"models--{model_cache_name}"

    if model_path.exists():
        # Find snapshot
        snapshots = model_path / "snapshots"
        if snapshots.exists():
            snapshot_dirs = list(snapshots.iterdir())
            if snapshot_dirs:
                snapshot_path = snapshot_dirs[0]
                # Check for model files
                model_files = list(snapshot_path.glob("model-*.safetensors"))
                if model_files:
                    # Calculate total size
                    total_size = sum(f.stat().st_size for f in model_files)
                    return True, snapshot_path, total_size

    return False, None, 0

def estimate_layer_distribution(model_size_gb=40, vram_gb=14, layer_count=80):
    """Estimate how many layers can fit in GPU."""
    bytes_per_layer = (model_size_gb * 1024**3) / layer_count
    vram_bytes = vram_gb * 1024**3

    # Reserve some VRAM for operations
    usable_vram = vram_bytes * 0.9
    layers_in_gpu = int(usable_vram / bytes_per_layer)

    return min(layers_in_gpu, layer_count)

def configure_device_map(args):
    """Configure device mapping based on arguments and available hardware."""
    # Force CPU mode
    if args.cpu or not torch.cuda.is_available():
        if not args.quiet:
            print("📋 Configuration: CPU-only mode")
            if not torch.cuda.is_available():
                print("   Reason: No CUDA GPU detected")
            else:
                print("   Reason: --cpu flag specified")
        return "cpu", None, torch.float32

    # GPU available
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
    ram_available = psutil.virtual_memory().available / (1024**3)

    # Determine mode based on available memory and arguments
    if args.mode == "auto":
        # Auto-detect best mode
        if vram_free >= 40:  # Enough for entire model
            mode = "gpu"
        elif vram_free >= 10:  # Enough for partial model
            mode = "hybrid"
        else:
            mode = "cpu"
    else:
        mode = args.mode

    if not args.quiet:
        print(f"📋 Configuration: {mode.upper()} mode")

    if mode == "gpu":
        # Try to fit entire model on GPU
        gpu_memory = args.gpu_memory if args.gpu_memory else int(vram_free * 0.9)
        if not args.quiet:
            print(f"   GPU allocation: {gpu_memory}GB")
        return "auto", {0: f"{gpu_memory}GiB"}, torch.float16

    elif mode == "hybrid":
        # Distribute between GPU and CPU
        gpu_memory = args.gpu_memory if args.gpu_memory else min(14, int(vram_free * 0.9))
        cpu_memory = args.cpu_memory if args.cpu_memory else int(ram_available * 0.8)

        layers_on_gpu = estimate_layer_distribution(vram_gb=gpu_memory)

        if not args.quiet:
            print(f"   GPU allocation: {gpu_memory}GB")
            print(f"   CPU allocation: {cpu_memory}GB")
            print(f"   Estimated layers on GPU: {layers_on_gpu}/80")
            print(f"   Remaining layers on CPU: {80 - layers_on_gpu}")

        max_memory = {
            0: f"{gpu_memory}GiB",
            "cpu": f"{cpu_memory}GiB"
        }
        return "auto", max_memory, torch.float16

    else:  # cpu mode
        return "cpu", None, torch.float32

def load_model(args):
    """Load the model with the specified configuration."""
    # Check cache
    is_cached, cache_path, cache_size = check_model_cache()
    if not args.quiet:
        if is_cached:
            print(f"📂 Model cached locally: {cache_path}")
            print(f"   Size on disk: {cache_size / (1024**3):.1f}GB")
        else:
            print("📂 Model will be downloaded from Hugging Face")
            print("   This will take time on first run!")

    # Configure device mapping
    device_map, max_memory, dtype = configure_device_map(args)

    # Load tokenizer
    if not args.quiet:
        print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if not args.quiet:
        print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.1f}s")

    # Prepare model loading arguments
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    if max_memory:
        model_kwargs["max_memory"] = max_memory

    # Add offload buffers for hybrid mode
    if device_map == "auto" and max_memory and "cpu" in max_memory:
        model_kwargs["offload_buffers"] = True
        model_kwargs["offload_folder"] = "offload"

    # Load model
    if not args.quiet:
        print(f"\n2. Loading model ({args.mode.upper()} mode)...")
        if device_map == "cpu":
            print("   ⚠️  CPU mode: Requires ~50GB RAM, inference will be slower")
        elif max_memory and "cpu" in max_memory:
            print("   ⚠️  Hybrid mode: Model distributed between GPU and CPU")
        print("   Progress: Loading checkpoint shards...")
        print("   💡 This may take 5-30 minutes depending on hardware")

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

    load_time = time.time() - start_time
    if not args.quiet:
        print(f"\n   ✓ Model loaded successfully in {load_time/60:.1f} minutes")

        # Check actual distribution
        if hasattr(model, 'hf_device_map'):
            gpu_layers = sum(1 for d in model.hf_device_map.values() if d == 0)
            cpu_layers = sum(1 for d in model.hf_device_map.values() if d == 'cpu')
            if gpu_layers > 0 and cpu_layers > 0:
                print(f"   Actual distribution: {gpu_layers} layers on GPU, {cpu_layers} on CPU")

        # Memory usage
        if torch.cuda.is_available() and device_map != "cpu":
            torch.cuda.synchronize()
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"   GPU memory used: {gpu_memory_used:.1f}GB")

        ram_used = psutil.Process().memory_info().rss / (1024**3)
        print(f"   RAM used: {ram_used:.1f}GB")

    return model, tokenizer

def test_generation(model, tokenizer, args):
    """Test model generation with a simple prompt."""
    if args.skip_test:
        return

    if not args.quiet:
        print("\n" + "="*60)
        print("Testing Model Generation")
        print("="*60)

    test_prompt = "What is the capital of France?"
    if not args.quiet:
        print(f"\nTest prompt: {test_prompt}")

    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    # Move inputs to appropriate device
    if torch.cuda.is_available() and hasattr(model, 'device') and model.device.type == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}

    if not args.quiet:
        print("Generating response...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )

    gen_time = time.time() - start_time
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the assistant's response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    if not args.quiet:
        print(f"\nResponse: {response}")
        print(f"Generation time: {gen_time:.1f}s")
        tokens_generated = len(tokenizer.encode(response))
        print(f"Tokens/second: {tokens_generated/gen_time:.2f}")

def interactive_chat(model, tokenizer, args):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print(f"Interactive Chat Mode ({args.mode.upper()})")
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

        # Move to appropriate device
        if torch.cuda.is_available() and hasattr(model, 'device') and model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print(f"\nGenerating ({args.mode} mode)...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )

        gen_time = time.time() - start_time
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(f"\nAssistant: {response}")
        print(f"[Generated in {gen_time:.1f}s]")

        # Memory monitoring in verbose mode
        if args.verbose:
            if torch.cuda.is_available() and hasattr(model, 'device') and model.device.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated() / (1024**3)
                print(f"[GPU: {gpu_mem:.1f}GB used]")
            ram_used = psutil.Process().memory_info().rss / (1024**3)
            print(f"[RAM: {ram_used:.1f}GB used]")

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Next-80B with flexible memory configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best mode (default)
  python run_qwen3_80b.py

  # Force CPU-only mode
  python run_qwen3_80b.py --cpu

  # Force hybrid GPU/CPU mode
  python run_qwen3_80b.py --mode hybrid

  # Custom memory limits
  python run_qwen3_80b.py --gpu-memory 12 --cpu-memory 80

  # Quiet mode with custom generation params
  python run_qwen3_80b.py --quiet --temperature 0.9 --max-tokens 200
        """
    )

    # Mode selection
    parser.add_argument("--mode", choices=["auto", "cpu", "gpu", "hybrid"], default="auto",
                        help="Model loading mode (default: auto)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode (equivalent to --mode cpu)")

    # Memory configuration
    parser.add_argument("--gpu-memory", type=int, metavar="GB",
                        help="GPU memory limit in GB (default: auto)")
    parser.add_argument("--cpu-memory", type=int, metavar="GB",
                        help="CPU memory limit in GB (default: auto)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1)")

    # Output control
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed information")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")

    # Behavior control
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test generation")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Exit after test generation")

    args = parser.parse_args()

    # Handle --cpu flag
    if args.cpu:
        args.mode = "cpu"

    # Validate arguments
    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")

    try:
        if not args.quiet:
            print("="*60)
            print("Qwen3-Next-80B Unified Runner")
            print("="*60)

        print_system_info(verbose=args.verbose or not args.quiet)

        # Check memory requirements
        ram_available = psutil.virtual_memory().available / (1024**3)
        if args.mode == "cpu" and ram_available < 45:
            if not args.quiet:
                print(f"⚠️  Warning: Only {ram_available:.1f}GB RAM available")
                print("   Recommended: 50GB+ for CPU-only mode")
                if not args.verbose:
                    response = input("   Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        print("Exiting...")
                        return 0

        # Load model
        model, tokenizer = load_model(args)

        # Test generation
        test_generation(model, tokenizer, args)

        # Interactive mode
        if not args.no_interactive:
            interactive_chat(model, tokenizer, args)

        return 0

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ Out of memory error")
            print("\n💡 Solutions:")
            print("   1. Use --cpu for CPU-only mode")
            print("   2. Use --mode hybrid for GPU/CPU distribution")
            print("   3. Reduce memory allocation with --gpu-memory")
            print("   4. Close other memory-intensive applications")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif "b_q_weight is not on GPU" in str(e):
            print("\n❌ Quantization error: b_q_weight is not on GPU")
            print("\n💡 Solutions:")
            print("   1. Use --cpu for CPU-only mode")
            print("   2. The quantization library requires all weights on the same device")
        else:
            print(f"\n❌ Runtime Error: {e}")
            if args.verbose:
                traceback.print_exc()
        return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            print("\nDetailed error:")
            traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Try --cpu for CPU-only mode")
        print("   2. Try --mode hybrid for mixed GPU/CPU")
        print("   3. Ensure you have sufficient memory")
        print("   4. Update transformers: pip install --upgrade transformers")
        return 1

if __name__ == "__main__":
    sys.exit(main())