#!/usr/bin/env python3
"""
Unified CLI for Qwen3-Next-80B AutoRound model
Combines all functionality from various scripts
"""

import argparse
import os
import sys
import time
import gc
import torch
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

# Suppress warnings unless in verbose mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def setup_environment(threads: Optional[int] = None, verbose: bool = False, offline: bool = False) -> int:
    """Configure environment for optimal performance"""
    cpu_count = threads or os.cpu_count()

    # Threading configuration
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if var not in os.environ:  # Only set if not already configured
            os.environ[var] = str(cpu_count)
        elif verbose:
            print(f"   ℹ️  {var} already set to {os.environ[var]}")

    # GPU configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Python GIL
    os.environ["PYTHON_GIL"] = "0"

    # Offline mode if requested
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if verbose:
            print("   📡 Running in offline mode")

    # PyTorch threads
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    if verbose:
        print(f"🔧 Environment configured for {cpu_count} threads")

    return cpu_count


def check_model_cache(model_name: str = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound") -> tuple[bool, Path, int]:
    """Check if model is cached locally and calculate total cache size"""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = model_name.replace("/", "--")
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
                    # Calculate total size including all relevant files
                    total_size = 0

                    # Model safetensor files
                    total_size += sum(f.stat().st_size for f in model_files)

                    # Config and tokenizer files
                    for pattern in ["*.json", "*.txt", "tokenizer*", "special_tokens_map.json",
                                   "generation_config.json", "*.tiktoken", "*.model"]:
                        total_size += sum(f.stat().st_size for f in snapshot_path.glob(pattern) if f.is_file())

                    # Also check for any .bin files (some models use these)
                    total_size += sum(f.stat().st_size for f in snapshot_path.glob("*.bin") if f.is_file())

                    return True, snapshot_path, total_size

    return False, None, 0


def get_offload_folder(args: argparse.Namespace) -> str:
    """Get the offload folder path"""
    if hasattr(args, 'offload_folder') and args.offload_folder:
        return args.offload_folder
    # Use system temp directory by default
    return tempfile.gettempdir()


def load_model(args: argparse.Namespace):
    """Load the model with specified configuration"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

    # Optionally disable GPTQModel
    if args.no_gptq:
        os.environ["NO_GPTQMODEL"] = "1"
        print("   ℹ️  GPTQModel disabled - using standard transformers loading")

    model_name = args.model

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print(f"Qwen3-Next-80B Loader v2.1")
        print("=" * 60)

    # Check cache
    is_cached, cache_path, cache_size = check_model_cache(model_name)

    # Set offline mode if model is cached and user didn't explicitly disable it
    if is_cached and args.offline is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.verbose:
            print("   📡 Auto-enabled offline mode (model is cached)")

    if not args.quiet:
        if is_cached:
            print(f"\n📂 Model cached locally: {cache_path}")
            print(f"   Size on disk: {cache_size / (1024**3):.1f}GB")
        else:
            print(f"\n📂 Model will be downloaded from Hugging Face")
            print(f"   This will take time on first run!")

    # System info
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)

    if not args.quiet:
        print(f"\n📊 System Resources:")
        print(f"   RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

    # GPU detection and quantized model handling
    device_map = args.device_map
    max_memory = None
    force_cpu_for_quantized = False

    if device_map == "auto" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if not args.quiet:
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f}GB")

        # For quantized models with GPTQModel, we need enough VRAM for the entire model
        # But if --no-gptq is used, we can do hybrid loading
        if gpu_memory < 30 and not args.no_gptq:  # Quantized model needs ~29GB with GPTQModel
            force_cpu_for_quantized = True
            if not args.quiet:
                print(f"   ⚠️  Insufficient VRAM for quantized model ({gpu_memory:.1f}GB < 30GB)")
                print(f"   ❌ GPTQModel quantization requires full GPU loading")
                print(f"   💡 Alternatives:")
                print(f"      1. Use --no-gptq flag for hybrid CPU/GPU loading")
                print(f"      2. Use a GPU with 30GB+ VRAM")
                print(f"      3. Use GGUF format models for CPU inference")
            print("\n❌ Cannot run with GPTQModel on limited GPU")
            print("   Try: ./qwen3_80b.py --no-gptq  (for hybrid CPU/GPU mode)")
            return None, None
        elif gpu_memory < 30 and args.no_gptq:
            # With GPTQModel disabled, we can try hybrid loading
            if not args.quiet:
                print(f"   ℹ️  Limited VRAM ({gpu_memory:.1f}GB) - using hybrid CPU/GPU mode")
                print(f"   ✓ GPTQModel disabled - MoE offloading enabled")

            # Set up hybrid memory allocation
            gpu_limit = args.gpu_memory or min(14, int(gpu_memory * 0.85))
            cpu_limit = args.cpu_memory or int(available_ram * 0.75)

            max_memory = {
                0: f"{gpu_limit}GiB",
                "cpu": f"{cpu_limit}GiB"
            }

            if not args.quiet:
                print(f"   Memory limits: GPU={gpu_limit}GiB, CPU={cpu_limit}GiB")

        else:
            # GPU with sufficient VRAM (30GB+)
            gpu_limit = args.gpu_memory or int(gpu_memory * 0.85)
            cpu_limit = args.cpu_memory or int(available_ram * 0.9)

            max_memory = {
                0: f"{gpu_limit}GiB",
                "cpu": f"{cpu_limit}GiB"
            }

            if not args.quiet:
                print(f"   Memory limits: GPU={gpu_limit}GiB, CPU={cpu_limit}GiB")

    if device_map == "cpu":
        if not args.quiet:
            print("   Mode: CPU-only requested")
            print("   ⚠️  WARNING: This GPTQModel quantized model has internal CUDA kernels")
            print("   ⚠️  CPU-only mode may not work correctly")
            print("   💡 For CPU inference, consider using GGUF format models instead")

    # Loading info
    if not args.quiet:
        print("\n⏳ Loading model...")
        if not args.no_warnings:
            print("   ⚠️  Expect ~20-30 minutes for first load (single-threaded limitation)")
            print("   ⚠️  You may see a '28GB buffer' warning - this is normal")
            print("   💡 Monitor with 'htop' - only 1 CPU thread will be at 100%")

    # Load tokenizer
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(cache_path) if is_cached else model_name,
        local_files_only=is_cached,
        trust_remote_code=True
    )

    if args.verbose:
        print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model
    start_time = time.time()

    # Re-enable warnings for model loading if verbose
    if args.verbose:
        warnings.filterwarnings("default", category=UserWarning)

    # Get offload folder
    offload_folder = get_offload_folder(args)
    if args.verbose and not args.no_offload and device_map != "cpu":
        print(f"   📁 Offload folder: {offload_folder}")

    # Special handling for CPU-only mode to avoid quantization issues
    load_kwargs = {
        "dtype": torch.float16 if not args.fp32 else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": is_cached,
    }

    if device_map == "cpu":
        # For CPU mode, don't use device_map or max_memory
        load_kwargs["device_map"] = None  # Let model handle device placement
    else:
        # For GPU mode with sufficient VRAM
        load_kwargs["device_map"] = device_map
        load_kwargs["max_memory"] = max_memory
        load_kwargs["offload_buffers"] = not args.no_offload
        load_kwargs["offload_folder"] = offload_folder

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(cache_path) if is_cached else model_name,
            **load_kwargs
        )

        # For CPU mode, ensure model is properly on CPU
        if device_map == "cpu":
            if args.verbose:
                print("   Ensuring model is on CPU...")
            # Don't explicitly move - let the model handle its own device placement
            # Some quantized models have internal CUDA kernels that shouldn't be moved

    except RuntimeError as e:
        if "not on GPU" in str(e) or "b_q_weight" in str(e):
            if not args.quiet:
                print(f"\n⚠️  Quantized model requires all weights on same device")
                print(f"   Retrying with CPU-only mode...")

            # Retry with CPU-only
            device_map = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                str(cache_path) if is_cached else model_name,
                dtype=torch.float16 if not args.fp32 else torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=is_cached,
            )
            # Don't explicitly move - let the model handle device placement
        else:
            raise

    load_time = time.time() - start_time

    if not args.quiet:
        print(f"\n✅ Model loaded in {load_time:.1f}s ({load_time/60:.1f} minutes)")

    # Show device map if verbose
    if args.verbose and hasattr(model, 'hf_device_map'):
        print("\n📍 Model layer distribution:")
        devices = {}
        for name, device in model.hf_device_map.items():
            devices[device] = devices.get(device, 0) + 1
        for device, count in sorted(devices.items())[:3]:
            print(f"   {device}: {count} layers")

    return model, tokenizer


def interactive_mode(model, tokenizer, args):
    """Run interactive chat mode"""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    if args.thinking:
        print("Thinking mode enabled - will show reasoning process")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Tokenize
            inputs = tokenizer(user_input, return_tensors="pt")

            # Properly detect model device and move inputs
            try:
                model_device = next(model.parameters()).device
            except:
                model_device = torch.device('cpu')

            # Move inputs to model device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate
            if not args.quiet:
                print("\nGenerating...")

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=not args.thinking)
            gen_time = time.time() - start_time

            # Handle thinking mode
            if args.thinking and "</think>" in response:
                parts = response.split("</think>")
                if len(parts) >= 2:
                    thinking = parts[0].replace(user_input, "").strip()
                    answer = parts[1].strip()

                    if thinking and thinking.startswith("<think>"):
                        print("\n💭 Thinking:")
                        print(thinking.replace("<think>", "").strip())

                    print("\n💬 Response:")
                    print(answer)
                else:
                    print("\nResponse:")
                    print(response[len(user_input):].strip())
            else:
                print("\nResponse:")
                print(response[len(user_input):].strip())

            if args.verbose:
                print(f"\n⏱️  Time: {gen_time:.2f}s ({args.max_tokens/gen_time:.1f} tokens/s)")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def benchmark_mode(model, tokenizer, args):
    """Run benchmark tests"""
    print("\n" + "=" * 60)
    print("Benchmark Mode")
    print("=" * 60)

    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about artificial intelligence.",
        "What is the meaning of life?",
        "How do neural networks work?"
    ]

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt")

        # Properly detect model device and move inputs
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = torch.device('cpu')

        # Move inputs to model device
        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_time = time.time() - start_time
        tokens_per_sec = 50 / gen_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Response: {response[len(prompt):].strip()[:100]}...")
        print(f"   Speed: {tokens_per_sec:.1f} tokens/s")

        results.append(tokens_per_sec)

    # Summary
    avg_speed = sum(results) / len(results)
    print(f"\n📊 Benchmark Results:")
    print(f"   Average speed: {avg_speed:.1f} tokens/s")
    print(f"   Min speed: {min(results):.1f} tokens/s")
    print(f"   Max speed: {max(results):.1f} tokens/s")


def test_dependencies():
    """Test all required dependencies"""
    print("Testing dependencies...\n")

    deps = {
        "torch": None,
        "transformers": None,
        "accelerate": None,
        "safetensors": None,
        "psutil": None,
    }

    for dep in deps:
        try:
            module = __import__(dep)
            deps[dep] = getattr(module, "__version__", "unknown")
            print(f"✅ {dep}: {deps[dep]}")
        except ImportError:
            print(f"❌ {dep}: not installed")
            deps[dep] = None

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("\n⚠️  CUDA not available (CPU-only mode)")
    except:
        pass

    # Check Python version
    print(f"\n🐍 Python: {sys.version}")

    # Check if all deps are installed
    if all(deps.values()):
        print("\n✅ All dependencies installed!")
        return True
    else:
        print("\n❌ Some dependencies missing. Install with:")
        print("   uv pip install -r requirements.txt")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Next-80B AutoRound - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive mode with defaults
  %(prog)s --benchmark        # Run performance benchmark
  %(prog)s --thinking         # Enable thinking/reasoning display
  %(prog)s --cpu              # CPU-only mode (no GPU)
  %(prog)s --check            # Check if model is cached
  %(prog)s --test-deps        # Test dependencies

Advanced:
  %(prog)s --gpu-memory 12 --cpu-memory 80   # Custom memory limits
  %(prog)s --temperature 0.9 --max-tokens 200 # Generation parameters
  %(prog)s --verbose          # Show detailed information
  %(prog)s --quiet            # Minimal output

Note: First load takes ~20-30 minutes due to single-threaded loading.
      Subsequent loads are faster. The '28GB buffer' warning is normal.
        """)

    # Model configuration
    parser.add_argument("--model", default="Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
                        help="Model name or path (default: Intel's AutoRound model)")
    parser.add_argument("--device-map", choices=["auto", "cpu"], default="auto",
                        help="Device mapping strategy (default: auto)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode (same as --device-map cpu)")

    # Memory configuration
    parser.add_argument("--gpu-memory", type=int, metavar="GB",
                        help="Max GPU memory in GiB (default: 85%% of available)")
    parser.add_argument("--cpu-memory", type=int, metavar="GB",
                        help="Max CPU memory in GiB (default: 90%% of available)")
    parser.add_argument("--no-offload", action="store_true",
                        help="Disable buffer offloading (not recommended)")
    parser.add_argument("--offload-folder", type=str, metavar="PATH",
                        help="Folder for offloaded buffers (default: system temp)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens to generate (default: 150)")

    # Modes
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode (default if no other mode specified)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark tests")
    parser.add_argument("--thinking", action="store_true",
                        help="Show model's thinking process (if available)")
    parser.add_argument("--check", action="store_true",
                        help="Check model cache status and exit")
    parser.add_argument("--test-deps", action="store_true",
                        help="Test dependencies and exit")

    # Output control
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--no-warnings", action="store_true",
                        help="Suppress warning messages")

    # Performance
    parser.add_argument("--threads", type=int,
                        help="Number of CPU threads (default: all)")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 precision instead of FP16")
    parser.add_argument("--no-gptq", action="store_true",
                        help="Disable GPTQModel for better CPU/hybrid compatibility")

    # Network
    parser.add_argument("--offline", action="store_true", dest="offline", default=None,
                        help="Force offline mode (no network access)")
    parser.add_argument("--online", action="store_false", dest="offline",
                        help="Force online mode (allow network access)")

    args = parser.parse_args()

    # Handle shortcuts
    if args.cpu:
        args.device_map = "cpu"

    # Test dependencies mode
    if args.test_deps:
        success = test_dependencies()
        return 0 if success else 1

    # Check cache mode
    if args.check:
        is_cached, cache_path, cache_size = check_model_cache(args.model)
        if is_cached:
            print(f"✅ Model cached at: {cache_path}")
            print(f"   Size: {cache_size / (1024**3):.1f}GB")
            shards = list(cache_path.glob("model-*.safetensors"))
            print(f"   Shards: {len(shards)}")
        else:
            print("❌ Model not cached. Will download on first run.")
        return 0

    # Setup environment
    setup_environment(args.threads, args.verbose, args.offline if args.offline is not None else False)

    # Load model
    try:
        model, tokenizer = load_model(args)
        if model is None:
            print("\n❌ Unable to load model with current configuration")
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Loading interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during model loading: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Run mode - default to interactive if no mode specified
    try:
        if args.benchmark:
            benchmark_mode(model, tokenizer, args)
        else:  # Default to interactive mode
            interactive_mode(model, tokenizer, args)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Error during {('benchmark' if args.benchmark else 'interactive')} mode: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())