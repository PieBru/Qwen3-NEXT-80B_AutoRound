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

    # GPTQModel is disabled by default for better compatibility
    # Only enable if explicitly requested for multi-GPU systems
    if args.use_gptq:
        if not args.quiet:
            print("   ℹ️  GPTQModel enabled for multi-GPU/CPU optimization")
    else:
        os.environ["NO_GPTQMODEL"] = "1"
        if args.verbose:
            print("   ℹ️  Using standard transformers loading (GPTQModel disabled)")

    model_name = args.model

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print(f"Qwen3-Next-80B Loader v3.0")
        print("https://github.com/PieBru/Qwen3-NEXT-80B_AutoRound")
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

    # Determine loading strategy BEFORE any model loading attempts
    device_map = args.device_map
    max_memory = None
    loading_strategy = None  # 'gpu_full', 'cpu_only', or 'hybrid'

    # Model requirements (INT4 quantized)
    MODEL_SIZE_GB = 40  # Actual model size on disk
    MIN_GPU_VRAM_FOR_FULL = 30  # Minimum VRAM needed for full GPU loading
    MIN_GPU_VRAM_FOR_HYBRID = 14  # Minimum VRAM for hybrid (non-experts only)
    MIN_RAM_FOR_CPU = 50  # Minimum RAM for CPU-only mode

    if device_map == "auto" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        gpu_used = torch.cuda.memory_allocated(0) / (1024**3)

        if not args.quiet:
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f}GB total, {gpu_free:.1f}GB available, {gpu_used:.1f}GB used")

        # DECISION LOGIC - Determine strategy based on available resources
        if gpu_memory >= MIN_GPU_VRAM_FOR_FULL:
            # Strategy 1: Full GPU loading (best performance)
            loading_strategy = 'gpu_full'
            if not args.quiet:
                print(f"\n✅ Loading Strategy: FULL GPU")
                print(f"   Sufficient VRAM ({gpu_memory:.1f}GB >= {MIN_GPU_VRAM_FOR_FULL}GB)")
                print(f"   All model layers will load on GPU")

            gpu_limit = args.gpu_memory or int(gpu_memory * 0.85)
            cpu_limit = args.cpu_memory or int(available_ram * 0.9)

            max_memory = {
                0: f"{gpu_limit}GiB",
                "cpu": f"{cpu_limit}GiB"
            }

        elif args.use_gptq:
            # GPTQModel requires full GPU, but we don't have enough VRAM
            if not args.quiet:
                print(f"\n❌ Cannot use GPTQModel with {gpu_memory:.1f}GB VRAM")
                print(f"   GPTQModel requires {MIN_GPU_VRAM_FOR_FULL}GB+ for full GPU loading")
                print(f"   Remove --use-gptq flag to enable CPU-only mode")
            return None, None

        else:
            # Strategy 2: CPU-only (for INT4 quantized models with limited VRAM)
            # We use CPU-only because INT4 quantized models have issues with mixed placement
            loading_strategy = 'cpu_only'
            device_map = "cpu"

            if not args.quiet:
                print(f"\n⚠️  Loading Strategy: CPU-ONLY")
                print(f"   Limited VRAM ({gpu_memory:.1f}GB < {MIN_GPU_VRAM_FOR_FULL}GB)")
                print(f"   INT4 quantized weights don't support mixed CPU/GPU well")
                print(f"   Using CPU-only to avoid loading failures")

            if available_ram < MIN_RAM_FOR_CPU:
                print(f"\n⚠️  WARNING: Low RAM ({available_ram:.1f}GB < {MIN_RAM_FOR_CPU}GB)")
                print(f"   Model may run slowly or fail to load")

    elif device_map == "auto" and not torch.cuda.is_available():
        # No GPU available
        loading_strategy = 'cpu_only'
        device_map = "cpu"
        if not args.quiet:
            print("\n📊 Loading Strategy: CPU-ONLY (no GPU detected)")

    if device_map == "cpu":
        loading_strategy = 'cpu_only'
        if not args.quiet and loading_strategy != 'cpu_only':  # Don't repeat if already set
            print("\n📊 Loading Strategy: CPU-ONLY (requested)")

        if args.use_gptq:
            print("   ⚠️  WARNING: GPTQModel may not work in CPU-only mode")
            print("   💡 Remove --use-gptq flag for better compatibility")

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

        # Show memory usage after loading
        if torch.cuda.is_available():
            gpu_used_after = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_free_after = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
            print(f"\n📊 Memory Usage After Loading:")
            print(f"   VRAM: {gpu_used_after:.1f}GB allocated, {gpu_reserved:.1f}GB reserved, {gpu_free_after:.1f}GB free")

        # RAM usage
        import psutil
        process = psutil.Process()
        ram_usage_gb = process.memory_info().rss / (1024**3)
        print(f"   RAM: {ram_usage_gb:.1f}GB used by process")

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

    # Show current memory status
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n📊 Current VRAM: {gpu_used:.1f}GB / {gpu_total:.1f}GB")

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
        description="Qwen3-Next-80B AutoRound - Smart Loading with Auto Strategy Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s                    # Auto-detect best strategy and run
  %(prog)s --thinking         # Show model's reasoning process
  %(prog)s --benchmark        # Run performance tests
  %(prog)s --check            # Check model cache status

LOADING STRATEGIES (auto-detected):
  <30GB VRAM:  CPU-only mode (avoids failed GPU attempts)
  ≥30GB VRAM:  Full GPU mode (best performance)

ADVANCED OPTIONS:
  %(prog)s --cpu                      # Force CPU-only mode
  %(prog)s --gpu-memory 14             # Limit GPU memory (GiB)
  %(prog)s --cpu-memory 80             # Limit CPU memory (GiB)
  %(prog)s --use-gptq                 # Enable GPTQModel (multi-GPU only)
  %(prog)s --dry-run                  # Show strategy without loading

GENERATION PARAMETERS:
  %(prog)s --temperature 0.9           # Sampling temperature (0-2)
  %(prog)s --max-tokens 200            # Max tokens to generate

OUTPUT CONTROL:
  %(prog)s --verbose                   # Detailed information
  %(prog)s --quiet                     # Minimal output

v3.0 IMPROVEMENTS:
  • Smart strategy selection saves 25-30 minutes
  • No more double shard loading for <30GB VRAM
  • Automatic resource detection and optimization
  • See LOADING_STRATEGIES.md for resource matrix

LOAD TIME: ~25-30 minutes (9 shards × 4.5GB each, sequential loading)
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Show loading strategy without actually loading model")

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
    parser.add_argument("--use-gptq", action="store_true",
                        help="Enable GPTQModel optimization (for multi-GPU/CPU systems)")

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

    # Dry run mode - just show strategy and exit
    if args.dry_run:
        import psutil
        print("\n🔍 Dry Run Mode - Analyzing Strategy")
        print("=" * 60)

        # Show resources
        total_ram = psutil.virtual_memory().total / (1024**3)
        available_ram = psutil.virtual_memory().available / (1024**3)
        print(f"📊 System Resources:")
        print(f"   RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f}GB")

            if gpu_memory >= 30:
                print(f"\n✅ Would use: FULL GPU LOADING")
                print(f"   All shards on GPU, best performance")
            else:
                print(f"\n⚠️  Would use: CPU-ONLY LOADING")
                print(f"   Limited VRAM ({gpu_memory:.1f}GB < 30GB)")
                print(f"   INT4 model doesn't support mixed placement")
        else:
            print(f"   GPU: Not available")
            print(f"\n📊 Would use: CPU-ONLY LOADING")

        print("\nDry run complete. Use without --dry-run to actually load.")
        return 0

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