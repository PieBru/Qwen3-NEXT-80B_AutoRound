#!/usr/bin/env python3
"""
Unified CLI for Qwen3-Next-80B AutoRound model
Combines all functionality from various scripts
"""

import argparse
import os
import sys

# CRITICAL: Disable GPTQModel BEFORE any imports that might trigger it
# This prevents automatic enabling for MoE models
if "--use-gptq" not in sys.argv:
    os.environ["NO_GPTQMODEL"] = "1"
    os.environ["DISABLE_GPTQMODEL"] = "1"

# Disable CUDA when using no-gpu mode to suppress warnings
if "--load-strategy" in sys.argv:
    idx = sys.argv.index("--load-strategy")
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "no-gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Additional environment variables to minimize CUDA probing
        # Note: The "No CUDA runtime" warning comes from PyTorch's C++ layer
        # and cannot be fully suppressed, but it's harmless in CPU-only mode

import time
import gc
import torch
import tempfile
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress warnings unless in verbose mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import configuration
try:
    from config import CACHE_CONFIG, MEMORY_REQUIREMENTS, MODEL_NAME as DEFAULT_MODEL_NAME
    CACHE_DIR = CACHE_CONFIG["cache_dir"]
    CACHE_VERSION = CACHE_CONFIG["cache_version"]
except ImportError:
    # Fallback if config.py not available
    CACHE_DIR = Path.home() / ".cache/qwen3_fast_loader"
    CACHE_VERSION = "v3.0"
    DEFAULT_MODEL_NAME = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"


def accelerate_shard_loading() -> None:
    """Configure environment for faster shard loading"""
    # Set environment variables to optimize loading
    os.environ["TRANSFORMERS_USE_MULTIPROCESSING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Increase number of threads for torch operations
    if "OMP_NUM_THREADS" not in os.environ:
        num_threads = os.cpu_count()
        if num_threads:
            # Use 75% of available threads for loading
            loading_threads = max(4, int(num_threads * 0.75))
            os.environ["OMP_NUM_THREADS"] = str(loading_threads)
            torch.set_num_threads(loading_threads)
            if not "--quiet" in sys.argv and "-q" not in sys.argv:
                print(f"🚀 Using {loading_threads} threads for optimized loading")


class ModelCache:
    """Fast model caching system"""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_paths(self, model_name: str, device_type: str) -> Dict[str, Path]:
        """Get cache file paths"""
        cache_key = f"{model_name.replace('/', '_')}_{device_type}"
        model_dir = self.cache_dir / cache_key
        model_dir.mkdir(exist_ok=True)
        return {
            'model': model_dir / 'model.pkl',
            'tokenizer': model_dir / 'tokenizer',
            'metadata': model_dir / 'metadata.json'
        }

    def is_cached(self, model_name: str, device_type: str) -> bool:
        """Check if model is cached"""
        paths = self.get_cache_paths(model_name, device_type)
        return paths['model'].exists() and paths['metadata'].exists()

    def save(self, model: Any, tokenizer: Any, model_name: str, device_type: str, is_ipex_optimized: bool = False) -> bool:
        """Save model to cache - handles models with hooks and IPEX optimization"""
        paths = self.get_cache_paths(model_name, device_type)
        print(f"\n💾 Creating fast-load cache...")
        if is_ipex_optimized:
            print(f"   📌 Saving IPEX-optimized model (skips repacking on load!)")
        print(f"   Location: {paths['model'].parent}")
        start_time = time.time()

        try:
            # Save metadata
            metadata = {
                'model_name': model_name,
                'device_type': device_type,
                'cache_version': CACHE_VERSION,
                'is_ipex_optimized': is_ipex_optimized,
                'timestamp': time.time(),
                'pytorch_version': torch.__version__,
            }

            # Add IPEX version if available
            if is_ipex_optimized:
                try:
                    import intel_extension_for_pytorch as ipex
                    metadata['ipex_version'] = ipex.__version__
                except:
                    metadata['ipex_version'] = None

            # Use atomic write for metadata
            import tempfile
            temp_fd, temp_path = tempfile.mkstemp(dir=paths['metadata'].parent, suffix='.tmp')
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(metadata, f, indent=2)
                # Atomic rename
                os.replace(temp_path, paths['metadata'])
            except:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            # Save tokenizer
            print("   Saving tokenizer...")
            tokenizer.save_pretrained(paths['tokenizer'])

            # For IPEX-optimized models, use torch.save instead of pickle
            if is_ipex_optimized:
                print("   Saving IPEX-optimized state (preserves repacking)...")
                # Atomic save for IPEX model
                temp_model = paths['model'].with_suffix('.tmp')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': model.config.to_dict() if hasattr(model, 'config') else {}
                }, temp_model)
                # Atomic rename
                temp_model.rename(paths['model'])

                save_time = time.time() - start_time
                size_gb = paths['model'].stat().st_size / (1024**3)
                print(f"   ✅ IPEX cache created in {save_time:.1f}s")
                print(f"   📦 Cache size: {size_gb:.1f}GB")
                print(f"   🚀 Next load will skip repacking phase!")
            else:
                # Try standard pickle for non-IPEX models
                print("   Attempting to serialize model...")
                try:
                    # Atomic save for standard model
                    temp_model = paths['model'].with_suffix('.tmp')
                    with open(temp_model, 'wb') as f:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # Atomic rename
                    temp_model.rename(paths['model'])

                    save_time = time.time() - start_time
                    size_gb = paths['model'].stat().st_size / (1024**3)
                    print(f"   ✅ Cache created in {save_time:.1f}s")
                    print(f"   📦 Cache size: {size_gb:.1f}GB")
                    print(f"   🚀 Next load will be <1 minute!")

                except (pickle.PicklingError, TypeError, AttributeError) as e:
                    # Pickle failed - model has hooks or local functions
                    print(f"   ⚠️  Standard caching not possible due to model hooks")
                    print(f"   ℹ️  Models with CPU offloading cannot be cached")
                    print(f"   💡 TIP: Use --load-strategy no-gpu for cacheable loading")

                    # Clean up partial files
                    if paths['model'].exists():
                        paths['model'].unlink()

                    # Remove metadata since we can't cache
                    if paths['metadata'].exists():
                        paths['metadata'].unlink()

                    return False

        except Exception as e:
            print(f"   ❌ Cache creation failed: {e}")
            # Clean up on any error
            import shutil
            if paths['model'].parent.exists():
                shutil.rmtree(paths['model'].parent)
            return False

        return True

    def load(self, model_name: str, device_type: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model from cache"""
        if not self.is_cached(model_name, device_type):
            return None, None

        paths = self.get_cache_paths(model_name, device_type)
        print(f"\n🚀 Loading from fast cache...")
        print(f"   Cache: {paths['model'].parent}")
        start_time = time.time()

        try:
            # Load metadata
            with open(paths['metadata'], 'r') as f:
                metadata = json.load(f)

            if metadata.get('cache_version') != CACHE_VERSION:
                print(f"   ⚠️  Cache version mismatch, rebuilding...")
                return None, None

            # Load tokenizer
            print("   Loading tokenizer...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                paths['tokenizer'],
                local_files_only=True,
                trust_remote_code=True
            )

            # Load model - check if IPEX-optimized
            print("   Loading model from cache...")
            if metadata.get('is_ipex_optimized', False):
                # Load IPEX-optimized model with torch.load
                print("   (IPEX-optimized model detected, using torch.load)")
                checkpoint = torch.load(paths['model'], map_location='cpu')
                from transformers import AutoConfig, AutoModelForCausalLM
                config = AutoConfig.from_dict(checkpoint['model_config'])

                # Recreate model structure
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )

                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Load standard model with pickle
                with open(paths['model'], 'rb') as f:
                    model = pickle.load(f)

            load_time = time.time() - start_time
            print(f"   ✅ Model loaded in {load_time:.1f}s!")
            if load_time < 60:
                speedup = 2700 / load_time  # 45 minutes avg = 2700 seconds
                print(f"   🎉 That's {speedup:.0f}x faster than normal loading!")

            return model, tokenizer

        except Exception as e:
            print(f"   ❌ Cache loading failed: {e}")
            print(f"   Will rebuild cache...")
            import shutil
            if paths['model'].parent.exists():
                shutil.rmtree(paths['model'].parent)
            return None, None

    def clear(self) -> None:
        """Clear all cached models"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"✅ Cleared cache: {self.cache_dir}")


def setup_environment(threads: Optional[int] = None, verbose: bool = False, offline: bool = False, use_gptq: bool = False, no_gpu: bool = False) -> int:
    """Configure environment for optimal performance"""
    cpu_count = threads or os.cpu_count()

    # IMPORTANT: Disable GPTQModel EARLY before any imports
    # This must happen before transformers is imported
    if not use_gptq:
        os.environ["NO_GPTQMODEL"] = "1"
        os.environ["DISABLE_GPTQMODEL"] = "1"

    # Disable IPEX auto-initialization to speed up loading
    # IPEX will be loaded AFTER model loading for inference optimization
    os.environ["IPEX_DISABLE_AUTO_OPTIMIZATION"] = "1"
    os.environ["DISABLE_IPEX_AUTOCAST"] = "1"

    # For CPU-only mode, hide GPU completely
    if no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Threading configuration
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if var not in os.environ:  # Only set if not already configured
            os.environ[var] = str(cpu_count)
        elif verbose:
            print(f"   ℹ️  {var} already set to {os.environ[var]}")

    # GPU configuration (only if not in no-gpu mode)
    if not no_gpu:
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

    # Enable multi-threaded loading optimization
    accelerate_shard_loading()

    # GPTQModel status (already set in setup_environment)
    if args.use_gptq and not args.quiet:
        print("   ℹ️  GPTQModel enabled (may conflict with hybrid strategies)")
    elif args.verbose:
        print("   ℹ️  GPTQModel disabled for compatibility with hybrid loading")

    model_name = args.model

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print(f"Qwen3-Next-80B Loader v3.4 (with Fast Caching)")
        print("https://github.com/PieBru/Qwen3-NEXT-80B_AutoRound")
        print("=" * 60)

    # Handle fast caching (enabled by default, disable with --bypass-cache)
    if not args.bypass_cache:
        cache = ModelCache()
        device_type = "cpu" if args.load_strategy == "no-gpu" else "cuda"

        # Clear cache if rebuild requested
        if args.rebuild_cache:
            if not args.quiet:
                print("🗑️  Clearing existing cache...")
            cache.clear()

        # Try loading from cache
        if not args.rebuild_cache:
            model, tokenizer = cache.load(model_name, device_type)
            if model is not None and tokenizer is not None:
                # Apply IPEX if available for CPU mode
                if device_type == "cpu" and not args.no_ipex:
                    try:
                        import intel_extension_for_pytorch as ipex
                        if not args.quiet:
                            print(f"\n🚀 Applying IPEX optimizations...")
                        model = ipex.optimize(model, dtype=torch.float16 if not args.fp32 else torch.float32)
                        if not args.quiet:
                            print("   ✅ IPEX applied - 2-4x inference speedup!")
                    except ImportError:
                        if args.verbose:
                            print("   ℹ️  IPEX not available")
                return model, tokenizer

        # If we get here, cache miss or rebuild - will load normally and save to cache
        # Mark that we need to save to cache after loading
        args._cache_needs_save = True
        args._cache_device_type = device_type  # Store for later use

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

    # System info with swap
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)
    used_ram = psutil.virtual_memory().used / (1024**3)
    swap_total = psutil.swap_memory().total / (1024**3)
    swap_used = psutil.swap_memory().used / (1024**3)
    swap_free = psutil.swap_memory().free / (1024**3)

    if not args.quiet:
        print(f"\n📊 System Resources:")
        print(f"   RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available, {used_ram:.1f}GB used")
        if swap_total > 0:
            print(f"   Swap: {swap_total:.1f}GB total, {swap_free:.1f}GB available, {swap_used:.1f}GB used")
        else:
            print(f"   Swap: Not configured")

    # Determine loading strategy BEFORE any model loading attempts
    device_map = "auto"  # Default, will be overridden based on strategy
    max_memory = None
    loading_strategy = None  # 'gpu_full', 'cpu_only', or 'hybrid'

    # Model requirements (INT4 quantized)
    MODEL_SIZE_GB = 40  # Actual model size on disk (~40GB quantized)
    NON_EXPERT_LAYERS_GB = 12  # Size of non-expert layers (~12GB)
    EXPERT_LAYERS_GB = 28  # Size of expert layers (~28GB)
    MIN_RAM_FOR_CPU = 50  # Minimum RAM for CPU-only mode
    MIN_RAM_FOR_HYBRID = 60  # Minimum RAM for hybrid mode

    # Initialize memory warning for later use
    memory_warning = None

    # Get GPU info if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
        has_gpu = True
    else:
        has_gpu = False
        gpu_memory = 0
        gpu_free = 0

    # Strategy 1: NO-GPU - Load everything on CPU
    if args.load_strategy == "no-gpu":
        loading_strategy = 'cpu_only'
        device_map = "cpu"

        # Store memory requirements info for later display
        if not args.quiet:
            # Check memory situation - updated based on actual measurements
            total_available = available_ram + swap_free
            memory_warning = None
            IPEX_PEAK_REQUIREMENT = 230  # Actual measured: model ~117GB, IPEX doubles it

            if not args.no_ipex and total_available < IPEX_PEAK_REQUIREMENT:
                memory_warning = f"Only {total_available:.1f}GB available (RAM+swap), need ~{IPEX_PEAK_REQUIREMENT}GB for IPEX"
                # Suggest swap setup
                memory_warning += " - consider 256GB swap or use --no-ipex"
            elif not args.no_ipex and swap_free > 0 and available_ram < 120:
                memory_warning = "IPEX will use swap during optimization (normal for first run)"
            elif available_ram < MIN_RAM_FOR_CPU and swap_total == 0:
                memory_warning = f"Low RAM ({available_ram:.1f}GB < {MIN_RAM_FOR_CPU}GB)"
                memory_warning += " - add swap for better performance"

    # Strategy 2: MIN-GPU (default) - Minimum non-expert layers on GPU
    elif args.load_strategy == "min-gpu" or args.load_strategy == "auto":
        if has_gpu and gpu_memory >= 10:  # Need at least 10GB for min-gpu
            loading_strategy = 'hybrid_min'
            if not args.quiet:
                if args.load_strategy == "auto":
                    print("\n🔄 Loading Strategy: AUTO (same as MIN-GPU)")
                else:
                    print("\n🔄 Loading Strategy: MIN-GPU")
                print(f"   Minimal non-expert layers on GPU (~{NON_EXPERT_LAYERS_GB}GB)")
                print(f"   All expert layers on CPU (~{EXPERT_LAYERS_GB}GB)")
                print("   Optimized for memory efficiency")

            # Conservative GPU allocation for non-expert layers only
            gpu_limit = args.gpu_memory or min(int(gpu_memory * 0.85), NON_EXPERT_LAYERS_GB + 2)  # +2GB headroom
            cpu_limit = args.cpu_memory or int(available_ram * 0.8)
            max_memory = {
                0: f"{gpu_limit}GiB",
                "cpu": f"{cpu_limit}GiB"
            }
        else:
            # Fall back to CPU if insufficient GPU
            loading_strategy = 'cpu_only'
            device_map = "cpu"
            if not args.quiet:
                print("\n📊 Loading Strategy: MIN-GPU → CPU-ONLY (fallback)")
                if not has_gpu:
                    print("   No GPU detected")
                else:
                    print(f"   Insufficient VRAM ({gpu_memory:.1f}GB < 10GB minimum)")
                print("   All layers will load on CPU/RAM")

    # Strategy 3: MAX-GPU - Fill available VRAM with as many layers as possible
    elif args.load_strategy == "max-gpu":
        if has_gpu:
            loading_strategy = 'gpu_max'

            # Calculate how much VRAM to use
            if args.gpu_memory:
                # User specified limit
                gpu_limit = min(args.gpu_memory, int(gpu_memory * 0.95))
            else:
                # Use most of available VRAM
                gpu_limit = int(gpu_memory * 0.90)

            cpu_limit = args.cpu_memory or int(available_ram * 0.8)
            max_memory = {
                0: f"{gpu_limit}GiB",
                "cpu": f"{cpu_limit}GiB"
            }

            if not args.quiet:
                print("\n🚀 Loading Strategy: MAX-GPU")
                print(f"   GPU: {gpu_name}")
                print(f"   VRAM: {gpu_memory:.1f}GB total, {gpu_free:.1f}GB available")
                print(f"   Will use up to {gpu_limit}GB of VRAM")

                if gpu_limit >= MODEL_SIZE_GB - 5:
                    print("   Most/all layers will load on GPU")
                elif gpu_limit >= NON_EXPERT_LAYERS_GB:
                    print(f"   Non-experts + some experts will load on GPU")
                else:
                    print(f"   Partial non-expert layers on GPU")
        else:
            # No GPU available, fall back to CPU
            loading_strategy = 'cpu_only'
            device_map = "cpu"
            if not args.quiet:
                print("\n📊 Loading Strategy: MAX-GPU → CPU-ONLY (no GPU)")
                print("   No GPU detected, using CPU-only mode")

    # Handle edge cases
    else:
        # This shouldn't happen with our choices, but default to min-gpu behavior
        loading_strategy = 'hybrid_min'
        if not args.quiet:
            print("\n🔄 Loading Strategy: DEFAULT (MIN-GPU)")

    # Print GPU info if available and not already shown
    if has_gpu and not args.quiet and loading_strategy != 'cpu_only':
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f}GB total, {gpu_free:.1f}GB available")

    # Final configuration based on loading strategy
    if loading_strategy == 'cpu_only':
        device_map = "cpu"
        max_memory = None  # No memory limits for CPU-only mode
        if args.use_gptq:
            print("   ⚠️  WARNING: GPTQModel may not work in CPU-only mode")
            print("   💡 Remove --use-gptq flag for better compatibility")
    elif loading_strategy in ['hybrid_min', 'gpu_max']:
        # For hybrid strategies, keep device_map as "auto" with max_memory limits
        device_map = "auto"
        # max_memory is already set in the strategy selection above
    else:
        # Fallback
        device_map = "auto"

    # Loading info - consolidated all loading messages here
    if not args.quiet:
        print(f"\n⏳ Loading model ({loading_strategy.replace('_', ' ')} mode)...")

        # Show relevant details based on strategy
        if loading_strategy == 'cpu_only':
            # More accurate constants based on testing
            print(f"   • Loading ~40GB model in shards (expect 60-90 min first load)")
            if not args.no_ipex:
                # Updated based on actual measurements: model uses ~117GB, IPEX doubles it
                ipex_peak = 230  # GB - actual measured peak
                print(f"   • IPEX optimization: needs ~{ipex_peak}GB peak (one-time)")
                total_available_mem = total_ram + swap_total
                if total_available_mem < ipex_peak:
                    print(f"   • ⚠️  Consider 256GB swap for IPEX (you have {total_available_mem:.0f}GB total)")
            print(f"   • Monitor: 'htop' shows 1 CPU at 100% (library limitation)")

            # Show any memory warnings
            if memory_warning:
                print(f"   ⚠️  {memory_warning}")
        elif loading_strategy == 'hybrid_min':
            print(f"   • Non-expert layers to GPU (~12GB)")
            print(f"   • Expert layers to CPU (~28GB)")
        elif loading_strategy == 'gpu_max':
            print(f"   • Loading as much as possible to GPU")
            print(f"   • Using up to {max_memory[0] if max_memory else 'all'} VRAM")

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

    # Explain the upcoming warnings and processes for CPU mode
    if device_map == "cpu" and not args.quiet:
        print("\n   ℹ️  Note: You may see warnings and progress bars below:")
        print("   1. 'fast path' warning - harmless, about GPU optimizations")
        print("   2. 'Loading checkpoint shards' - loading model files from disk")
        print("   3. 'repacking to CPU/XPU format' - transformers optimizing for CPU")
        print("   These are all normal and expected!\n")

    # Special handling for CPU-only mode to avoid quantization issues
    load_kwargs = {
        "dtype": torch.float16 if not args.fp32 else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": is_cached,
    }

    if device_map == "cpu":
        # For CPU mode, don't use device_map at all - this is critical for CPU-only loading
        # Using device_map=None allows the model to load properly on CPU without CUDA issues
        pass  # Don't add device_map to load_kwargs
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
        error_msg = str(e)
        if "not on GPU" in error_msg or "b_q_weight" in error_msg or "Expected all tensors to be on the same device" in error_msg:
            if not args.quiet:
                print(f"\n⚠️  Error: {error_msg[:200]}...")

            # Handle based on strategy
            if loading_strategy == 'hybrid_min' or loading_strategy == 'gpu_max':
                # For GPU strategies that failed, try with less GPU memory
                if not args.quiet:
                    print(f"   Strategy {loading_strategy} failed, trying with reduced GPU memory...")

                # Try with even more conservative GPU memory
                reduced_gpu_limit = min(8, int(gpu_memory * 0.5))  # Use only 8GB or 50% of VRAM
                cpu_limit = int(available_ram * 0.8)
                max_memory = {
                    0: f"{reduced_gpu_limit}GiB",
                    "cpu": f"{cpu_limit}GiB"
                }

                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(cache_path) if is_cached else model_name,
                        dtype=torch.float16 if not args.fp32 else torch.float32,
                        device_map="auto",
                        max_memory=max_memory,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=is_cached,
                        offload_buffers=not args.no_offload,
                        offload_folder=offload_folder
                    )
                except:
                    # Final fallback to CPU-only
                    if not args.quiet:
                        print(f"   Still failing, falling back to CPU-only mode...")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(cache_path) if is_cached else model_name,
                        dtype=torch.float16 if not args.fp32 else torch.float32,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=is_cached,
                    )
            else:
                # For other strategies, fall back to CPU-only
                if not args.quiet:
                    print(f"   Falling back to CPU-only mode...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(cache_path) if is_cached else model_name,
                    dtype=torch.float16 if not args.fp32 else torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=is_cached,
                )
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

        # RAM and swap usage
        import psutil
        process = psutil.Process()
        ram_usage_gb = process.memory_info().rss / (1024**3)
        swap_info = psutil.swap_memory()
        print(f"   RAM: {ram_usage_gb:.1f}GB used by process")
        if swap_info.total > 0:
            swap_used_gb = swap_info.used / (1024**3)
            swap_free_gb = swap_info.free / (1024**3)
            print(f"   Swap: {swap_used_gb:.1f}GB used, {swap_free_gb:.1f}GB free")

    # Show device map if verbose
    if args.verbose and hasattr(model, 'hf_device_map'):
        print("\n📍 Model layer distribution:")
        devices = {}
        for name, device in model.hf_device_map.items():
            devices[device] = devices.get(device, 0) + 1
        for device, count in sorted(devices.items())[:3]:
            print(f"   {device}: {count} layers")

    # Show cache info if we're going to create cache
    if not args.bypass_cache and hasattr(args, '_cache_needs_save') and args._cache_needs_save:
        if not args.quiet:
            if device_map == "cpu":
                print("\n📝 Will create fast-load cache after optimization")
                print("   Cache location: ~/.cache/qwen3_fast_loader/")
                print("   Cache size: ~40GB (saves IPEX-optimized model)")
                print("   Next run: <1 minute load time!")
            else:
                print("\n📝 Note: Hybrid GPU+CPU loading cannot be cached")

    # Apply IPEX optimization first (only for CPU mode)
    ipex_applied = False
    if device_map == "cpu" and not args.no_ipex:
        try:
            # Force garbage collection before IPEX to free any unnecessary memory
            if not args.quiet:
                print(f"\n🧹 Preparing for IPEX optimization...")
                mem_before_gc = psutil.Process().memory_info().rss / (1024**3)
                print(f"   Memory before cleanup: {mem_before_gc:.1f}GB")

            # Aggressive garbage collection
            gc.collect()
            gc.collect()  # Run twice for thorough cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)  # Give OS time to reclaim memory

            if not args.quiet:
                mem_after_gc = psutil.Process().memory_info().rss / (1024**3)
                if mem_after_gc < mem_before_gc:
                    print(f"   Memory after cleanup: {mem_after_gc:.1f}GB (freed {mem_before_gc - mem_after_gc:.1f}GB)")
                else:
                    print(f"   Memory after cleanup: {mem_after_gc:.1f}GB")

            # Try to import IPEX only AFTER model is loaded
            import intel_extension_for_pytorch as ipex

            if not args.quiet:
                print(f"\n🚀 Applying Intel Extension for PyTorch optimizations...")
                print(f"   IPEX version: {ipex.__version__}")

                # Show memory before IPEX
                mem_before = psutil.Process().memory_info().rss / (1024**3)
                swap_before = psutil.swap_memory().used / (1024**3)
                mem_available = psutil.virtual_memory().available / (1024**3)
                swap_free = psutil.swap_memory().free / (1024**3)

                print(f"   Current usage: {mem_before:.1f}GB RAM, {swap_before:.1f}GB swap")
                print(f"   Available: {mem_available:.1f}GB RAM, {swap_free:.1f}GB swap")

                total_available = mem_available + swap_free
                IPEX_REQUIREMENT = 115  # Additional memory needed for IPEX optimization
                if total_available < IPEX_REQUIREMENT:
                    print(f"   ⚠️  WARNING: Only {total_available:.1f}GB available for IPEX (needs ~{IPEX_REQUIREMENT}GB more)")
                    print(f"      Process may be slow (using swap) or killed (OOM)")

                print(f"   Note: Model already repacked by transformers, IPEX optimizing further...")

            # Optimize model for CPU inference
            ipex_start = time.time()
            model = ipex.optimize(model, dtype=torch.float16 if not args.fp32 else torch.float32)
            ipex_time = time.time() - ipex_start
            ipex_applied = True

            if not args.quiet:
                # Show memory after IPEX
                mem_after = psutil.Process().memory_info().rss / (1024**3)
                swap_after = psutil.swap_memory().used / (1024**3)
                print(f"   ✅ IPEX optimization complete in {ipex_time:.1f}s")
                print(f"   Memory after IPEX: {mem_after:.1f}GB RAM, {swap_after:.1f}GB swap")
                print(f"   Expecting 2-4x inference speedup!")
        except ImportError as e:
            if args.verbose:
                print("\n   ℹ️  Intel Extension for PyTorch not available")
                print("      CPU inference will work but without 2-4x speedup")
                print(f"      Install with: pip install intel-extension-for-pytorch")
            # Log the specific import error for debugging
            if args.verbose:
                import traceback
                print(f"      Debug info: {str(e)}")
        except RuntimeError as e:
            # IPEX optimization can fail with specific runtime errors
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                print(f"\n   ❌ IPEX optimization failed: Out of Memory")
                print(f"      Consider adding more swap space or using --no-ipex")
                if args.fail_on_error:
                    raise
            elif "unsupported" in error_msg:
                print(f"\n   ⚠️  IPEX optimization failed: Unsupported operation")
                print(f"      Details: {e}")
                print(f"      Continuing without IPEX optimization")
            else:
                print(f"\n   ⚠️  IPEX optimization failed: {e}")
                print(f"      Continuing without IPEX speedup")
            # Clear any partial state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            # Catch-all for unexpected errors
            if not args.quiet:
                print(f"\n   ⚠️  Unexpected error during IPEX optimization: {type(e).__name__}")
                print(f"      Error: {e}")
                if args.verbose:
                    import traceback
                    print("      Stack trace:")
                    traceback.print_exc()
                print("      Continuing without IPEX speedup")
            # Log to file for debugging
            try:
                import logging
                logging.basicConfig(filename='ipex_errors.log', level=logging.ERROR)
                logging.error(f"IPEX optimization failed: {e}", exc_info=True)
            except:
                pass

    # Save to cache AFTER IPEX optimization (if applicable)
    if not args.bypass_cache and hasattr(args, '_cache_needs_save'):
        cache = ModelCache()
        device_type = getattr(args, '_cache_device_type', "cpu" if device_map == "cpu" else "cuda")

        # Save the IPEX-optimized model if IPEX was applied
        success = cache.save(model, tokenizer, model_name, device_type, is_ipex_optimized=ipex_applied)

        if not success and not args.quiet:
            print("\n⚠️  Note: Caching is not available for hybrid CPU/GPU loading")
            print("   For cacheable loading, use: --load-strategy no-gpu")

    return model, tokenizer


def interactive_mode(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
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


def benchmark_mode(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
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


def run_performance_tests(args) -> int:
    """Run hardware performance tests"""
    results = {}

    print("\n" + "="*60)
    print("🚀 HARDWARE PERFORMANCE TESTING")
    print("="*60)

    # Import benchmark modules
    try:
        if args.perf_test or args.test_memory:
            from memory_benchmark import MemoryBenchmark
            mem_bench = MemoryBenchmark()
            print("\n[1/3] Running Memory Tests...")
            results['memory'] = mem_bench.run_comprehensive_test()

        if args.perf_test or args.test_cpu:
            from cpu_benchmark import CPUBenchmark
            cpu_bench = CPUBenchmark()
            print("\n[2/3] Running CPU Tests...")
            results['cpu'] = cpu_bench.run_comprehensive_test()

        if args.perf_test or args.test_storage:
            from storage_benchmark import StorageBenchmark
            storage_bench = StorageBenchmark()
            print("\n[3/3] Running Storage Tests...")
            results['storage'] = storage_bench.run_comprehensive_test()

    except ImportError as e:
        print(f"❌ Error importing benchmark modules: {e}")
        print("Make sure memory_benchmark.py, cpu_benchmark.py, and storage_benchmark.py are available")
        return 1

    # Generate combined report
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)

    # Memory summary
    if 'memory' in results:
        mem = results['memory']
        if mem.get('analysis', {}).get('ram_bandwidth'):
            print(f"\n💾 Memory:")
            print(f"   RAM Bandwidth: {mem['analysis']['ram_bandwidth']:.1f} GB/s")
            print(f"   System Memory: {mem['system_memory_gb']:.1f} GB")

    # CPU summary
    if 'cpu' in results:
        cpu = results['cpu']
        cpu_info = cpu.get('cpu_info', {})
        tests = cpu.get('tests', {})
        print(f"\n🖥️ CPU:")
        print(f"   Model: {cpu_info.get('model', 'Unknown')[:50]}")
        print(f"   Cores: {cpu_info.get('physical_cores', 'Unknown')} physical")
        if 'matrix_mult' in tests:
            print(f"   Performance: {tests['matrix_mult']['avg_gflops']:.1f} GFLOPS")
        if cpu_info.get('features'):
            print(f"   SIMD: {', '.join(cpu_info['features'][:3])}")

    # Storage summary
    if 'storage' in results:
        storage = results['storage']
        summary = storage.get('summary', {})
        if summary.get('best_location'):
            print(f"\n💾 Storage:")
            print(f"   Best Location: {summary['best_location']}")
            print(f"   Sequential Read: {summary.get('best_speed_mbps', 0):.1f} MB/s")

        if 'model_simulation' in storage:
            sim = storage['model_simulation']
            if 'estimated_80b_load_time' in sim:
                print(f"   Est. Model Load: {sim['estimated_80b_load_time']:.0f} seconds")

    # Model inference predictions
    print(f"\n🔮 Performance Predictions for Qwen3-80B:")

    # Based on benchmarks, estimate performance
    if 'memory' in results and 'cpu' in results:
        ram_bw = results['memory'].get('analysis', {}).get('ram_bandwidth', 20)
        cpu_gflops = results['cpu'].get('tests', {}).get('matrix_mult', {}).get('avg_gflops', 50)

        # Use config thresholds if available
        try:
            from config import BENCHMARK_CONFIG
            scoring = BENCHMARK_CONFIG.get('scoring', {})
            bw_excellent = scoring.get('ram_bandwidth_excellent', 50)
            bw_good = scoring.get('ram_bandwidth_good', 30)
            bw_moderate = scoring.get('ram_bandwidth_moderate', 20)
            cpu_excellent = scoring.get('cpu_gflops_excellent', 100)
            cpu_good = scoring.get('cpu_gflops_good', 50)
            cpu_moderate = scoring.get('cpu_gflops_moderate', 20)
        except:
            # Fallback values
            bw_excellent, bw_good, bw_moderate = 50, 30, 20
            cpu_excellent, cpu_good, cpu_moderate = 100, 50, 20

        # Simple heuristic predictions
        if ram_bw > bw_excellent and cpu_gflops > cpu_excellent:
            perf_level = "Excellent"
            tokens_estimate = "2-4"
        elif ram_bw > bw_good and cpu_gflops > cpu_good:
            perf_level = "Good"
            tokens_estimate = "1-2"
        elif ram_bw > bw_moderate and cpu_gflops > cpu_moderate:
            perf_level = "Moderate"
            tokens_estimate = "0.5-1"
        else:
            perf_level = "Limited"
            tokens_estimate = "<0.5"

        print(f"   Performance Level: {perf_level}")
        print(f"   Expected Tokens/sec: {tokens_estimate}")
        print(f"   Recommended Mode: ", end="")

        if perf_level in ["Excellent", "Good"]:
            print("min-gpu or max-gpu")
        else:
            print("no-gpu (CPU-only with caching)")

    # Save results if requested
    if args.perf_output:
        output_file = args.perf_output
        if not output_file.endswith('.json'):
            output_file += '.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Full results saved to: {output_file}")

    print("\n" + "="*60)
    print("✅ Performance testing complete!")
    print("="*60)

    return 0


def test_dependencies() -> bool:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-Next-80B AutoRound - Smart Loading with Auto Strategy Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s                         # Default: min-gpu strategy
  %(prog)s --load-strategy no-gpu  # Pure CPU mode (reliable, no GPU issues)
  %(prog)s --load-strategy min-gpu # Minimal GPU usage (default)
  %(prog)s --load-strategy max-gpu # Fill available VRAM
  %(prog)s --load-strategy auto    # Same as min-gpu
  %(prog)s --thinking              # Show model's reasoning
  %(prog)s --benchmark             # Run performance tests
  %(prog)s --check                 # Check model cache

LOADING STRATEGIES:
  min-gpu: (DEFAULT) Load non-expert layers on GPU (~12GB), experts on CPU
  no-gpu:  Pure CPU-only mode (25-30 min load, but reliable - no GPU issues)
  max-gpu: Fill VRAM with as many layers as possible
  auto:    Same as min-gpu

MEMORY USAGE BY STRATEGY:
  no-gpu:  0GB VRAM, ~50GB RAM required
  min-gpu: ~12-14GB VRAM, ~40GB RAM (recommended for RTX 4090)
  max-gpu: Uses all available VRAM, remainder in RAM

ADVANCED OPTIONS:
  %(prog)s --gpu-memory 14         # Limit GPU memory (GiB)
  %(prog)s --cpu-memory 80         # Limit CPU memory (GiB)
  %(prog)s --use-gptq              # GPTQModel (multi-GPU only)
  %(prog)s --dry-run               # Show strategy without loading

GENERATION PARAMETERS:
  %(prog)s --temperature 0.9           # Sampling temperature (0-2)
  %(prog)s --max-tokens 200            # Max tokens to generate

OUTPUT CONTROL:
  %(prog)s --verbose                   # Detailed information
  %(prog)s --quiet                     # Minimal output

v3.3 IMPROVEMENTS:
  • min-gpu is now the default (optimized for RTX 4090)
  • Clearer strategy definitions: no-gpu, min-gpu, max-gpu
  • min-gpu loads only non-expert layers on GPU (~12GB)
  • max-gpu fills available VRAM with as many layers as possible

CACHING (ENABLED BY DEFAULT!):
  First run: Creates cache automatically (30-60 min + cache creation)
  Next runs: Loads from cache in <1 minute! (30-50x faster)

  IMPORTANT: Caching only works with CPU-only mode (--load-strategy no-gpu)
  Hybrid GPU+CPU loading cannot be cached due to memory hooks

  %(prog)s --load-strategy no-gpu  # CPU-only (cacheable!)
  %(prog)s --bypass-cache          # Disable caching (always load fresh)
  %(prog)s --clear-cache           # Clear cache and exit
  %(prog)s --rebuild-cache         # Force rebuild cache

LOAD TIME:
  Without cache: 30-60 minutes (9 shards × 4.5GB each)
  With cache: <1 minute! (CPU-only mode after first run)
        """)

    # Model configuration
    parser.add_argument("--model", default="Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
                        help="Model name or path (default: Intel's AutoRound model)")
    parser.add_argument("--load-strategy", choices=["auto", "no-gpu", "min-gpu", "max-gpu"], default="min-gpu",
                        help="Loading strategy: min-gpu (default), no-gpu (CPU only), max-gpu (fill VRAM), auto (same as min-gpu)")

    # Legacy compatibility
    parser.add_argument("--device-map", choices=["auto", "cpu"], help=argparse.SUPPRESS)  # Hidden, for backward compat
    parser.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)  # Hidden, for backward compat

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

    # Performance testing
    parser.add_argument("--perf-test", action="store_true",
                        help="Run complete hardware performance tests")
    parser.add_argument("--test-memory", action="store_true",
                        help="Test memory bandwidth performance")
    parser.add_argument("--test-cpu", action="store_true",
                        help="Test CPU performance")
    parser.add_argument("--test-storage", action="store_true",
                        help="Test storage I/O performance")
    parser.add_argument("--perf-output", type=str, metavar="FILE",
                        help="Save performance results to JSON file")

    # Output control
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--no-warnings", action="store_true",
                        help="Suppress warning messages")
    parser.add_argument("--no-ipex", action="store_true",
                        help="Disable Intel Extension for PyTorch optimizations (even if installed)")

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

    # Caching options
    parser.add_argument("--bypass-cache", action="store_true", default=False,
                        help="Disable fast caching (always load from scratch)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild cache even if it exists")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear model cache and exit")
    parser.add_argument("--create-cache-only", action="store_true",
                        help="Create cache file and exit (useful for sharing with others)")
    parser.add_argument("--cache-output", type=str, metavar="PATH",
                        help="Output path for cache file (with --create-cache-only)")

    args = parser.parse_args()

    # Handle legacy options for backward compatibility
    if args.cpu:
        args.load_strategy = "no-gpu"
        if args.verbose:
            print("Note: --cpu is deprecated, use --load-strategy no-gpu")
    elif args.device_map:
        if args.device_map == "cpu":
            args.load_strategy = "no-gpu"
        else:
            args.load_strategy = "auto"
        if args.verbose:
            print(f"Note: --device-map is deprecated, use --load-strategy")

    # Handle cache operations
    if args.clear_cache:
        cache = ModelCache()
        cache.clear()
        return 0

    # Create cache only mode
    if args.create_cache_only:
        print("\n🔨 Cache Creation Mode")
        print("=" * 50)

        # Determine device type based on load strategy
        if args.load_strategy == "no-gpu":
            device_type = "cpu"
            print("📦 Creating CPU-optimized cache (most portable)")
        else:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"📦 Creating {device_type.upper()}-optimized cache")

        # Load the model
        print(f"\n⏳ Loading model for cache creation...")
        print(f"   Model: {args.model}")
        print(f"   Strategy: {args.load_strategy}")

        try:
            # Override cache settings for cache creation
            args.bypass_cache = True  # Always load fresh for cache creation
            args.rebuild_cache = False  # Don't use existing cache
            model, tokenizer = load_model(args)

            # Save to cache
            cache = ModelCache()
            is_ipex = hasattr(model, '_ipex_optimized') and model._ipex_optimized
            success = cache.save(model, tokenizer, args.model, device_type, is_ipex_optimized=is_ipex)

            if success:
                paths = cache.get_cache_paths(args.model, device_type)
                print(f"\n✅ Cache created successfully!")
                print(f"   Location: {paths['model'].parent}")
                print(f"   Size: {paths['model'].stat().st_size / (1024**3):.2f} GB")

                # If output path specified, create a portable cache archive
                if args.cache_output:
                    import shutil
                    import tarfile

                    output_path = Path(args.cache_output)
                    if output_path.suffix not in ['.tar', '.gz', '.tar.gz']:
                        output_path = output_path.with_suffix('.tar.gz')

                    print(f"\n📦 Creating portable cache archive...")
                    with tarfile.open(output_path, 'w:gz') as tar:
                        cache_dir = paths['model'].parent
                        for item in cache_dir.iterdir():
                            tar.add(item, arcname=item.name)

                    print(f"✅ Portable cache saved to: {output_path}")
                    print(f"   Size: {output_path.stat().st_size / (1024**3):.2f} GB")
                    print(f"\n💡 Others can extract this with:")
                    print(f"   tar -xzf {output_path.name} -C ~/.cache/qwen3_fast_loader/")
            else:
                print("❌ Failed to create cache")
                return 1

        except Exception as e:
            print(f"❌ Error creating cache: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        # Clean up memory
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n✅ Cache creation complete!")
        return 0

    # Test dependencies mode
    if args.test_deps:
        success = test_dependencies()
        return 0 if success else 1

    # Performance testing modes
    if args.perf_test or args.test_memory or args.test_cpu or args.test_storage:
        return run_performance_tests(args)

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

    # Setup environment (disable GPTQModel early if not requested)
    setup_environment(args.threads, args.verbose, args.offline if args.offline is not None else False, args.use_gptq, args.load_strategy == "no-gpu")

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
        else:
            print(f"   GPU: Not available")

        # Show what strategy would be used
        print(f"\n🎯 Strategy: --load-strategy {args.load_strategy}")

        if args.load_strategy == "no-gpu":
            print(f"\n📊 Would use: NO-GPU (CPU-only)")
            print(f"   All layers on CPU/RAM")
            print(f"   Memory: 0GB VRAM, ~50GB RAM")
            if torch.cuda.is_available():
                print(f"   GPU available but not used per request")

        elif args.load_strategy == "min-gpu" or args.load_strategy == "auto":
            if torch.cuda.is_available() and gpu_memory >= 10:
                print(f"\n🔄 Would use: MIN-GPU (default)")
                print(f"   Non-expert layers on GPU (~12GB)")
                print(f"   Expert layers on CPU (~28GB)")
                print(f"   Memory: ~12-14GB VRAM, ~40GB RAM")
            else:
                print(f"\n📊 Would use: MIN-GPU → CPU-ONLY (fallback)")
                if not torch.cuda.is_available():
                    print(f"   No GPU detected")
                else:
                    print(f"   Insufficient VRAM ({gpu_memory:.1f}GB < 10GB)")
                print(f"   All layers on CPU/RAM")

        elif args.load_strategy == "max-gpu":
            if torch.cuda.is_available():
                gpu_limit = args.gpu_memory or int(gpu_memory * 0.90)
                print(f"\n🚀 Would use: MAX-GPU")
                print(f"   Fill up to {gpu_limit}GB of VRAM")
                if gpu_limit >= 35:
                    print(f"   Most/all layers will load on GPU")
                elif gpu_limit >= 20:
                    print(f"   Non-experts + some experts on GPU")
                else:
                    print(f"   Non-experts on GPU, experts on CPU")
                print(f"   Memory: up to {gpu_limit}GB VRAM, remainder in RAM")
            else:
                print(f"\n📊 Would use: MAX-GPU → CPU-ONLY (no GPU)")
                print(f"   No GPU detected, using CPU-only")

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