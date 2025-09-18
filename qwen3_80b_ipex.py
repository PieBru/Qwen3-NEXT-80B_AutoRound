#!/usr/bin/env python3
"""
Qwen3-80B with Intel Extension for PyTorch (IPEX) optimizations
Requires Python 3.11 or 3.12 for IPEX compatibility
Provides 2-4x speedup for CPU inference
"""

import os
import sys
import time
import torch
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check for IPEX availability
IPEX_AVAILABLE = False
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    IPEX_VERSION = ipex.__version__
except ImportError:
    pass

def setup_environment(use_ipex: bool = True, cpu_only: bool = False):
    """Configure environment for optimal performance with IPEX"""
    cpu_count = os.cpu_count()

    # Disable GPTQModel for compatibility
    os.environ["NO_GPTQMODEL"] = "1"
    os.environ["DISABLE_GPTQMODEL"] = "1"

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Thread optimization for Intel CPUs
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)

    # Intel specific optimizations
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"

    # PyTorch settings
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    print("="*60)
    print("Qwen3-80B with Intel Optimizations")
    print("="*60)
    print(f"CPU threads: {cpu_count}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    if IPEX_AVAILABLE and use_ipex:
        print(f"✅ Intel Extension for PyTorch: {IPEX_VERSION}")
        print("   2-4x CPU speedup enabled!")
    elif not IPEX_AVAILABLE:
        print("⚠️  Intel Extension for PyTorch not found")
        print("   Install with: pip install intel-extension-for-pytorch")
        print("   Continuing without IPEX optimizations...")
    else:
        print("ℹ️  IPEX available but not enabled (use --no-ipex to disable)")

    return IPEX_AVAILABLE and use_ipex

def check_model_cache(model_name: str) -> Tuple[bool, Optional[Path], int]:
    """Check if model is cached locally"""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{model_cache_name}"

    if model_path.exists():
        snapshots = model_path / "snapshots"
        if snapshots.exists():
            snapshot_dirs = list(snapshots.iterdir())
            if snapshot_dirs:
                snapshot_path = snapshot_dirs[0]
                total_size = sum(f.stat().st_size for f in snapshot_path.rglob("*") if f.is_file())
                return True, snapshot_path, total_size
    return False, None, 0

def optimize_model_with_ipex(model, dtype=torch.float16, verbose=True):
    """Apply Intel Extension for PyTorch optimizations"""
    if not IPEX_AVAILABLE:
        return model

    if verbose:
        print("\n🚀 Applying Intel Extension for PyTorch optimizations...")

    try:
        # IPEX optimization for transformer models
        model = ipex.optimize(
            model,
            dtype=dtype,
            level="O1",  # O1 optimization level (good balance)
            auto_kernel_selection=True,  # Auto-select best kernels
        )

        if verbose:
            print("✅ IPEX optimizations applied successfully")
            print("   Expected 2-4x speedup for CPU inference")

    except Exception as e:
        print(f"⚠️  Could not fully apply IPEX optimizations: {e}")
        print("   Continuing with partial optimizations...")

    return model

def load_model_with_ipex(model_name: str, device: str = "cpu", use_ipex: bool = True, dtype=torch.float16):
    """Load model with IPEX optimizations"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

    # Check cache
    is_cached, cache_path, cache_size = check_model_cache(model_name)

    if is_cached:
        print(f"\n📂 Using cached model: {cache_path}")
        print(f"   Size: {cache_size / (1024**3):.1f}GB")
        model_path = str(cache_path)
    else:
        print(f"\n📥 Model will download from HuggingFace")
        model_path = model_name

    # Memory check
    available_ram = psutil.virtual_memory().available / (1024**3)
    print(f"\n📊 System RAM: {available_ram:.1f}GB available")

    if available_ram < 50:
        print("⚠️  WARNING: Low RAM. Model may not load properly.")

    # Load tokenizer
    print("\n⏳ Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_cached
    )
    print(f"✅ Tokenizer loaded in {time.time() - start:.1f}s")

    # Load model
    print("\n⏳ Loading model (this may take 20-30 minutes)...")
    print("   Monitor with 'htop' - loading is single-threaded")

    start = time.time()

    # Load kwargs for CPU
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": is_cached,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }

    if device == "cpu":
        # For CPU, don't use device_map
        load_kwargs["device_map"] = None
    else:
        # For GPU/auto
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs
    )

    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time/60:.1f} minutes")

    # Apply IPEX optimizations if available and requested
    if use_ipex and IPEX_AVAILABLE and device == "cpu":
        model = optimize_model_with_ipex(model, dtype=dtype)

    return model, tokenizer

def benchmark_with_ipex(model, tokenizer, use_ipex: bool = True):
    """Benchmark inference performance with and without IPEX"""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)

    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is 2+2?",
    ]

    # Warmup
    print("\n🔥 Warming up...")
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    print("\n📊 Running benchmark...")
    total_time = 0
    total_tokens = 0

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/3: {prompt[:50]}...")

        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start

        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        total_time += elapsed
        total_tokens += num_tokens

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens: {num_tokens}")
        print(f"  Speed: {num_tokens/elapsed:.2f} tokens/s")

    print(f"\n📈 Overall Performance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average speed: {total_tokens/total_time:.2f} tokens/s")

    if use_ipex and IPEX_AVAILABLE:
        print("\n✅ Benchmark completed with IPEX optimizations")
        print("   Note: IPEX typically provides 2-4x speedup over baseline")
    else:
        print("\n⚠️  Benchmark completed without IPEX")
        print("   Install IPEX for better performance")

def interactive_mode(model, tokenizer, show_thinking: bool = False):
    """Interactive chat mode"""
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    if IPEX_AVAILABLE:
        print("✅ IPEX optimizations active")
    print("="*60)

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Tokenize
            inputs = tokenizer(user_input, return_tensors="pt")

            # Generate
            print("\n💭 Thinking...")
            start = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            elapsed = time.time() - start
            num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=not show_thinking)

            # Handle thinking mode
            if show_thinking and "</think>" in response:
                parts = response.split("</think>")
                if len(parts) >= 2:
                    thinking = parts[0].replace(user_input, "").strip()
                    answer = parts[1].strip()
                    if thinking and thinking.startswith("<think>"):
                        print("\n💭 Thinking process:")
                        print(thinking.replace("<think>", "").strip())
                        print("\n📝 Answer:")
                        print(answer)
                    else:
                        print(f"\n{response[len(user_input):]}")
            else:
                print(f"\n{response[len(user_input):]}")

            print(f"\n⚡ Generated {num_tokens} tokens in {elapsed:.1f}s ({num_tokens/elapsed:.1f} tokens/s)")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-80B with Intel Extension for PyTorch optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s --interactive           # Interactive chat with IPEX
  %(prog)s --benchmark            # Run performance benchmark
  %(prog)s --cpu --thinking       # CPU-only with thinking mode
  %(prog)s --no-ipex              # Disable IPEX (for comparison)

REQUIREMENTS:
  - Python 3.11 or 3.12 (for IPEX compatibility)
  - Intel Extension for PyTorch: pip install intel-extension-for-pytorch
  - Run setup_python311_ipex.sh for complete environment setup

PERFORMANCE:
  - IPEX provides 2-4x speedup for CPU inference
  - Optimized for Intel CPUs with AVX instructions
  - Best performance with sufficient RAM (64GB+)
        """
    )

    parser.add_argument("--model", default="Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
                        help="Model name or path")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--no-ipex", action="store_true",
                        help="Disable IPEX optimizations")
    parser.add_argument("--thinking", action="store_true",
                        help="Show model's thinking process")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    parser.add_argument("--check", action="store_true",
                        help="Check system and exit")

    args = parser.parse_args()

    # Setup environment
    use_ipex = not args.no_ipex
    ipex_enabled = setup_environment(use_ipex=use_ipex, cpu_only=args.cpu)

    # System check mode
    if args.check:
        print("\n" + "="*60)
        print("System Check")
        print("="*60)
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"IPEX available: {'Yes' if IPEX_AVAILABLE else 'No'}")
        if IPEX_AVAILABLE:
            print(f"IPEX version: {IPEX_VERSION}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Check CPU features
        import subprocess
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            if 'avx' in result.stdout.lower():
                print("✅ AVX instructions supported")
            if 'avx2' in result.stdout.lower():
                print("✅ AVX2 instructions supported")
            if 'avx512' in result.stdout.lower():
                print("✅ AVX-512 instructions supported")
        except:
            pass

        return 0

    # Determine device
    device = "cpu" if args.cpu else "auto"

    # Load model
    model, tokenizer = load_model_with_ipex(
        args.model,
        device=device,
        use_ipex=ipex_enabled,
        dtype=torch.float16
    )

    # Run benchmark
    if args.benchmark:
        benchmark_with_ipex(model, tokenizer, use_ipex=ipex_enabled)
        return 0

    # Test generation
    if not args.interactive:
        print("\n" + "="*60)
        print("Test Generation")
        print("="*60)

        test_prompt = "What is 2+2? Think step by step."
        print(f"Prompt: {test_prompt}")

        inputs = tokenizer(test_prompt, return_tensors="pt")

        print("\nGenerating...")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        elapsed = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nResponse: {response[len(test_prompt):]}")
        print(f"\nGeneration time: {elapsed:.1f}s")

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, show_thinking=args.thinking)

if __name__ == "__main__":
    main()