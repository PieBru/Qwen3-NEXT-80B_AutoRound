#!/usr/bin/env python3
"""
Performance optimization utilities for Qwen3-80B AutoRound model
Provides various optimization techniques for better inference speed
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def setup_optimal_environment():
    """Configure environment for maximum performance"""
    cpu_count = os.cpu_count()

    optimizations = {
        # Thread optimizations
        "OMP_NUM_THREADS": str(cpu_count),
        "MKL_NUM_THREADS": str(cpu_count),
        "OPENBLAS_NUM_THREADS": str(cpu_count),
        "VECLIB_MAXIMUM_THREADS": str(cpu_count),
        "NUMEXPR_NUM_THREADS": str(cpu_count),

        # PyTorch optimizations
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",

        # Tokenizer optimization
        "TOKENIZERS_PARALLELISM": "true",

        # Disable GPTQModel for compatibility
        "NO_GPTQMODEL": "1",
        "DISABLE_GPTQMODEL": "1",

        # Python GIL (for free-threaded Python)
        "PYTHON_GIL": "0",
    }

    for key, value in optimizations.items():
        os.environ[key] = value

    # PyTorch settings
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    # Enable PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print(f"✅ Environment optimized for {cpu_count} CPU threads")
    return optimizations

def check_system_info():
    """Display system information relevant to performance"""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)

    # CPU info
    import platform
    print(f"Processor: {platform.processor()}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Platform: {platform.platform()}")

    # Python info
    print(f"Python: {sys.version}")
    is_gil_free = hasattr(sys, '_is_gil_enabled')
    print(f"GIL-free build: {'Yes' if is_gil_free else 'No'}")

    # PyTorch info
    print(f"PyTorch: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {gpu_mem:.1f}GB")
    else:
        print("CUDA available: No")

    # Memory info
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total/(1024**3):.1f}GB total, {mem.available/(1024**3):.1f}GB available")

    # Check for optimizations
    print("\n" + "="*60)
    print("Available Optimizations")
    print("="*60)

    # Check Intel MKL
    try:
        torch_config = torch.__config__.show()
        if 'mkl' in torch_config.lower():
            print("✅ Intel MKL: Available")
        else:
            print("❌ Intel MKL: Not detected")
    except:
        print("⚠️  Intel MKL: Unable to detect")

    # Check Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✅ Intel Extension for PyTorch: {ipex.__version__}")
    except ImportError:
        print("❌ Intel Extension for PyTorch: Not installed (incompatible with Python 3.13t)")

    # Check AutoRound
    try:
        import auto_round
        print(f"✅ AutoRound: {auto_round.__version__}")
    except ImportError:
        print("❌ AutoRound: Not installed")

    # Check for torch.compile support
    if hasattr(torch, 'compile'):
        print("✅ torch.compile: Available")
        print("   Backends: inductor (default), cudagraphs, onnxrt, tensorrt")
    else:
        print("❌ torch.compile: Not available (requires PyTorch 2.0+)")

def benchmark_operations():
    """Benchmark basic operations to test optimization effectiveness"""
    print("\n" + "="*60)
    print("Performance Benchmarks")
    print("="*60)

    import time
    import numpy as np

    # Matrix multiplication benchmark
    sizes = [1024, 2048, 4096]

    for size in sizes:
        # CPU benchmark
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        # Warmup
        _ = torch.mm(a, b)

        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = torch.mm(a, b)
        cpu_time = (time.time() - start) / 10

        print(f"Matrix multiply ({size}x{size}): {cpu_time*1000:.2f}ms")

        # GPU benchmark if available
        if torch.cuda.is_available():
            a_gpu = a.cuda()
            b_gpu = b.cuda()

            # Warmup
            _ = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(10):
                _ = torch.mm(a_gpu, b_gpu)
                torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 10

            print(f"  GPU: {gpu_time*1000:.2f}ms (Speedup: {cpu_time/gpu_time:.1f}x)")

def optimize_model_loading():
    """Provide recommendations for optimal model loading"""
    print("\n" + "="*60)
    print("Model Loading Optimization Recommendations")
    print("="*60)

    recommendations = [
        ("Use cached model", "Model loads 10x faster from cache vs downloading"),
        ("Enable offline mode", "Set HF_HUB_OFFLINE=1 when model is cached"),
        ("Use appropriate strategy", "--load-strategy min-gpu for mixed CPU/GPU"),
        ("Allocate sufficient memory", "Keep 10-20% RAM free for overhead"),
        ("Use SSD for offloading", "Set --offload-folder to SSD path"),
        ("Disable unnecessary features", "Use --no-warnings to reduce output"),
        ("Monitor with htop", "Loading is single-threaded, expect 1 core at 100%"),
    ]

    for title, desc in recommendations:
        print(f"\n• {title}")
        print(f"  {desc}")

    print("\n" + "="*60)
    print("Optimal Commands")
    print("="*60)

    print("\nFor fastest loading (with GPU):")
    print("  python qwen3_80b.py --load-strategy min-gpu --quiet")

    print("\nFor CPU-only (reliable but slow):")
    print("  python qwen3_80b.py --load-strategy no-gpu")

    print("\nFor maximum GPU utilization:")
    print("  python qwen3_80b.py --load-strategy max-gpu --gpu-memory 20")

    print("\nFor benchmarking:")
    print("  python qwen3_80b_autoround_optimized.py --benchmark")

def main():
    parser = argparse.ArgumentParser(
        description="Performance optimization utilities for Qwen3-80B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s --check        # Check system and optimizations
  %(prog)s --benchmark    # Run performance benchmarks
  %(prog)s --optimize     # Show optimization recommendations
  %(prog)s --all          # Run all checks
        """
    )

    parser.add_argument("--check", action="store_true",
                        help="Check system information and available optimizations")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    parser.add_argument("--optimize", action="store_true",
                        help="Show optimization recommendations")
    parser.add_argument("--all", action="store_true",
                        help="Run all checks and benchmarks")

    args = parser.parse_args()

    # Default to showing all if no options specified
    if not any([args.check, args.benchmark, args.optimize, args.all]):
        args.all = True

    # Setup optimal environment
    setup_optimal_environment()

    if args.check or args.all:
        check_system_info()

    if args.benchmark or args.all:
        benchmark_operations()

    if args.optimize or args.all:
        optimize_model_loading()

    print("\n✅ Optimization check complete!")

if __name__ == "__main__":
    main()