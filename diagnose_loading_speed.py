#!/usr/bin/env python3
"""
Diagnose why model loading is slower with certain PyTorch/IPEX configurations
"""

import os
import sys
import time
import torch
import platform
import subprocess

def check_environment():
    """Check current environment configuration"""
    print("="*60)
    print("Environment Diagnostic")
    print("="*60)

    # Python version
    print(f"Python: {sys.version}")

    # PyTorch version and build info
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch build: {torch.__config__.show()[:200]}...")

    # Check if IPEX is installed and potentially interfering
    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX: {ipex.__version__} (INSTALLED)")
        print("⚠️  IPEX might be affecting loading speed even when not explicitly used!")
    except ImportError:
        print("IPEX: Not installed")

    # Check MKL
    has_mkl = 'mkl' in torch.__config__.show().lower()
    print(f"Intel MKL: {'Yes' if has_mkl else 'No'}")

    # Check OpenMP settings
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')
    mkl_threads = os.environ.get('MKL_NUM_THREADS', 'Not set')
    print(f"OMP_NUM_THREADS: {omp_threads}")
    print(f"MKL_NUM_THREADS: {mkl_threads}")

    # Check if IPEX is auto-importing and affecting things
    loaded_modules = [m for m in sys.modules.keys() if 'ipex' in m.lower() or 'intel' in m.lower()]
    if loaded_modules:
        print("\n⚠️  Intel/IPEX modules loaded:")
        for m in loaded_modules[:10]:
            print(f"  - {m}")

    print()

def test_tensor_operations():
    """Test basic tensor operations to see if IPEX is interfering"""
    print("Testing tensor operation speeds...")

    # Test 1: Simple tensor creation
    start = time.time()
    for _ in range(1000):
        x = torch.randn(1000, 1000)
    create_time = time.time() - start
    print(f"Tensor creation (1000x): {create_time:.3f}s")

    # Test 2: Tensor loading (similar to what happens in checkpoint loading)
    start = time.time()
    data = torch.randn(1000, 1000)
    for _ in range(100):
        # Simulate loading and converting operations
        x = data.clone()
        x = x.to(torch.float16)
        x = x.contiguous()
    load_time = time.time() - start
    print(f"Tensor loading ops (100x): {load_time:.3f}s")

    # Test 3: Memory allocation pattern
    start = time.time()
    tensors = []
    for i in range(100):
        tensors.append(torch.zeros(1000, 1000))
    alloc_time = time.time() - start
    print(f"Memory allocation (100x): {alloc_time:.3f}s")

    print()

def check_ipex_hooks():
    """Check if IPEX has installed any hooks that might slow loading"""
    print("Checking for IPEX hooks and modifications...")

    # Check if torch functions have been modified
    original_functions = ['torch.load', 'torch.nn.Module', 'torch.Tensor']

    for func_name in original_functions:
        try:
            parts = func_name.split('.')
            obj = torch
            for part in parts[1:]:
                obj = getattr(obj, part)

            # Check if it's been wrapped or modified
            if hasattr(obj, '__wrapped__') or 'ipex' in str(obj).lower():
                print(f"⚠️  {func_name} may have been modified by IPEX")
            else:
                print(f"✅ {func_name} appears unmodified")
        except:
            pass

    print()

def suggest_solutions():
    """Suggest solutions based on findings"""
    print("="*60)
    print("Potential Solutions")
    print("="*60)

    has_ipex = 'intel_extension_for_pytorch' in sys.modules or any('ipex' in m for m in sys.modules)

    if has_ipex:
        print("🔴 IPEX is installed and may be slowing down loading!")
        print("\nSolutions:")
        print("1. Disable IPEX auto-optimization:")
        print("   export IPEX_DISABLE_AUTO_OPTIMIZATION=1")
        print("   export DISABLE_IPEX_AUTOCAST=1")
        print()
        print("2. Use a separate environment without IPEX:")
        print("   uv venv .venv-no-ipex")
        print("   source .venv-no-ipex/bin/activate")
        print("   uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
        print("   uv pip install -r requirements-base.txt")
        print()
        print("3. Uninstall IPEX from current environment:")
        print("   uv pip uninstall intel-extension-for-pytorch")
        print()
        print("4. Force single-threaded loading (may help):")
        print("   export OMP_NUM_THREADS=1")
        print("   export MKL_NUM_THREADS=1")

    # Check PyTorch version
    if "2.5.0" in torch.__version__:
        print("\n⚠️  PyTorch 2.5.0 detected")
        print("There might be a regression in checkpoint loading speed.")
        print("Consider using PyTorch 2.5.1 instead:")
        print("   uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")

    print("\n" + "="*60)
    print("Recommendation:")
    print("="*60)
    print("For fastest model loading, use:")
    print("• PyTorch 2.5.1 (not 2.5.0)")
    print("• WITHOUT IPEX installed (IPEX is for inference, not loading)")
    print("• IPEX should only be loaded AFTER the model is loaded")
    print()

def main():
    check_environment()
    test_tensor_operations()

    try:
        import intel_extension_for_pytorch
        check_ipex_hooks()
    except ImportError:
        print("IPEX not installed - skipping hook check\n")

    suggest_solutions()

    print("Quick test command:")
    print("time python -c \"from transformers import AutoModelForCausalLM; import time; start=time.time(); print('Starting load...'); m = AutoModelForCausalLM.from_pretrained('Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound', device_map=None, torch_dtype='auto', trust_remote_code=True, low_cpu_mem_usage=True); print(f'Loaded in {time.time()-start:.1f}s')\"")

if __name__ == "__main__":
    main()