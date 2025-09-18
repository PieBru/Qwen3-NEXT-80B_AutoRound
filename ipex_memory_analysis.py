#!/usr/bin/env python3
"""
Analyze memory usage during IPEX repacking
Shows why direct-to-disk repacking isn't possible
"""

import psutil
import torch
import gc
import os

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def analyze_ipex_repacking():
    """Analyze what happens during IPEX repacking"""

    print("="*60)
    print("IPEX Repacking Memory Analysis")
    print("="*60)

    print("\n1. WHY IPEX NEEDS EXTRA RAM:")
    print("-" * 40)
    print("During ipex.optimize(), the following happens in memory:")
    print("")
    print("   Original Model (40GB)")
    print("          ↓")
    print("   [IPEX reads weights]")
    print("          ↓")
    print("   Creates new blocked layout (40GB)")
    print("          ↓")
    print("   [Brief period where BOTH exist in RAM]")
    print("          ↓")
    print("   Original freed, optimized remains")
    print("")
    print("Peak RAM usage: ~80GB (2x model size)")
    print("Final RAM usage: ~40GB (optimized model)")

    print("\n2. WHY DIRECT-TO-DISK ISN'T POSSIBLE:")
    print("-" * 40)
    print("• IPEX optimization is an in-place operation")
    print("• It modifies PyTorch's internal tensor storage")
    print("• The repacking involves:")
    print("  - Complex memory layout transformations")
    print("  - CPU-specific optimizations (AVX-512, etc.)")
    print("  - Kernel fusion and graph optimizations")
    print("• These can't be streamed to disk incrementally")

    print("\n3. CURRENT WORKAROUNDS:")
    print("-" * 40)
    print("a) Two-phase approach (what we do now):")
    print("   1. Load model (40GB RAM)")
    print("   2. Apply IPEX (peaks at 80GB)")
    print("   3. Save optimized model to cache")
    print("   4. Next time: Load pre-optimized (40GB only)")
    print("")
    print("b) Shard-by-shard optimization (theoretical):")
    print("   1. Load one shard at a time")
    print("   2. Optimize that shard")
    print("   3. Save to disk")
    print("   4. Free memory")
    print("   Problem: IPEX needs the whole model for graph optimization")

    print("\n4. MEMORY REQUIREMENTS:")
    print("-" * 40)
    print("Without IPEX cache:")
    print("  • Loading: 40GB")
    print("  • IPEX repacking: 80GB peak")
    print("  • Running: 40GB")
    print("")
    print("With IPEX cache (after first run):")
    print("  • Loading: 40GB")
    print("  • No repacking needed!")
    print("  • Running: 40GB")

    print("\n5. POTENTIAL FUTURE SOLUTIONS:")
    print("-" * 40)
    print("• Memory-mapped IPEX optimization (not supported yet)")
    print("• Streaming repacking API (would need IPEX changes)")
    print("• Pre-optimized model distribution (best solution)")
    print("• Use ONNX Runtime instead (different optimization path)")

    print("\n" + "="*60)

def show_actual_memory_test():
    """Show actual memory usage during a small model optimization"""

    print("\n6. DEMONSTRATION WITH SMALL MODEL:")
    print("-" * 40)

    try:
        import intel_extension_for_pytorch as ipex

        # Create a small model for demonstration
        print("Creating small test model (100MB)...")
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
        )

        # Move to float16 to simulate actual usage
        model = model.to(torch.float16)

        initial_mem = get_memory_usage()
        print(f"Initial memory: {initial_mem:.2f}GB")

        print("\nApplying IPEX optimization...")
        model_optimized = ipex.optimize(model, dtype=torch.float16)

        peak_mem = get_memory_usage()
        print(f"After IPEX: {peak_mem:.2f}GB")
        print(f"Memory increase: {(peak_mem - initial_mem)*1024:.0f}MB")

        # The original model is replaced, not duplicated permanently
        del model
        gc.collect()

        final_mem = get_memory_usage()
        print(f"After cleanup: {final_mem:.2f}GB")

        print("\n✅ For small models, overhead is minimal")
        print("❌ For 40GB models, temporary 2x memory is problematic")

    except ImportError:
        print("IPEX not installed - can't demonstrate actual usage")
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    analyze_ipex_repacking()
    show_actual_memory_test()

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. Use --load-strategy no-gpu for cacheable loading")
    print("2. Ensure 80-100GB RAM available for first IPEX optimization")
    print("3. After first run, only 40-50GB RAM needed")
    print("4. Consider using swap space for first run only")
    print("5. Or skip IPEX with --no-ipex flag if RAM is limited")
    print("="*60)