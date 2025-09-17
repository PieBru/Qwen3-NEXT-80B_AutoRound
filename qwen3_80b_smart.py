#!/usr/bin/env python3
"""
Smart Qwen3-Next-80B Loader
Automatically selects optimal loading strategy based on available resources
"""

import os
import sys
import time
import torch
import psutil
from pathlib import Path
import argparse

# Disable GPTQModel by default
os.environ["NO_GPTQMODEL"] = "1"
os.environ["PYTHON_GIL"] = "0"

class SmartLoader:
    """Intelligent model loader that picks the best strategy"""

    # Model constants
    MODEL_SIZE_GB = 40  # INT4 quantized size
    SHARDS = 9
    SHARD_SIZE_GB = 4.5

    # Memory requirements (with safety margins)
    FULL_GPU_MIN_VRAM = 30
    HYBRID_MIN_VRAM = 14
    HYBRID_MIN_RAM = 60
    CPU_MIN_RAM = 50

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.strategy = None
        self.device_map = None
        self.max_memory = None
        self.dtype = torch.float16

    def detect_resources(self):
        """Detect available system resources"""
        resources = {
            'ram_total': psutil.virtual_memory().total / (1024**3),
            'ram_available': psutil.virtual_memory().available / (1024**3),
            'swap_total': psutil.swap_memory().total / (1024**3),
            'cpu_cores': psutil.cpu_count(),
            'gpu_available': torch.cuda.is_available(),
        }

        if resources['gpu_available']:
            resources['gpu_name'] = torch.cuda.get_device_name(0)
            resources['vram_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            resources['vram_free'] = (torch.cuda.get_device_properties(0).total_memory -
                                     torch.cuda.memory_allocated(0)) / (1024**3)
        else:
            resources['gpu_name'] = None
            resources['vram_total'] = 0
            resources['vram_free'] = 0

        return resources

    def determine_strategy(self, resources, force_cpu=False, use_gptq=False):
        """Determine optimal loading strategy based on resources"""

        vram = resources['vram_total']
        ram = resources['ram_available']

        print("\n" + "="*60)
        print("🧠 Smart Loading Strategy Selector")
        print("="*60)

        print(f"\n📊 Detected Resources:")
        print(f"   RAM: {resources['ram_total']:.1f}GB total, {ram:.1f}GB available")
        if resources['gpu_available']:
            print(f"   GPU: {resources['gpu_name']}")
            print(f"   VRAM: {vram:.1f}GB total, {resources['vram_free']:.1f}GB available")
        else:
            print("   GPU: Not available")

        # Decision logic
        if force_cpu:
            return self._setup_cpu_strategy(ram)

        if not resources['gpu_available']:
            return self._setup_cpu_strategy(ram)

        # GPU available - determine best approach
        if vram >= self.FULL_GPU_MIN_VRAM:
            if use_gptq:
                return self._setup_full_gpu_gptq_strategy(vram, ram)
            else:
                return self._setup_full_gpu_strategy(vram, ram)

        # Limited VRAM - need CPU assistance or full CPU
        if vram >= self.HYBRID_MIN_VRAM and ram >= self.HYBRID_MIN_RAM:
            # Could try hybrid, but INT4 quantized has issues
            print(f"\n⚠️  Hybrid loading not recommended for INT4 quantized model")
            print(f"   VRAM: {vram:.1f}GB (could support hybrid)")
            print(f"   RAM: {ram:.1f}GB (sufficient for offloading)")
            print(f"   Using CPU-only mode for stability")
            return self._setup_cpu_strategy(ram)

        # Insufficient resources for hybrid
        if ram >= self.CPU_MIN_RAM:
            return self._setup_cpu_strategy(ram)

        # Insufficient resources
        print(f"\n❌ Insufficient resources for this model!")
        print(f"   Need either:")
        print(f"   • {self.FULL_GPU_MIN_VRAM}GB+ VRAM for GPU loading")
        print(f"   • {self.CPU_MIN_RAM}GB+ RAM for CPU loading")
        print(f"\n💡 Suggestions:")
        print(f"   1. Free up RAM (current: {ram:.1f}GB)")
        print(f"   2. Use GGUF format with llama.cpp")
        print(f"   3. Use a smaller model")
        return None

    def _setup_full_gpu_strategy(self, vram, ram):
        """Setup for full GPU loading"""
        self.strategy = 'full_gpu'
        self.device_map = 'auto'
        self.dtype = torch.float16

        # Conservative memory allocation
        gpu_limit = min(int(vram * 0.85), vram - 2)  # Leave 2GB or 15% free
        self.max_memory = {
            0: f"{gpu_limit}GiB",
            "cpu": f"{int(ram * 0.5)}GiB"  # CPU as fallback
        }

        print(f"\n✅ Strategy: FULL GPU LOADING")
        print(f"   All {self.SHARDS} shards will load to GPU")
        print(f"   Memory allocation: {gpu_limit}GB GPU, {int(ram * 0.5)}GB CPU (fallback)")
        print(f"   Expected performance: ⚡⚡⚡⚡⚡ (50-100 tokens/sec)")
        print(f"   Load time: ~25 minutes (first load)")

        return True

    def _setup_full_gpu_gptq_strategy(self, vram, ram):
        """Setup for full GPU loading with GPTQModel"""
        self.strategy = 'full_gpu_gptq'
        self.device_map = 'auto'
        self.dtype = torch.float16

        gpu_limit = min(int(vram * 0.9), vram - 1)
        self.max_memory = {
            0: f"{gpu_limit}GiB",
            "cpu": f"{int(ram * 0.5)}GiB"
        }

        print(f"\n✅ Strategy: FULL GPU + GPTQMODEL")
        print(f"   Optimized for multi-GPU or high-performance")
        print(f"   Memory allocation: {gpu_limit}GB GPU")
        print(f"   Expected performance: ⚡⚡⚡⚡⚡ (Best possible)")

        return True

    def _setup_cpu_strategy(self, ram):
        """Setup for CPU-only loading"""
        if ram < self.CPU_MIN_RAM:
            print(f"\n⚠️  Low RAM for CPU mode ({ram:.1f}GB < {self.CPU_MIN_RAM}GB)")
            print(f"   Model will load but may swap to disk (very slow)")

        self.strategy = 'cpu_only'
        self.device_map = 'cpu'
        self.dtype = torch.float32  # FP32 for CPU stability
        self.max_memory = None

        print(f"\n✅ Strategy: CPU-ONLY LOADING")
        print(f"   All {self.SHARDS} shards will load to RAM")
        print(f"   Memory usage: ~{self.MODEL_SIZE_GB * 1.5:.0f}GB RAM")
        print(f"   Expected performance: ⚡ (0.5-5 tokens/sec)")
        print(f"   Load time: ~25-30 minutes")

        if ram >= 100:
            print(f"   ✓ Plenty of RAM - should run smoothly")
        elif ram >= 64:
            print(f"   ✓ Adequate RAM - reasonable performance")
        else:
            print(f"   ⚠️  Limited RAM - expect slow performance")

        return True

    def get_load_kwargs(self):
        """Get kwargs for model loading based on selected strategy"""
        base_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
            'torch_dtype': self.dtype,
        }

        if self.strategy in ['full_gpu', 'full_gpu_gptq']:
            base_kwargs.update({
                'device_map': self.device_map,
                'max_memory': self.max_memory,
                'offload_buffers': True,
                'offload_folder': '/tmp'
            })
        elif self.strategy == 'cpu_only':
            base_kwargs.update({
                'device_map': None,  # Let model handle placement
            })

        return base_kwargs

    def print_loading_tips(self):
        """Print tips for the selected strategy"""
        print("\n💡 Loading Tips:")

        if self.strategy == 'full_gpu':
            print("   • Close other GPU applications to free VRAM")
            print("   • Monitor with: watch -n 1 nvidia-smi")
            print("   • First token will be slow (model warming up)")

        elif self.strategy == 'cpu_only':
            print("   • Close unnecessary applications to free RAM")
            print("   • Monitor with: htop or top")
            print("   • Consider using GGUF format for better CPU performance")
            print("   • Enable swap if you have fast NVMe (sudo swapon)")

        print("\n⏳ Loading will take 20-30 minutes...")
        print("   Single-threaded shard loading (model limitation)")
        print("   Monitor CPU - one core will be at 100%")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Qwen3-Next-80B Loader - Automatic Strategy Selection"
    )

    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--use-gptq", action="store_true",
                        help="Enable GPTQModel (for multi-GPU)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show strategy without loading")

    args = parser.parse_args()

    # Initialize smart loader
    loader = SmartLoader(verbose=args.verbose)

    # Detect resources
    resources = loader.detect_resources()

    # Determine strategy
    if not loader.determine_strategy(resources, args.force_cpu, args.use_gptq):
        return 1

    # Print loading tips
    loader.print_loading_tips()

    if args.dry_run:
        print("\n✅ Dry run complete - strategy selected")
        print(f"   Would use: {loader.strategy}")
        return 0

    # Load model
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    # Check cache
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
        print(f"📂 Model will be downloaded: {model_name}")
        local_only = False

    # Load tokenizer
    print("\n⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_only
    )

    # Load model with selected strategy
    print("⏳ Loading model shards...")

    # Set environment based on strategy
    if args.use_gptq and loader.strategy == 'full_gpu_gptq':
        # Enable GPTQModel
        del os.environ["NO_GPTQMODEL"]
        print("   GPTQModel enabled")

    load_kwargs = loader.get_load_kwargs()
    load_kwargs['local_files_only'] = local_only

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs
    )

    load_time = time.time() - start_time
    print(f"\n✅ Model loaded in {load_time:.1f}s ({load_time/60:.1f} minutes)")

    # Show memory usage
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"\n📊 GPU Memory: {gpu_used:.1f}GB used")

    process = psutil.Process()
    ram_used = process.memory_info().rss / (1024**3)
    print(f"   RAM: {ram_used:.1f}GB used by process")

    # Test generation
    print("\n🧪 Testing generation...")
    test_prompt = "What is 2+2?"

    inputs = tokenizer(test_prompt, return_tensors="pt")

    # Move inputs based on strategy
    if loader.strategy in ['full_gpu', 'full_gpu_gptq'] and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")

    print("\n✅ Model ready for use!")
    print(f"   Strategy: {loader.strategy}")

    return 0


if __name__ == "__main__":
    sys.exit(main())