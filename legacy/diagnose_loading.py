#!/usr/bin/env python3
"""
Diagnostic script to identify loading bottlenecks
"""

import os
import time
import psutil
import torch
from pathlib import Path
import threading
from collections import deque

# Global monitoring data
cpu_usage = deque(maxlen=100)
io_stats = deque(maxlen=100)
monitoring = True

def monitor_system():
    """Background thread to monitor CPU and I/O"""
    global monitoring
    process = psutil.Process()

    while monitoring:
        # CPU usage per core
        cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
        active_cores = sum(1 for cpu in cpu_percents if cpu > 10)

        # I/O stats
        io = process.io_counters()

        cpu_usage.append({
            'time': time.time(),
            'active_cores': active_cores,
            'max_cpu': max(cpu_percents),
            'avg_cpu': sum(cpu_percents) / len(cpu_percents)
        })

        io_stats.append({
            'time': time.time(),
            'read_mb': io.read_bytes / (1024*1024)
        })

        time.sleep(0.5)

def main():
    print("=" * 60)
    print("Model Loading Diagnostics")
    print("=" * 60)

    # System info
    print(f"\n📊 System Info:")
    print(f"   CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"   Python GIL disabled: {os.environ.get('PYTHON_GIL', '1') == '0'}")

    # Set threading environment
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count())
    os.environ["PYTHON_GIL"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()

    print("\n🔍 Starting diagnostic load...")
    print("   Monitoring CPU and I/O usage during loading")

    # Model path
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = "models--Intel--Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"
    snapshot_hash = "23ee65d2680f86eb4f6f5b1ef654682daefb8b0a"
    local_model_path = cache_dir / model_cache_name / "snapshots" / snapshot_hash

    # Clear caches
    os.system("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
    print("   ✓ Cleared OS cache for accurate measurement")

    # Test 1: Raw file reading
    print("\n📖 Test 1: Raw file reading speed")
    shard_files = sorted(local_model_path.glob("model-*.safetensors"))

    start = time.time()
    total_bytes = 0
    for shard in shard_files[:3]:  # Test first 3 shards
        with open(shard, 'rb') as f:
            data = f.read()
            total_bytes += len(data)
            print(f"   Read {shard.name}: {len(data)/(1024**3):.1f}GB")

    elapsed = time.time() - start
    speed = (total_bytes / (1024**3)) / elapsed
    print(f"   Speed: {speed:.1f}GB/s in {elapsed:.1f}s")

    # Test 2: Safetensors loading
    print("\n🔧 Test 2: Safetensors loading")
    from safetensors import safe_open

    start = time.time()
    for shard in shard_files[:1]:  # Just one shard
        with safe_open(shard, framework="pt", device="cpu") as f:
            print(f"   Loading {shard.name}...")
            print(f"   Keys in shard: {len(f.keys())}")
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.1f}s")

    # Test 3: Actual model loading with monitoring
    print("\n🤖 Test 3: Full model loading")
    print("   Watch CPU usage - should it be using multiple cores?")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Clear monitoring data
    cpu_usage.clear()
    io_stats.clear()

    start = time.time()
    initial_io = io_stats[-1]['read_mb'] if io_stats else 0

    print("   Starting transformers model loading...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            dtype=torch.float16,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "100GiB"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_buffers=True,
            offload_folder="/tmp"
        )

        elapsed = time.time() - start
        print(f"\n   ✓ Model loaded in {elapsed:.1f}s")

    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n   ⚠️  Loading interrupted after {elapsed:.1f}s")

    finally:
        # Stop monitoring
        global monitoring
        monitoring = False
        time.sleep(1)

        # Analyze results
        if cpu_usage:
            avg_active = sum(c['active_cores'] for c in cpu_usage) / len(cpu_usage)
            max_active = max(c['active_cores'] for c in cpu_usage)
            avg_cpu = sum(c['avg_cpu'] for c in cpu_usage) / len(cpu_usage)

            print("\n📊 Performance Analysis:")
            print(f"   Average active cores: {avg_active:.1f} / {psutil.cpu_count()}")
            print(f"   Peak active cores: {max_active}")
            print(f"   Average CPU usage: {avg_cpu:.1f}%")

            if io_stats:
                final_io = io_stats[-1]['read_mb'] if io_stats else 0
                data_read = final_io - initial_io
                io_speed = data_read / elapsed / 1024  # GB/s
                print(f"   Data read: {data_read/1024:.1f}GB")
                print(f"   I/O speed: {io_speed:.1f}GB/s")

            print("\n🔍 Bottleneck Analysis:")
            if avg_active < 2:
                print("   ❌ Single-threaded bottleneck detected!")
                print("   → The transformers library safetensors loader is single-threaded")
                print("   → This is a known limitation of the current implementation")
            elif io_speed < 1:
                print("   ❌ I/O bottleneck detected!")
                print("   → Disk read speed is limiting factor")
            else:
                print("   ✓ Multi-core loading detected")

        print("\n💡 Recommendations:")
        print("   1. The safetensors format loads sequentially by design")
        print("   2. Consider using vLLM or TGI for production (they optimize loading)")
        print("   3. Keep model in RAM between runs if possible")
        print("   4. SSD speed helps but won't overcome single-thread limitation")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()