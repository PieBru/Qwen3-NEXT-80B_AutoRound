#!/usr/bin/env python3
"""
Hardware benchmark script for Qwen3-80B
Gathers performance data for README comparison table
"""

import os
import sys
import time
import torch
import psutil
import json
from datetime import datetime
from pathlib import Path

# Disable CUDA warnings for CPU mode
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_system_info():
    """Get detailed system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "model": "Unknown",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "speed": "Unknown",
            "type": "Unknown",
        },
        "swap": {
            "total_gb": psutil.swap_memory().total / (1024**3),
            "used_gb": psutil.swap_memory().used / (1024**3),
        },
        "gpu": None,
        "python": sys.version.split()[0],
    }

    # Try to get detailed CPU model name
    try:
        import platform
        import subprocess

        # Try lscpu for Linux
        if sys.platform == "linux":
            try:
                result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=2)
                for line in result.stdout.split('\n'):
                    if 'Model name:' in line:
                        info["cpu"]["model"] = line.split(':', 1)[1].strip()
                        break
            except:
                pass
        # Try wmic for Windows
        elif sys.platform == "win32":
            try:
                result = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True, timeout=2)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info["cpu"]["model"] = lines[1].strip()
            except:
                pass

        # Fallback to platform
        if info["cpu"]["model"] == "Unknown":
            info["cpu"]["model"] = platform.processor() or "Unknown"
    except:
        pass

    # Try to get RAM speed and type (Linux only with dmidecode)
    if sys.platform == "linux":
        # Try to get detailed memory info without sudo first
        try:
            # Try lshw without sudo (may have limited info)
            result = subprocess.run(['lshw', '-short', '-C', 'memory'],
                                 stdout=subprocess.PIPE, text=True, timeout=2,
                                 stderr=subprocess.DEVNULL)
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.split('\n'):
                    if 'DDR' in line:
                        # Extract memory type if available
                        if 'DDR5' in line:
                            info["ram"]["type"] = "DDR5"
                        elif 'DDR4' in line:
                            info["ram"]["type"] = "DDR4"
                        elif 'DDR3' in line:
                            info["ram"]["type"] = "DDR3"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Alternative: Try reading from /proc/meminfo for basic info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        # Already captured above, but can verify
                        pass
                    elif line.startswith('HugePages_Total:'):
                        # Can indicate high-performance memory config
                        huge_pages = int(line.split()[1])
                        if huge_pages > 0:
                            info["ram"]["huge_pages"] = huge_pages
        except:
            pass

        # Note: dmidecode removed for security - requires sudo
        # Users can run 'sudo dmidecode --type 17' manually if needed

    # Get GPU info if available
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "driver": torch.version.cuda,
        }

    # Check for IPEX
    try:
        import intel_extension_for_pytorch as ipex
        info["ipex"] = {
            "version": ipex.__version__,
            "available": True,
        }
    except:
        info["ipex"] = {"available": False}

    return info

def run_benchmark(model, tokenizer, num_runs=5):
    """Run performance benchmark"""

    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the benefits of exercise?",
        "How do neural networks learn?",
    ]

    results = []

    print("\n" + "="*60)
    print("Running Performance Benchmark")
    print("="*60)

    for i, prompt in enumerate(test_prompts[:num_runs], 1):
        print(f"\nTest {i}/{num_runs}: '{prompt[:50]}...'")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to model device
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = torch.device('cpu')

        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Warmup for first run
        if i == 1:
            print("   Warming up...")
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )

        # Actual benchmark
        print("   Generating...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - start_time

        # Calculate metrics
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_sec = output_tokens / gen_time

        result = {
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_time": gen_time,
            "tokens_per_second": tokens_per_sec,
        }
        results.append(result)

        print(f"   Speed: {tokens_per_sec:.2f} tokens/sec")
        print(f"   Generated {output_tokens} tokens in {gen_time:.2f}s")

    return results

def save_benchmark_results(system_info, benchmark_results, load_time=None):
    """Save benchmark results to JSON"""

    # Calculate aggregate stats
    speeds = [r["tokens_per_second"] for r in benchmark_results]

    report = {
        "system": system_info,
        "benchmark": {
            "results": benchmark_results,
            "summary": {
                "avg_tokens_per_sec": sum(speeds) / len(speeds),
                "min_tokens_per_sec": min(speeds),
                "max_tokens_per_sec": max(speeds),
                "load_time_seconds": load_time,
            }
        }
    }

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Hardware: {system_info.get('gpu', {}).get('name', 'CPU Only')}")
    print(f"VRAM/RAM: {system_info.get('gpu', {}).get('vram_gb', system_info['ram']['total_gb']):.1f}GB")
    print(f"IPEX: {'Yes' if system_info['ipex']['available'] else 'No'}")
    print(f"Average Speed: {report['benchmark']['summary']['avg_tokens_per_sec']:.2f} tokens/sec")
    print(f"Range: {report['benchmark']['summary']['min_tokens_per_sec']:.2f} - {report['benchmark']['summary']['max_tokens_per_sec']:.2f} tokens/sec")
    if load_time:
        print(f"Load Time: {load_time/60:.1f} minutes")
    print(f"\nResults saved to: {filename}")

    # Print README-ready format
    print("\n" + "="*60)
    print("README Table Entry")
    print("="*60)

    # Format GPU info
    gpu_name = "CPU Only"
    vram = "-"
    if system_info.get('gpu'):
        gpu_name = system_info['gpu']['name']
        # Detect if it's a laptop GPU
        if 'laptop' in gpu_name.lower() or 'mobile' in gpu_name.lower():
            gpu_name += " Laptop"
        elif 'rtx 4090' in gpu_name.lower() and system_info['gpu']['vram_gb'] < 20:
            gpu_name = "RTX 4090 Laptop"  # 16GB version is laptop
        vram = f"{system_info['gpu']['vram_gb']:.0f}GB"

    # Format CPU info
    cpu_model = system_info['cpu']['model']
    # Shorten common CPU names for table
    cpu_short = cpu_model.replace('Intel(R) Core(TM) ', '').replace('AMD Ryzen ', 'R')

    # Format RAM info
    ram_size = f"{system_info['ram']['total_gb']:.0f}GB"
    ram_speed = system_info['ram'].get('speed', 'Unknown')
    if ram_speed != 'Unknown' and 'MHz' not in ram_speed:
        ram_speed += 'MHz'

    mode = "GPU" if system_info.get('gpu') and system_info.get('gpu', {}).get('vram_gb', 0) >= 30 else "CPU"

    ipex_speed = f"{report['benchmark']['summary']['avg_tokens_per_sec']:.1f}"
    no_ipex_speed = "TBD"  # Would need separate benchmark run without IPEX
    load_time_str = f"{load_time/60:.0f} min" if load_time else "N/A"

    # Print formatted table row
    print(f"| {gpu_name} | {vram} | {cpu_short} | {ram_size} | {ram_speed} | {mode} | {ipex_speed} | {no_ipex_speed} | {load_time_str} |")

    # Also print detailed info for reference
    print("\nDetailed System Info:")
    print(f"  GPU: {system_info.get('gpu', {}).get('name', 'None')}")
    print(f"  CPU: {cpu_model}")
    print(f"  RAM: {ram_size} {system_info['ram'].get('type', '')} @ {ram_speed}")
    print(f"  Python: {system_info['python']}")
    print(f"  IPEX: {'Yes' if system_info['ipex']['available'] else 'No'}")

    return report

def main():
    """Main benchmark function"""

    print("="*60)
    print("Qwen3-80B Hardware Benchmark")
    print("="*60)

    # Get system info
    system_info = get_system_info()

    print("\nSystem Information:")
    print(f"  CPU: {system_info['cpu']['model']}")
    print(f"  Cores: {system_info['cpu']['cores']} physical, {system_info['cpu']['threads']} logical")
    print(f"  RAM: {system_info['ram']['total_gb']:.1f}GB total, {system_info['ram']['available_gb']:.1f}GB available")

    if system_info['gpu']:
        print(f"  GPU: {system_info['gpu']['name']}")
        print(f"  VRAM: {system_info['gpu']['vram_gb']:.1f}GB")
    else:
        print("  GPU: Not available")

    print(f"  IPEX: {'Available' if system_info['ipex']['available'] else 'Not available'}")
    print(f"  Python: {system_info['python']}")

    # Check if model is already loaded (called from main script)
    if 'model' in globals() and 'tokenizer' in globals():
        print("\n✅ Using pre-loaded model")
        benchmark_results = run_benchmark(globals()['model'], globals()['tokenizer'])
        save_benchmark_results(system_info, benchmark_results)
    else:
        # Load model
        print("\n⏳ Loading model for benchmark...")
        print("   This will take 30-60 minutes on first run")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

        start_time = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu" if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 30*(1024**3) else "auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time/60:.1f} minutes")

            # Run benchmark
            benchmark_results = run_benchmark(model, tokenizer)
            save_benchmark_results(system_info, benchmark_results, load_time)

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("Try running from the main script: python qwen3_80b.py --benchmark")
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())