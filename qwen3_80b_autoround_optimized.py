#!/usr/bin/env python3
"""
Optimized Qwen3-80B loader using Intel AutoRound's native inference capabilities
This script leverages AutoRound's optimizations for better performance
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def setup_environment(cpu_only: bool = False):
    """Configure environment for optimal performance"""
    # Disable GPTQModel for better compatibility
    os.environ["NO_GPTQMODEL"] = "1"
    os.environ["DISABLE_GPTQMODEL"] = "1"

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("🔧 CPU-only mode enabled")

    # Set optimal thread count
    cpu_count = os.cpu_count()
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = str(cpu_count)

    # PyTorch optimizations
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    # Enable Intel optimizations if available
    try:
        import intel_extension_for_pytorch as ipex
        print("✅ Intel Extension for PyTorch detected - optimizations enabled")
        return True
    except ImportError:
        print("ℹ️  Intel Extension for PyTorch not found - using standard PyTorch")
        return False

def load_model_with_autoround(model_name: str, device: str = "auto", use_ipex: bool = False):
    """Load model using AutoRound's optimized loading"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from auto_round import AutoRoundConfig

    print("\n⏳ Loading model with AutoRound optimizations...")

    # Check cache
    is_cached, cache_path, cache_size = check_model_cache(model_name)

    if is_cached:
        print(f"📂 Using cached model: {cache_path}")
        print(f"   Size: {cache_size / (1024**3):.1f}GB")
        model_path = str(cache_path)
    else:
        print(f"📥 Model will download from HuggingFace")
        model_path = model_name

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_cached
    )

    # Configure AutoRound quantization
    # The model already has quantization_config, we just ensure it's properly loaded
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": is_cached,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16,
    }

    # Device configuration
    if device == "cpu":
        print("📊 Loading on CPU (this will take 30-60 minutes)...")
        load_kwargs["device_map"] = None  # Don't use device_map for CPU
    elif device == "auto":
        print("📊 Using automatic device placement...")
        load_kwargs["device_map"] = "auto"

        # Configure memory limits for hybrid loading
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Conservative GPU usage for stability
            load_kwargs["max_memory"] = {
                0: f"{min(14, int(gpu_memory * 0.8))}GiB",
                "cpu": "100GiB"
            }
    else:
        # Specific GPU device
        load_kwargs["device_map"] = device

    # Load model
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs
    )

    # Apply Intel optimizations if available
    if use_ipex and device == "cpu":
        try:
            import intel_extension_for_pytorch as ipex
            print("🚀 Applying Intel Extension optimizations...")
            model = ipex.optimize(model, dtype=torch.float16)
        except Exception as e:
            print(f"⚠️  Could not apply IPEX optimizations: {e}")

    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s ({load_time/60:.1f} minutes)")

    return model, tokenizer

def run_inference(model, tokenizer, prompt: str, max_tokens: int = 150, temperature: float = 0.7, show_thinking: bool = False):
    """Run inference with the model"""
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to same device as model if needed
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=not show_thinking)

    # Handle thinking mode
    if show_thinking and "</think>" in response:
        parts = response.split("</think>")
        if len(parts) >= 2:
            thinking = parts[0].replace(prompt, "").strip()
            answer = parts[1].strip()
            if thinking and thinking.startswith("<think>"):
                print("\n💭 Thinking:")
                print(thinking.replace("<think>", "").strip())
                print("\n📝 Answer:")
                print(answer)
            else:
                print(f"\nResponse: {response[len(prompt):]}")
    else:
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        print(f"\nResponse: {response}")

    return response

def benchmark_inference(model, tokenizer, num_runs: int = 3):
    """Benchmark inference performance"""
    print("\n" + "="*60)
    print("Running performance benchmark...")
    print("="*60)

    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
    ]

    total_time = 0
    total_tokens = 0

    for i, prompt in enumerate(test_prompts[:num_runs], 1):
        print(f"\nBenchmark {i}/{num_runs}")
        start = time.time()

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

    print(f"\n📊 Average Performance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average speed: {total_tokens/total_time:.2f} tokens/s")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized Qwen3-80B with Intel AutoRound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s                    # Default auto device placement
  %(prog)s --cpu              # Force CPU-only mode
  %(prog)s --benchmark        # Run performance benchmark
  %(prog)s --thinking         # Show reasoning process
  %(prog)s --interactive      # Interactive chat mode

OPTIMIZATION NOTES:
  - Uses Intel AutoRound's native loading
  - Supports Intel Extension for PyTorch (IPEX) if installed
  - Mixed-bit quantization (4/8/16-bit) for optimal performance
  - Automatic backend selection based on hardware
        """
    )

    parser.add_argument("--model", default="Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
                        help="Model name or path")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--device", default="auto",
                        help="Device placement (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--thinking", action="store_true",
                        help="Show model's thinking process")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")

    args = parser.parse_args()

    # Setup environment
    use_ipex = setup_environment(cpu_only=args.cpu)

    # Determine device
    device = "cpu" if args.cpu else args.device

    print("="*60)
    print("Qwen3-80B with Intel AutoRound Optimizations")
    print("="*60)

    # Load model
    model, tokenizer = load_model_with_autoround(args.model, device, use_ipex)

    # Run benchmark if requested
    if args.benchmark:
        benchmark_inference(model, tokenizer)
        return

    # Test prompt
    if not args.interactive:
        test_prompt = "What is 2+2? Think step by step."
        run_inference(model, tokenizer, test_prompt, args.max_tokens, args.temperature, args.thinking)

    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("Interactive Mode (type 'quit' to exit)")
        print("="*60)

        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                run_inference(model, tokenizer, user_input, args.max_tokens, args.temperature, args.thinking)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()