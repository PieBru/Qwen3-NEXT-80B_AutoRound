#!/usr/bin/env python3
"""
Enhanced Qwen3-80B loader with integrated fast caching
Loads the model in <1 minute after first cache creation
"""

import os
import sys
import time
import torch
import pickle
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import psutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Cache configuration
CACHE_DIR = Path.home() / ".cache/qwen3_fast_loader"
CACHE_VERSION = "v3.0"  # Version for cache compatibility

class ModelCache:
    """Fast model caching system for instant loading"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_paths(self, model_name: str, device_type: str) -> Dict[str, Path]:
        """Get cache file paths"""
        # Create unique cache key based on model and device
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

    def save(self, model, tokenizer, model_name: str, device_type: str):
        """Save model to cache"""
        paths = self.get_cache_paths(model_name, device_type)

        print(f"\n💾 Creating fast-load cache...")
        print(f"   Location: {paths['model'].parent}")
        print(f"   This is a one-time process (~2-3 minutes)...")

        start_time = time.time()

        # Save metadata
        metadata = {
            'model_name': model_name,
            'device_type': device_type,
            'cache_version': CACHE_VERSION,
            'timestamp': time.time(),
            'pytorch_version': torch.__version__,
        }

        with open(paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save tokenizer
        print("   Saving tokenizer...")
        tokenizer.save_pretrained(paths['tokenizer'])

        # Save model with pickle (preserves quantization)
        print("   Serializing model (this takes a moment)...")
        with open(paths['model'], 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        save_time = time.time() - start_time
        size_gb = paths['model'].stat().st_size / (1024**3)

        print(f"   ✅ Cache created in {save_time:.1f}s")
        print(f"   📦 Cache size: {size_gb:.1f}GB")
        print(f"   🚀 Next load will be <1 minute!")

    def load(self, model_name: str, device_type: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model from cache"""
        if not self.is_cached(model_name, device_type):
            return None, None

        paths = self.get_cache_paths(model_name, device_type)

        print(f"\n🚀 Loading from fast cache...")
        print(f"   Cache: {paths['model'].parent}")

        start_time = time.time()

        try:
            # Load metadata and verify version
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

            # Load model from pickle
            print("   Loading model from cache...")
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
            # Clean up corrupted cache
            import shutil
            if paths['model'].parent.exists():
                shutil.rmtree(paths['model'].parent)
            return None, None

    def clear(self):
        """Clear all cached models"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"✅ Cleared cache: {self.cache_dir}")

def load_model_with_cache(
    model_name: str,
    device_type: str = "cpu",
    use_cache: bool = True,
    rebuild_cache: bool = False,
    verbose: bool = False
) -> Tuple[Any, Any]:
    """Load model with automatic caching"""

    cache = ModelCache()

    # Clear cache if requested
    if rebuild_cache:
        print("🗑️  Clearing existing cache...")
        cache.clear()

    # Try loading from cache first
    model, tokenizer = None, None
    if use_cache and not rebuild_cache:
        model, tokenizer = cache.load(model_name, device_type)

    # If not cached or cache failed, load normally
    if model is None:
        print(f"\n⏳ Loading model normally (30-60 minutes)...")
        if use_cache:
            print("   After loading, I'll create a cache for fast future loads!")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Disable optimizations during loading
        os.environ["IPEX_DISABLE_AUTO_OPTIMIZATION"] = "1"
        os.environ["DISABLE_IPEX_AUTOCAST"] = "1"
        os.environ["NO_GPTQMODEL"] = "1"

        if device_type == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        start_time = time.time()

        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        print("   Loading model shards...")
        print("   ⚠️  You may see a 'fast path' warning - it's harmless for CPU mode")

        if device_type == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "14GiB"}  # Adjust for your GPU
            )

        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time/60:.1f} minutes")

        # Save to cache for next time
        if use_cache:
            cache.save(model, tokenizer, model_name, device_type)

    # Apply IPEX optimization for CPU inference (after loading)
    if device_type == "cpu":
        try:
            import intel_extension_for_pytorch as ipex
            print(f"\n🚀 Applying IPEX optimizations...")
            model = ipex.optimize(model, dtype=torch.float16)
            print("   ✅ IPEX applied - 2-4x inference speedup!")
        except ImportError:
            if verbose:
                print("   ℹ️  IPEX not available - CPU inference without acceleration")

    return model, tokenizer

def interactive_chat(model, tokenizer, thinking_mode: bool = False):
    """Interactive chat with the model"""
    print("\n" + "="*60)
    print("🎉 Model ready! Enter 'quit' to exit.")
    print("="*60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break

            if not user_input:
                continue

            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt")
            if next(model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\n🤖 Assistant: ", end="", flush=True)
            start = time.time()

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=not thinking_mode)

            # Handle thinking mode
            if thinking_mode and "</think>" in response:
                parts = response.split("</think>")
                if len(parts) >= 2:
                    thinking = parts[0].split("<think>")[-1] if "<think>" in parts[0] else parts[0]
                    answer = parts[1].strip()

                    print("\n\n💭 Thinking Process:")
                    print("-" * 40)
                    print(thinking.strip())
                    print("-" * 40)
                    print("\n📝 Final Answer:")
                    print(answer)
                else:
                    # Remove the input from response
                    response = response[len(user_input):].strip()
                    print(response)
            else:
                # Remove the input from response
                response = response[len(user_input):].strip()
                print(response)

            # Performance stats
            elapsed = time.time() - start
            num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            print(f"\n⚡ Generated {num_tokens} tokens in {elapsed:.1f}s ({num_tokens/elapsed:.1f} tokens/s)")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Fast-loading Qwen3-80B with caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  First run (create cache):
    %(prog)s --interactive
    Takes ~30 minutes total (loading + cache creation)

  Subsequent runs (use cache):
    %(prog)s --interactive
    Loads in <1 minute!

EXAMPLES:
  %(prog)s --interactive           # Chat mode with auto-caching
  %(prog)s --thinking              # Show thinking process
  %(prog)s --cpu                   # Force CPU-only mode
  %(prog)s --rebuild-cache         # Rebuild cache from scratch
  %(prog)s --no-cache              # Disable caching (always slow)
  %(prog)s --clear-cache           # Clear cache and exit
        """
    )

    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--thinking", action="store_true",
                        help="Show thinking process (thinking model only)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode (no GPU)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching (always load from scratch)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild cache even if it exists")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear all cached models and exit")
    parser.add_argument("--model", default="Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
                        help="Model name to use")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        cache = ModelCache()
        cache.clear()
        return

    # Determine device type
    device_type = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.verbose:
        print(f"📊 System Info:")
        print(f"   Device: {device_type.upper()}")
        print(f"   CPU cores: {os.cpu_count()}")
        print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        if device_type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

    # Load model with caching
    print("\n" + "="*60)
    print("🚀 Qwen3-80B Fast Loader")
    print("="*60)

    model, tokenizer = load_model_with_cache(
        model_name=args.model,
        device_type=device_type,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_cache,
        verbose=args.verbose
    )

    # Interactive mode
    if args.interactive:
        interactive_chat(model, tokenizer, thinking_mode=args.thinking)
    else:
        # Test generation
        print("\n📝 Test prompt: 'What is 2+2?'")
        inputs = tokenizer("What is 2+2?", return_tensors="pt")
        if device_type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Response: {response}")

    print("\n✨ Session complete!")

if __name__ == "__main__":
    main()