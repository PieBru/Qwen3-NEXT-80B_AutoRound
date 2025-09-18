#!/usr/bin/env python3
"""
Fast Model Caching System for Qwen3-80B
Saves the loaded model to disk in a format that can be loaded instantly
Avoids the 30-60 minute loading time after first load
"""

import os
import sys
import time
import torch
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import Optional, Tuple
import psutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

CACHE_DIR = Path.home() / ".cache/qwen3_fast_load"
CACHE_VERSION = "v1.0"  # Increment if cache format changes

def get_cache_key(model_name: str, device: str, dtype: str) -> str:
    """Generate unique cache key for model configuration"""
    key_str = f"{model_name}_{device}_{dtype}_{CACHE_VERSION}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_path(model_name: str, device: str = "cpu", dtype: str = "float16") -> Path:
    """Get cache file path for the model"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(model_name, device, dtype)
    return CACHE_DIR / f"model_{cache_key}.pt"

def save_model_to_cache(model, tokenizer, model_name: str, device: str = "cpu", dtype: str = "float16"):
    """Save loaded model to fast cache"""
    cache_path = get_cache_path(model_name, device, dtype)

    print(f"\n💾 Saving model to fast cache...")
    print(f"   Cache location: {cache_path}")
    print(f"   This will take a few minutes but only needs to be done once...")

    start_time = time.time()

    # Create cache data
    cache_data = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config.to_dict() if hasattr(model, 'config') else {},
        'tokenizer_state': tokenizer.save_pretrained(CACHE_DIR / "tokenizer", save_serialization=True),
        'cache_version': CACHE_VERSION,
        'dtype': dtype,
        'device': device,
        'model_name': model_name,
        'timestamp': time.time()
    }

    # Save with torch (much faster than pickle for large tensors)
    torch.save(cache_data, cache_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    save_time = time.time() - start_time
    file_size_gb = cache_path.stat().st_size / (1024**3)

    print(f"   ✅ Cache saved in {save_time:.1f}s")
    print(f"   📦 Cache size: {file_size_gb:.1f}GB")
    print(f"   🚀 Next load will be ~10-100x faster!")

    return cache_path

def load_model_from_cache(cache_path: Path, model_name: str):
    """Load model from fast cache"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if not cache_path.exists():
        return None, None

    print(f"\n🚀 Loading from fast cache...")
    print(f"   Cache: {cache_path}")

    start_time = time.time()

    try:
        # Load cache data
        print("   Loading cache file...")
        cache_data = torch.load(cache_path, map_location='cpu')

        # Verify cache version
        if cache_data.get('cache_version') != CACHE_VERSION:
            print(f"   ⚠️  Cache version mismatch, need to rebuild cache")
            return None, None

        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            CACHE_DIR / "tokenizer",
            local_files_only=True,
            trust_remote_code=True
        )

        # Create model with config
        print("   Creating model structure...")
        config = AutoConfig.from_dict(cache_data['model_config'])

        # We need the model class, so we still need to load it properly
        # But we can skip the slow weight loading by using our cached state dict
        from transformers import AutoModelForCausalLM

        # Create empty model
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if cache_data['dtype'] == 'float16' else torch.float32
            )

        # Load state dict
        print("   Loading model weights from cache...")
        model.load_state_dict(cache_data['model_state_dict'], strict=False)

        # Move to CPU explicitly
        model = model.to('cpu')

        load_time = time.time() - start_time
        print(f"   ✅ Model loaded from cache in {load_time:.1f}s!")
        print(f"   🎉 Saved ~30-60 minutes compared to normal loading!")

        return model, tokenizer

    except Exception as e:
        print(f"   ❌ Cache loading failed: {e}")
        print(f"   Will fall back to normal loading...")
        return None, None

def clear_cache():
    """Clear all cached models"""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"✅ Cleared cache directory: {CACHE_DIR}")
    else:
        print("ℹ️  No cache to clear")

def list_cached_models():
    """List all cached models"""
    if not CACHE_DIR.exists():
        print("ℹ️  No cached models found")
        return

    cache_files = list(CACHE_DIR.glob("model_*.pt"))
    if not cache_files:
        print("ℹ️  No cached models found")
        return

    print(f"\n📦 Cached models in {CACHE_DIR}:")
    total_size = 0
    for cache_file in cache_files:
        size_gb = cache_file.stat().st_size / (1024**3)
        total_size += size_gb

        # Try to load metadata
        try:
            data = torch.load(cache_file, map_location='cpu', weights_only=False)
            model_name = data.get('model_name', 'Unknown')
            timestamp = data.get('timestamp', 0)
            from datetime import datetime
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
            print(f"   • {cache_file.name}")
            print(f"     Model: {model_name}")
            print(f"     Size: {size_gb:.1f}GB")
            print(f"     Cached: {date_str}")
        except:
            print(f"   • {cache_file.name} ({size_gb:.1f}GB)")

    print(f"\n   Total cache size: {total_size:.1f}GB")

def main():
    parser = argparse.ArgumentParser(
        description="Fast cache management for Qwen3-80B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s --save              # Load model normally and save to cache
  %(prog)s --load              # Load from cache (instant!)
  %(prog)s --list              # List cached models
  %(prog)s --clear             # Clear cache

WORKFLOW:
  1. First time: %(prog)s --save
     (Takes 30-60 minutes, saves to cache)
  2. Next times: %(prog)s --load
     (Takes <1 minute!)
        """
    )

    parser.add_argument("--save", action="store_true",
                        help="Load model normally and save to cache")
    parser.add_argument("--load", action="store_true",
                        help="Load model from cache (fast!)")
    parser.add_argument("--list", action="store_true",
                        help="List cached models")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all cached models")
    parser.add_argument("--model", default="Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
                        help="Model name")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive mode after loading")

    args = parser.parse_args()

    if args.clear:
        clear_cache()
        return

    if args.list:
        list_cached_models()
        return

    model_name = args.model
    cache_path = get_cache_path(model_name, "cpu", "float16")

    if args.load or (not args.save and cache_path.exists()):
        # Try to load from cache
        model, tokenizer = load_model_from_cache(cache_path, model_name)

        if model is None:
            print("\n⚠️  Cache loading failed, falling back to normal loading...")
            args.save = True

    if args.save or (model is None):
        # Load model normally
        print("\n⏳ Loading model normally (this will take 30-60 minutes)...")
        print("   After this, we'll save it for instant loading next time!")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Disable IPEX during loading
        os.environ["IPEX_DISABLE_AUTO_OPTIMIZATION"] = "1"
        os.environ["DISABLE_IPEX_AUTOCAST"] = "1"
        os.environ["NO_GPTQMODEL"] = "1"
        os.environ["DISABLE_GPTQMODEL"] = "1"

        start_time = time.time()

        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        print("   Loading model (30-60 minutes)...")
        print("   Monitor with htop - 1 CPU core at 100%")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  # CPU only
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time/60:.1f} minutes")

        # Save to cache
        if args.save or not cache_path.exists():
            save_model_to_cache(model, tokenizer, model_name)

    # Apply IPEX optimization if available
    try:
        import intel_extension_for_pytorch as ipex
        print(f"\n🚀 Applying IPEX optimizations...")
        model = ipex.optimize(model, dtype=torch.float16)
        print("   ✅ IPEX applied - 2-4x inference speedup!")
    except ImportError:
        print("\n   ℹ️  IPEX not available - CPU inference without acceleration")

    print("\n" + "="*60)
    print("🎉 Model ready for inference!")
    print("="*60)

    # Interactive mode
    if args.interactive:
        print("\nEntering interactive mode (type 'quit' to exit)...")
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                # Generate response
                inputs = tokenizer(user_input, return_tensors="pt")

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

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n{response[len(user_input):]}")

                elapsed = time.time() - start
                num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                print(f"\n⚡ {num_tokens} tokens in {elapsed:.1f}s ({num_tokens/elapsed:.1f} tokens/s)")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()