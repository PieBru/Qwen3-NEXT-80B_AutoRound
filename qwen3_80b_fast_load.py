#!/usr/bin/env python3
"""
Fast loading version using model serialization
Saves loaded model to disk after first load for faster subsequent loads
"""

import os
import sys
import time
import torch
import pickle
import gc
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Disable GPTQModel by default
os.environ["NO_GPTQMODEL"] = "1"
os.environ["PYTHON_GIL"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CACHE_DIR = Path.home() / ".cache/qwen3_80b_state"
MODEL_STATE_FILE = CACHE_DIR / "model_state.pt"
TOKENIZER_STATE_FILE = CACHE_DIR / "tokenizer.pkl"
CONFIG_FILE = CACHE_DIR / "config.pkl"

def save_model_state(model, tokenizer, model_name: str):
    """Save model and tokenizer state for fast loading"""
    print("\n💾 Saving model state for fast loading...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    print("   Saving model weights...")
    start_time = time.time()
    torch.save(model.state_dict(), MODEL_STATE_FILE)
    print(f"   ✓ Model saved ({time.time() - start_time:.1f}s)")

    # Save tokenizer
    print("   Saving tokenizer...")
    with open(TOKENIZER_STATE_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    print("   ✓ Tokenizer saved")

    # Save config
    config = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'timestamp': time.time()
    }
    with open(CONFIG_FILE, 'wb') as f:
        pickle.dump(config, f)

    total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*")) / (1024**3)
    print(f"\n✅ State saved to {CACHE_DIR} ({total_size:.1f}GB)")

def load_model_state(model_name: str):
    """Load model from saved state"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not MODEL_STATE_FILE.exists():
        return None, None

    print("\n⚡ Loading from saved state (fast mode)...")

    # Load config
    with open(CONFIG_FILE, 'rb') as f:
        config = pickle.load(f)

    print(f"   Cached model: {config['model_name']}")
    print(f"   Cached on: {time.strftime('%Y-%m-%d %H:%M', time.localtime(config['timestamp']))}")

    # Load tokenizer
    print("   Loading tokenizer...")
    with open(TOKENIZER_STATE_FILE, 'rb') as f:
        tokenizer = pickle.load(f)

    # Create model architecture
    print("   Creating model architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False  # We're loading state dict manually
    )

    # Load state dict
    print("   Loading model weights...")
    start_time = time.time()
    state_dict = torch.load(MODEL_STATE_FILE, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"   ✓ Model loaded ({time.time() - start_time:.1f}s)")

    del state_dict
    gc.collect()

    return model, tokenizer

def load_model_fresh(model_name: str, save_state: bool = True):
    """Load model from scratch"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

    print("\n📂 Loading model from HuggingFace cache...")

    # Get cache path
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_cache_name = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{model_cache_name}"

    if model_path.exists():
        snapshots = model_path / "snapshots"
        snapshot_dirs = list(snapshots.iterdir())
        if snapshot_dirs:
            model_path = snapshot_dirs[0]
            print(f"   Using: {model_path}")
            model_name = str(model_path)

    # System info
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)
    print(f"\n📊 System Resources:")
    print(f"   RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

    # Load tokenizer
    print("\n⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )

    # Load model (CPU mode for INT4 quantized)
    print("⏳ Loading model (this will take 20-30 minutes)...")
    print("   Using CPU mode for INT4 quantized weights")

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,  # CPU mode
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    load_time = time.time() - start_time
    print(f"\n✅ Model loaded in {load_time:.1f}s ({load_time/60:.1f} minutes)")

    # Save state for next time
    if save_state:
        save_model_state(model, tokenizer, model_name)

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Fast-loading Qwen3-Next-80B (using state caching)"
    )

    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear saved state and reload from scratch")
    parser.add_argument("--save-only", action="store_true",
                        help="Load model and save state, then exit")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save state after loading")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-Next-80B Fast Loader")
    print("=" * 60)

    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    # Clear cache if requested
    if args.clear_cache:
        if CACHE_DIR.exists():
            print(f"\n🗑️  Clearing cache at {CACHE_DIR}")
            import shutil
            shutil.rmtree(CACHE_DIR)
            print("   ✓ Cache cleared")
        else:
            print("\n   No cache to clear")

    # Try loading from cache
    model, tokenizer = None, None
    if not args.clear_cache and MODEL_STATE_FILE.exists():
        try:
            model, tokenizer = load_model_state(model_name)
        except Exception as e:
            print(f"\n⚠️  Failed to load from cache: {e}")
            print("   Will load from scratch...")

    # Load fresh if needed
    if model is None:
        model, tokenizer = load_model_fresh(model_name, save_state=not args.no_save)

    if args.save_only:
        print("\n✅ Model saved. Exiting.")
        return 0

    # Show memory usage
    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / (1024**3)
    print(f"\n📊 Memory Usage:")
    print(f"   RAM: {ram_usage:.1f}GB used by process")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")

            print("\nGenerating...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(user_input):].strip()

            gen_time = time.time() - start_time
            print(f"\nResponse: {response}")
            print(f"Time: {gen_time:.2f}s")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())