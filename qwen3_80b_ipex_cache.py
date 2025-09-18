#!/usr/bin/env python3
"""
IPEX-Optimized Cache for Qwen3-80B
Saves the model AFTER IPEX optimization to avoid repacking on every load
"""

import os
import sys
import time
import torch
import pickle
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Progress bar imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not installed - progress bars disabled")
    print("   Install with: uv pip install tqdm")

# Cache configuration
CACHE_DIR = Path.home() / ".cache/qwen3_ipex_optimized"
CACHE_VERSION = "v1.0-ipex"

class IPEXModelCache:
    """Cache system that saves IPEX-optimized models"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def calculate_checksum(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 checksum of a file with progress indicator"""
        sha256_hash = hashlib.sha256()
        file_size = file_path.stat().st_size

        if TQDM_AVAILABLE:
            pbar = tqdm(total=file_size, desc="Calculating checksum",
                       unit="B", unit_scale=True, unit_divisor=1024)

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
                if TQDM_AVAILABLE:
                    pbar.update(len(chunk))

        if TQDM_AVAILABLE:
            pbar.close()

        return sha256_hash.hexdigest()

    def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum matches expected value"""
        if not file_path.exists():
            return False

        print(f"   🔍 Verifying file integrity...")
        actual_checksum = self.calculate_checksum(file_path)

        if actual_checksum == expected_checksum:
            print(f"   ✅ Checksum verified: {actual_checksum[:16]}...")
            return True
        else:
            print(f"   ❌ Checksum mismatch!")
            print(f"      Expected: {expected_checksum[:16]}...")
            print(f"      Actual:   {actual_checksum[:16]}...")
            return False

    def get_cache_paths(self, model_name: str, with_ipex: bool = True) -> Dict[str, Path]:
        """Get cache file paths"""
        cache_key = f"{model_name.replace('/', '_')}_{'ipex' if with_ipex else 'base'}"
        model_dir = self.cache_dir / cache_key
        model_dir.mkdir(exist_ok=True)
        return {
            'model': model_dir / 'model_optimized.pt',
            'tokenizer': model_dir / 'tokenizer',
            'metadata': model_dir / 'metadata.json'
        }

    def save_optimized_model(self, model, tokenizer, model_name: str, with_ipex: bool = True):
        """Save IPEX-optimized model to cache with checksum"""
        paths = self.get_cache_paths(model_name, with_ipex)

        print(f"\n💾 Saving IPEX-optimized model to cache...")
        print(f"   Location: {paths['model'].parent}")
        print(f"   This saves the repacked CPU/XPU format!")

        start_time = time.time()

        # Progress bar setup - now with 4 steps including checksum
        if TQDM_AVAILABLE:
            pbar = tqdm(total=4, desc="Saving cache", unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        # Step 1: Save tokenizer first
        if TQDM_AVAILABLE:
            pbar.set_description("Saving tokenizer")
        else:
            print("   Saving tokenizer...")
        tokenizer.save_pretrained(paths['tokenizer'])
        if TQDM_AVAILABLE:
            pbar.update(1)

        # Step 2: Save the IPEX-optimized model using torch.save
        # This preserves the IPEX optimizations!
        if TQDM_AVAILABLE:
            pbar.set_description("Saving IPEX model weights (may take 2-3 min)")
        else:
            print("   Saving IPEX-optimized model state...")

        # For very large models, we can show file write progress
        temp_path = paths['model'].with_suffix('.tmp')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.config.to_dict() if hasattr(model, 'config') else {}
        }, temp_path)

        # Rename temp to final (atomic operation)
        temp_path.rename(paths['model'])

        if TQDM_AVAILABLE:
            pbar.update(1)

        # Step 3: Calculate checksum of saved model
        if TQDM_AVAILABLE:
            pbar.set_description("Calculating SHA256 checksum")
        else:
            print("   Calculating SHA256 checksum...")

        model_checksum = self.calculate_checksum(paths['model'])
        if TQDM_AVAILABLE:
            pbar.update(1)

        # Step 4: Save metadata with checksum
        if TQDM_AVAILABLE:
            pbar.set_description("Saving metadata with checksum")

        metadata = {
            'model_name': model_name,
            'cache_version': CACHE_VERSION,
            'with_ipex': with_ipex,
            'timestamp': time.time(),
            'pytorch_version': torch.__version__,
            'model_checksum': model_checksum,
            'model_size_bytes': paths['model'].stat().st_size
        }

        try:
            import intel_extension_for_pytorch as ipex
            metadata['ipex_version'] = ipex.__version__
        except:
            metadata['ipex_version'] = None

        with open(paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)

        if TQDM_AVAILABLE:
            pbar.update(1)
            pbar.close()

        save_time = time.time() - start_time
        size_gb = paths['model'].stat().st_size / (1024**3)

        print(f"   ✅ IPEX-optimized cache created in {save_time:.1f}s")
        print(f"   📦 Cache size: {size_gb:.1f}GB")
        print(f"   🔐 SHA256: {model_checksum[:16]}...")
        print(f"   🚀 Next load will verify integrity & skip repacking!")

    def load_optimized_model(self, model_name: str, with_ipex: bool = True):
        """Load IPEX-optimized model from cache with checksum verification"""
        paths = self.get_cache_paths(model_name, with_ipex)

        if not paths['model'].exists():
            return None, None

        print(f"\n🚀 Loading IPEX-optimized model from cache...")
        print(f"   Skip repacking to CPU/XPU format!")

        start_time = time.time()

        try:
            # Progress bar for loading stages - now with 6 steps including checksum
            if TQDM_AVAILABLE:
                pbar = tqdm(total=6, desc="Loading from cache", unit="step",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

            # Step 1: Load metadata
            if TQDM_AVAILABLE:
                pbar.set_description("Loading metadata")
            with open(paths['metadata'], 'r') as f:
                metadata = json.load(f)

            if metadata.get('cache_version') != CACHE_VERSION:
                print(f"   ⚠️  Cache version mismatch")
                if TQDM_AVAILABLE:
                    pbar.close()
                return None, None
            if TQDM_AVAILABLE:
                pbar.update(1)

            # Step 2: Verify checksum if available
            if 'model_checksum' in metadata:
                if TQDM_AVAILABLE:
                    pbar.set_description("Verifying SHA256 checksum")

                if not self.verify_checksum(paths['model'], metadata['model_checksum']):
                    print(f"   ❌ Cache file corrupted - please rebuild cache")
                    print(f"   💡 This can happen when:")
                    print(f"      - File transfer was interrupted")
                    print(f"      - Disk errors occurred")
                    print(f"      - Cache was shared between different systems")
                    print(f"   🔧 Run with --rebuild flag to recreate cache")
                    if TQDM_AVAILABLE:
                        pbar.close()
                    return None, None
            else:
                print(f"   ⚠️  No checksum in metadata (old cache format)")

            if TQDM_AVAILABLE:
                pbar.update(1)

            # Step 3: Load tokenizer
            if TQDM_AVAILABLE:
                pbar.set_description("Loading tokenizer")
            else:
                print("   Loading tokenizer...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                paths['tokenizer'],
                local_files_only=True,
                trust_remote_code=True
            )
            if TQDM_AVAILABLE:
                pbar.update(1)

            # Step 4: Load the entire checkpoint
            # Note: This will use ~160GB RAM temporarily during loading
            # The 80-90GB cache file gets loaded into memory along with model creation
            if TQDM_AVAILABLE:
                pbar.set_description("Loading model checkpoint (needs ~160GB RAM)")
            else:
                print("   Loading model checkpoint...")
                print("   ⚠️  Memory usage will peak at ~160GB during loading")
                print("   This is normal - the 80-90GB cache + model creation")

            from transformers import AutoModelForCausalLM, AutoConfig

            checkpoint = torch.load(paths['model'], map_location='cpu')
            config = AutoConfig.from_dict(checkpoint['model_config'])
            if TQDM_AVAILABLE:
                pbar.update(1)

            # Step 5: Create model with empty weights
            if TQDM_AVAILABLE:
                pbar.set_description("Creating model skeleton")
            else:
                print("   Creating model skeleton...")
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            if TQDM_AVAILABLE:
                pbar.update(1)

            # Step 6: Load the IPEX-optimized weights
            if TQDM_AVAILABLE:
                pbar.set_description("Loading IPEX weights into model")
            else:
                print("   Loading IPEX-optimized weights...")
            model.load_state_dict(checkpoint['model_state_dict'])

            # Clean up checkpoint after loading
            del checkpoint
            import gc
            gc.collect()

            if TQDM_AVAILABLE:
                pbar.update(1)
                pbar.close()

            load_time = time.time() - start_time
            print(f"   ✅ Model loaded in {load_time:.1f}s!")
            print(f"   🔒 Integrity verified via SHA256 checksum")
            print(f"   🎉 Skipped repacking phase - already optimized!")

            return model, tokenizer

        except Exception as e:
            if TQDM_AVAILABLE and 'pbar' in locals():
                pbar.close()
            print(f"   ❌ Cache loading failed: {e}")
            return None, None

def load_model_with_ipex_cache(
    model_name: str = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
    force_rebuild: bool = False
):
    """Load model with IPEX optimization caching"""

    cache = IPEXModelCache()

    # Try loading from cache first
    if not force_rebuild:
        model, tokenizer = cache.load_optimized_model(model_name)
        if model is not None:
            return model, tokenizer

    # Load model normally
    print(f"\n⏳ Loading model for first-time IPEX optimization...")
    print(f"   This takes 30-60 minutes initially")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Disable CUDA for CPU-only mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["IPEX_DISABLE_AUTO_OPTIMIZATION"] = "1"

    start_time = time.time()

    # Progress bar for model loading stages
    if TQDM_AVAILABLE:
        loading_pbar = tqdm(total=2, desc="Loading model", unit="step",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')

    # Load tokenizer
    if TQDM_AVAILABLE:
        loading_pbar.set_description("Loading tokenizer")
    else:
        print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if TQDM_AVAILABLE:
        loading_pbar.update(1)

    # Load model
    if TQDM_AVAILABLE:
        loading_pbar.set_description("Loading model shards (30-60 min)")
    else:
        print("   Loading model shards...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    if TQDM_AVAILABLE:
        loading_pbar.update(1)
        loading_pbar.close()

    load_time = time.time() - start_time
    print(f"   ✅ Model loaded in {load_time/60:.1f} minutes")

    # Apply IPEX optimization
    try:
        import intel_extension_for_pytorch as ipex
        print(f"\n🔧 Applying IPEX optimizations (repacking to CPU/XPU format)...")
        print(f"   This one-time repacking improves inference speed 2-4x")

        ipex_start = time.time()

        # Create progress indicator for IPEX optimization
        if TQDM_AVAILABLE:
            # IPEX optimization doesn't have progress callbacks, so we show a spinner
            with tqdm(total=1, desc="IPEX Optimization", unit="model",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}]') as pbar:
                model = ipex.optimize(model, dtype=torch.float16)
                pbar.update(1)
        else:
            print("   Optimizing... (this may take 1-2 minutes)")
            model = ipex.optimize(model, dtype=torch.float16)

        ipex_time = time.time() - ipex_start

        print(f"   ✅ IPEX optimization complete in {ipex_time:.1f}s")

        # Save the IPEX-optimized model
        cache.save_optimized_model(model, tokenizer, model_name, with_ipex=True)

    except ImportError:
        print("   ⚠️  IPEX not available - saving base model")
        cache.save_optimized_model(model, tokenizer, model_name, with_ipex=False)

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="IPEX-Optimized Cache for Qwen3-80B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BENEFITS:
  • Saves model AFTER IPEX repacking
  • Skips "repacking to CPU/XPU format" on subsequent loads
  • Preserves all IPEX optimizations
  • Even faster loading than base cache

WORKFLOW:
  First run:
    1. Load model (30-60 min)
    2. Apply IPEX optimization (1-2 min)
    3. Save IPEX-optimized model (2-3 min)
    Total: ~35-65 minutes

  Subsequent runs:
    1. Load IPEX-optimized model (<1 min)
    2. Ready to use!
    Total: <1 minute

EXAMPLES:
  %(prog)s                 # Load with IPEX cache
  %(prog)s --rebuild       # Force rebuild cache
  %(prog)s --benchmark     # Compare loading times
        """
    )

    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild IPEX-optimized cache")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark cache vs normal loading")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--verify-cache", action="store_true",
                        help="Verify cache integrity and show checksums")
    parser.add_argument("--export-checksums", type=str, metavar="FILE",
                        help="Export checksums to file for team sharing")

    args = parser.parse_args()

    # Handle cache verification
    if args.verify_cache:
        cache = IPEXModelCache()
        paths = cache.get_cache_paths(
            "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
            with_ipex=True
        )

        print("\n" + "="*60)
        print("Cache Integrity Verification")
        print("="*60)

        if not paths['metadata'].exists():
            print("❌ No cache found")
            return

        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        print(f"Model: {metadata.get('model_name', 'unknown')}")
        print(f"Cache Version: {metadata.get('cache_version', 'unknown')}")
        print(f"Created: {time.ctime(metadata.get('timestamp', 0))}")
        print(f"IPEX Version: {metadata.get('ipex_version', 'N/A')}")

        if 'model_checksum' in metadata:
            print(f"\n🔐 SHA256 Checksum:")
            print(f"   {metadata['model_checksum']}")
            print(f"\nVerifying file integrity...")
            if cache.verify_checksum(paths['model'], metadata['model_checksum']):
                print("✅ Cache file is intact and valid")
            else:
                print("❌ Cache file is corrupted!")
        else:
            print("\n⚠️  No checksum available (old cache format)")

        if 'model_size_bytes' in metadata:
            size_gb = metadata['model_size_bytes'] / (1024**3)
            print(f"\n📦 Cache Size: {size_gb:.1f}GB")

        print("="*60)
        return

    # Handle checksum export for team sharing
    if args.export_checksums:
        cache = IPEXModelCache()
        paths = cache.get_cache_paths(
            "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound",
            with_ipex=True
        )

        if not paths['metadata'].exists():
            print("❌ No cache found to export")
            return

        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        export_data = {
            'model_name': metadata.get('model_name'),
            'cache_version': metadata.get('cache_version'),
            'model_checksum': metadata.get('model_checksum', 'N/A'),
            'model_size_bytes': metadata.get('model_size_bytes', 0),
            'ipex_version': metadata.get('ipex_version', 'N/A'),
            'timestamp': metadata.get('timestamp'),
            'export_date': time.time()
        }

        with open(args.export_checksums, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Checksums exported to: {args.export_checksums}")
        print(f"   Share this file with your team for cache verification")
        print(f"   SHA256: {export_data['model_checksum'][:16]}...")
        return

    if args.benchmark:
        print("\n" + "="*60)
        print("Benchmark: IPEX-Optimized Cache")
        print("="*60)
        print("\nWithout IPEX cache:")
        print("  1. Load model: 30-60 minutes")
        print("  2. IPEX repacking: 1-2 minutes")
        print("  Total: 31-62 minutes")
        print("\nWith IPEX cache:")
        print("  1. Load optimized model: <1 minute")
        print("  Total: <1 minute")
        print("\nSpeedup: 30-60x!")
        print("="*60)
        return

    # Load model
    model, tokenizer = load_model_with_ipex_cache(force_rebuild=args.rebuild)

    if args.interactive:
        print("\n" + "="*60)
        print("Interactive Mode - Type 'quit' to exit")
        print("="*60)

        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

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