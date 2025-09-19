# Qwen3-NEXT-80B AutoRound

EDUCATIONAL and EXPERIMENTAL while waiting for `llama.cpp` integration: run Qwen3-Next-80B FP8+FP4 model on consumer hardware using Intel's 4-bit AutoRound quantization, CPU, IPEX, GPU, Flash Linear Attention, GPTQ, and more.

## TL;DR Quick Start

```bash
# Clone and setup
git clone https://github.com/PieBru/Qwen3-NEXT-80B_AutoRound.git
cd Qwen3-NEXT-80B_AutoRound

# Option A: Python 3.11/3.12 (RECOMMENDED - 2-4x faster CPU inference with IPEX)
./setup_python311_ipex.sh  # Complete setup with IPEX

# Option B: Python 3.13+ (if already installed)
pyenv install 3.13.3t
pyenv local 3.13.3t
uv venv
source .venv/bin/activate
export PYTHON_GIL=0  # For Python 3.13 only

# Install dependencies (auto-detects Python version)
python install_requirements.py

# Authenticate with HuggingFace (required for model download)
hf auth login

# For Consumer Hardware (64-128GB RAM):
./setup_swap.sh  # Auto-configures swap if needed for IPEX
python qwen3_80b.py --load-strategy no-gpu --interactive
# After first run: ./remove_swap.sh

# Quick Commands:
python qwen3_80b.py --help           # See all options
python qwen3_80b.py --test-deps      # Verify environment
python qwen3_80b.py --thinking       # Show reasoning process
python qwen3_80b.py --benchmark      # Run performance test
python benchmark_hardware.py         # Detailed hardware benchmark
```

## Features

- ✅ **Fast Caching System** (v3.4) - **Enabled by default!** Load model in <1 minute after first run (vs 30-60 minutes)
- ✅ **Consumer Hardware Support** - Optimized for GPU-poor (<20GB VRAM) and low-RAM (64-128GB) systems with automatic swap management and mandatory IPEX for usable inference speeds
- ✅ **Hardware Performance Benchmarking** - Comprehensive testing of memory bandwidth, CPU performance, and storage I/O to predict model inference speed
- ✅ **Cache Sharing** - Create portable cache files to share with team members for faster initial loads
- ✅ **Multi-threaded Loading** - Optimized shard loading using multiple CPU threads for faster model initialization
- ✅ **Smart Loading Strategy** (v3.0) - Auto-detects resources and picks optimal loading strategy, no more double loading!
- ✅ **4-bit Quantized Inference** - Run 80B models with <64GB memory footprint
- ✅ **Chain-of-Thought Support** - Parse and optionally display the model's thinking process
- ✅ **Interactive Chat Mode** - Conversational CLI (WebUI coming soon)
- ✅ **Memory Efficient** - Automatic and manual GPU memory management and CPU offloading
- 🚧 **OpenAI-Compatible API** - Coming soon

## 🎯 Guide for Consumer Hardware (64-128GB RAM, <20GB VRAM)

If you have a typical gaming PC or workstation with limited resources, this section is for you!

### Your Hardware Profile
- **RAM**: 64-128GB (common for enthusiasts)
- **GPU**: RTX 3090/4070/4080 with <24GB VRAM
- **CPU**: Intel/AMD with AVX2/AVX-512 support
- **Goal**: Run 80B model with decent speed on consumer hardware

### Quick Setup for Consumer Hardware

```bash
# 1. Check your system
free -h  # Check RAM and swap

# 2. If you have <80GB total (RAM + swap), run our auto-setup:
./setup_swap.sh

# 3. Run the model (IPEX will be applied automatically)
python qwen3_80b.py --load-strategy no-gpu --interactive

# 4. After first run completes, remove temporary swap:
./remove_swap.sh  # Created by setup_swap.sh

# Future runs: Super fast (<1 minute load) with just 45GB RAM!
python qwen3_80b.py --load-strategy no-gpu --interactive
```

### Why This Works

1. **First Run**: Uses temporary swap for IPEX optimization (one-time only)
2. **IPEX Benefit**: 2-4x faster inference on CPU (essential for usability!)
3. **Smart Caching**: Saves IPEX-optimized model for instant future loads
4. **After First Run**: Only needs 45GB RAM, no swap required

### Performance Expectations

| Hardware | First Run | Future Runs | Inference Speed |
|----------|-----------|-------------|-----------------|
| 64GB RAM + swap | 90-120 min | <1 minute | 2-5 tokens/sec |
| 128GB RAM | 60-90 min | <1 minute | 3-7 tokens/sec |
| With IPEX | Required! | Cached | 2-4x faster |
| Without IPEX | Faster load | No cache | 0.5-2 tokens/sec (unusable) |

**Bottom Line**: IPEX is NOT optional for consumer hardware - it's the difference between unusable (0.5 tok/s) and decent (3-5 tok/s) performance!

## Requirements

### Hardware Requirements
- **Disk Space**:
  - ~64GB for original model download
  - Additional ~90GB for IPEX cache (if using IPEX optimization)
  - Total: ~150GB free disk space recommended
- **For CPU inference**: 117GB+ RAM (300GB for first-time IPEX optimization - use swap, see [Consumer Guide](#guide-for-consumer-hardware-64-128gb-ram-20gb-vram))
- **For GPU inference**: 30GB+ VRAM (or use CPU mode)
- OPTIONAL: NVIDIA GPU with CUDA 12.1+

### Python Version Requirements
- **Recommended**: Python 3.11 or 3.12 (for Intel Extension support - 2-4x CPU speedup)
- **Alternative**: Python 3.13+ (works but no IPEX support yet)
- For Python 3.13: Use free-threaded build with `export PYTHON_GIL=0`

### 📁 Requirements Structure

#### 1. **`requirements-base.txt`**
- Common dependencies that work across all Python versions
- Includes Transformers, AutoRound, utilities

#### 2. **`requirements-py311-312.txt`** (RECOMMENDED)
- For Python 3.11 and 3.12
- **Includes Intel Extension for PyTorch (IPEX)**
- Provides 2-4x CPU inference speedup
- Optimal performance configuration

#### 3. **`requirements-py313.txt`**
- For Python 3.13+
- No IPEX support (not compatible with Python 3.13 yet)
- Alternative optimizations included
- Works with free-threaded Python builds

#### 4. **`requirements.txt`**
- Main file with instructions
- Points to appropriate version-specific files
- Minimal requirements for basic functionality

### 🚀 Installation Options

#### Automatic Installation (Recommended)
```bash
# Automatically detects Python version and installs appropriate dependencies
python install_requirements.py
```

#### Manual Installation by Python Version
```bash
# For Python 3.11 or 3.12 (best performance)
pip install -r requirements-py311-312.txt

# For Python 3.13+
pip install -r requirements-py313.txt
```

#### Setup Python 3.11 Environment for Best Performance
```bash
# Complete setup script for Python 3.11 with IPEX
./setup_python311_ipex.sh
```

### 📊 Performance Comparison

#### Python Version Impact
| Python Version | IPEX Support | CPU Performance | Recommendation   |
|----------------|--------------|-----------------|------------------|
| 3.11-3.12      | ✅ Yes       | 2-4x faster     | **Best choice**  |
| 3.13+          | ❌ No        | Baseline        | Works but slower |

#### Hardware Performance (Tokens/Second)
| GPU | VRAM | CPU | RAM | RAM Speed | Mode | w/ IPEX | w/o IPEX | Load Time |
|-----|------|-----|-----|-----------|------|---------|----------|-----------|
| RTX 4090 Desktop | 24GB | *Your CPU* | *Size* | *Speed* | GPU¹ | N/A | *tok/s* | *time* |
| RTX 4090 Laptop | 16GB | i9-14900HX | 128GB | 5600MHz | CPU² | *tok/s* | *tok/s* | *time* |
| RTX 3090 | 24GB | *Your CPU* | *Size* | *Speed* | CPU² | *tok/s* | *tok/s* | *time* |
| RTX 4080 | 16GB | *Your CPU* | *Size* | *Speed* | CPU² | *tok/s* | *tok/s* | *time* |
| RTX 4070 Ti | 12GB | *Your CPU* | *Size* | *Speed* | CPU² | *tok/s* | *tok/s* | *time* |
| Intel Arc A770 | 16GB | *Your CPU* | *Size* | *Speed* | CPU² | *tok/s* | *tok/s* | *time* |
| AMD RX 7900 XTX | 24GB | *Your CPU* | *Size* | *Speed* | CPU² | *tok/s* | *tok/s* | *time* |
| CPU Only | - | *Your CPU* | *Size* | *Speed* | CPU | *tok/s* | *tok/s* | *time* |

¹ Full GPU mode requires 30GB+ VRAM for INT4 model
² GPUs with <24GB VRAM must use CPU mode due to INT4 limitations

**Example Entry (Your System):**
- **GPU:** RTX 4090 Laptop (16GB VRAM)
- **CPU:** Intel Core i9-14900HX
- **RAM:** 128GB DDR5-5600 (CT2K64G56C46S5)
- **Mode:** CPU (due to <24GB VRAM limitation)
- **Performance:** *Pending benchmark results*

**Help us fill this table!** Run `python benchmark_hardware.py` and submit your results via PR or issue.

### Key Benefits

1. **Version-specific optimization**: Each Python version gets the best available packages
2. **IPEX for 3.11/3.12**: Major performance boost for CPU inference
3. **Clear documentation**: Users know exactly what to install
4. **Automatic detection**: The installer script handles version detection
5. **Fallback options**: Python 3.13 users still get a working setup

The intelligent installer (`install_requirements.py`) will:
- Detect your Python version
- Check for CUDA/GPU availability
- Install the optimal requirements
- Verify the installation
- Provide next steps

## Smart Loading Strategy (v3.0) 🎯

The loader now automatically detects your hardware and chooses the optimal loading strategy, **eliminating the double shard loading problem** that wasted 30-60 minutes!

### How It Works

1. **Pre-flight Check**: Detects VRAM and RAM before loading
2. **Smart Decision**: Chooses `full_gpu` or `cpu_only` based on resources
3. **No Retries**: Goes straight to the right strategy - no failed attempts!

### Loading Time Savings

| Your Hardware | Old Behavior | New Behavior | Time Saved |
|--------------|--------------|--------------|------------|
| 16GB VRAM + 128GB RAM | Try GPU (30-60min) → Fail → CPU (30-60min) | Straight to CPU (30-60min) | **30-60 minutes saved!** |
| 24GB VRAM + 64GB RAM | Try GPU (30-60min) → Fail → CPU (30-60min) | Straight to CPU (30-60min) | **30-60 minutes saved!** |
| 48GB VRAM + 64GB RAM | GPU Success (30-60min) | GPU Success (30-60min) | No change |

### Why This Matters

The INT4 quantized model requires either:
- **ALL weights on GPU** (needs 30GB+ VRAM), or
- **ALL weights on CPU** (needs 50GB+ RAM)
- **NOT mixed** (fails with "b_q_weight is not on GPU")

### Usage

```bash
# Automatic strategy selection (default)
./qwen3_80b.py

# See what strategy would be used without loading
./qwen3_80b.py --dry-run

# Force specific strategies
./qwen3_80b.py --cpu            # Force CPU-only
./qwen3_80b.py --use-gptq       # For 30GB+ VRAM (multi-GPU)
```

See [LOADING_STRATEGIES.md](docs/LOADING_STRATEGIES.md) for complete resource matrix.

## Fast Caching System 🚀 (Default for CPU Mode)

**No more waiting 30-60 minutes!** Caching is now **enabled by default** for CPU-only loading - the model loads in **<1 minute** after the first run!

### ⚠️ Important: Caching Limitation

**Caching only works with CPU-only mode** (`--load-strategy no-gpu`). Hybrid GPU+CPU loading cannot be cached due to memory offloading hooks used by the accelerate library.

### How It Works

1. **First Run**: Normal loading (30-60 min) + automatic cache creation (2-3 min)
2. **All Future Runs**: Load from cache in <1 minute! (30-50x faster)
3. **CPU-Only Required**: Use `--load-strategy no-gpu` for cacheable loading

### Quick Start for Cacheable Loading

```bash
# For cacheable CPU-only mode (RECOMMENDED for caching):
python qwen3_80b.py --load-strategy no-gpu --interactive

# First run: 30-60 minutes (creates cache)
# Future runs: <1 minute! (uses cache)

# For GPU+CPU hybrid (faster inference but NO caching):
python qwen3_80b.py --interactive  # Uses min-gpu by default

# If you need to bypass cache for testing:
python qwen3_80b.py --bypass-cache --load-strategy no-gpu --interactive

# Show thinking process (works with caching):
python qwen3_80b.py --thinking --load-strategy no-gpu --interactive
```

### Cache Management

```bash
# Clear cache and start fresh
python qwen3_80b.py --clear-cache

# Rebuild cache (if corrupted)
python qwen3_80b.py --rebuild-cache --interactive

# Disable cache temporarily (always load from scratch)
python qwen3_80b.py --bypass-cache --interactive

# Check cache performance
python test_cache_performance.py
```

### Performance Comparison

| Loading Method | Strategy | Time | Cacheable |
|---------------|----------|------|-----------|
| CPU-only (no cache) | `--load-strategy no-gpu` | 30-60 minutes | ✅ Yes |
| CPU-only (with cache) | `--load-strategy no-gpu` | <1 minute | ✅ Yes |
| Hybrid GPU+CPU | `--load-strategy min-gpu` | 30-60 minutes | ❌ No |
| Full GPU | `--load-strategy max-gpu` | 30-60 minutes | ❌ No |

### Cache Details

- **Location**: `~/.cache/qwen3_fast_loader/`
- **Size**: ~30GB (preserves 4-bit quantization)
- **Format**: Pickled PyTorch model (maintains exact state)
- **Compatibility**: Works with CPU and GPU modes

⚠️ **Known Issue**: Pickle cache loading can use ~160GB RAM temporarily. Consider using IPEX cache instead for better memory efficiency.

## Intel Extension for PyTorch (IPEX) Optimization 🚀

### New: Threading Optimization (v3.6)

The scripts now use **physical cores only** instead of all logical threads for better performance:
- **Why**: Hyperthreading hurts performance for memory-intensive operations
- **Benefit**: Better cache efficiency and reduced memory contention
- **Example**: On a 24-core CPU with hyperthreading:
  - Before: 48 threads (oversubscribed, cache thrashing)
  - After: 24 threads (one per physical core, optimal)

### New: IPEX Cache with Progress & Integrity Checking (v3.5)

The IPEX cache system now includes:
- **Progress Indicators**: Visual feedback during optimization and saving
- **SHA256 Checksums**: Automatic integrity verification
- **Team Sharing**: Export/verify checksums for collaboration
- **Cache Size**: ~80-90GB (2x larger than original 41GB model due to FP16 expansion)

#### Quick Start with Integrated Caching

```bash
# CPU mode automatically uses IPEX optimization for memory-efficient caching
python qwen3_80b.py --load-strategy no-gpu --interactive

# Check cache status (shows both HuggingFace and fast-load cache)
python qwen3_80b.py --check

# Use custom cache directory for large models
python qwen3_80b.py --cache-dir /mnt/large_drive/model_cache

# Force rebuild cache
python qwen3_80b.py --rebuild-cache
```

#### Progress Indicators

Install `tqdm` for visual progress during:
- Model loading (2 steps)
- IPEX optimization (real-time progress)
- Cache saving (4 steps with checksum)
- Cache loading (6 steps with verification)

```bash
# Install progress bar support
uv pip install tqdm
```

#### Integrity Verification

Every cached model now includes SHA256 checksum:
- **Automatic**: Verified on every load
- **Manual Check**: `--verify-cache` command
- **Team Sharing**: Export/import checksums
- **Corruption Detection**: Prevents loading corrupted files
- **File Size**: Cache will be ~80-90GB (2x larger than original)

Example output:
```
🚀 Loading IPEX-optimized model from cache...
   🔍 Verifying file integrity...
   ✅ Checksum verified: 3f4a2b5c8d9e1f6a...
   📦 Cache size: 85.3GB
   ✅ Model loaded in 45.2s!
   🔒 Integrity verified via SHA256 checksum
```

### What is IPEX "Repacking to CPU/XPU Format"?

When you see the message "repacking to CPU/XPU format", IPEX is optimizing the model for CPU execution. This process:

1. **Transforms model weights** into blocked memory layout for better CPU cache utilization
2. **Creates CPU-optimized kernels** using AVX-512 and other CPU instructions
3. **Applies graph-level optimizations** for faster inference
4. **Results in 2-4x speedup** for CPU inference

### Where Does Repacking Happen?

The repacking occurs when `ipex.optimize()` is called after model loading. This is a one-time optimization that modifies the model in memory.

### Memory Requirements During Repacking

⚠️ **Important**: IPEX repacking and cache saving temporarily requires **~300GB total memory**!

| Phase | Memory Usage | Duration |
|-------|--------------|----------|
| Model loaded | ~117GB | - |
| IPEX repacking starts | ~117GB → ~250GB peak | 1-2 minutes |
| Saving IPEX cache | ~250GB → ~300GB peak | 2-3 minutes |
| Complete & cached | ~117GB (runtime) | - |

### Using Temporary Swap for IPEX Optimization

**Confirmed Working Configuration**: 128GB RAM + 256GB swap = 384GB total (enough for ~300GB peak)

If you have less than 300GB total memory, you'll need **temporary swap space** for the one-time IPEX optimization. The model uses ~117GB RAM normally, but IPEX optimization and cache saving temporarily peaks at ~300GB.

#### Recommended: Create a Large 256GB Swap File

For systems with 64-128GB RAM, we recommend creating a single large swap file to ensure IPEX optimization completes successfully:

```bash
# 1. Check current swap status
free -h
sudo swapon --show

# 2. Turn off ALL existing swap files
sudo swapoff -a

# 3. Remove old swap files (if any exist)
sudo rm -f /swapfile /swapfile2 /swapfile_qwen3*

# 4. Verify disk space available (need 256GB free)
df -h /

# 5. Create a single 256GB swap file
sudo fallocate -l 256G /swapfile_256g

# If fallocate fails, use dd instead (slower but more reliable):
# sudo dd if=/dev/zero of=/swapfile_256g bs=1G count=256 status=progress

# 6. Set correct permissions
sudo chmod 600 /swapfile_256g

# 7. Format as swap
sudo mkswap /swapfile_256g

# 8. Enable the swap
sudo swapon /swapfile_256g

# 9. Verify it's working
free -h
swapon --show
# Should show something like:
#                total        used        free
# Mem:           125Gi       7.4Gi       118Gi
# Swap:          256Gi       0.0Gi       256Gi
```

#### Step 2: Run Model with IPEX (One-Time Optimization)

```bash
# First run: Uses swap for IPEX optimization
python qwen3_80b.py --load-strategy no-gpu --interactive

# The process will:
# 1. Load model (~117GB RAM usage)
# 2. IPEX optimization (peaks at ~230GB, WILL use swap)
# 3. Save optimized cache
# 4. Run normally (~117GB RAM)

# Monitor memory usage in another terminal:
watch -n1 free -h
```

#### Step 3: Remove Swap (After Cache Created)

Once the IPEX-optimized cache is created, you can remove the swap:

```bash
# Remove the 256GB swap after cache is created
sudo swapoff /swapfile_256g
sudo rm /swapfile_256g
free -h

# Verify cache was created
ls -lh ~/.cache/qwen3_fast_loader/
# Should show ~40GB cache file
```

#### Memory Requirements Summary

| Configuration | First Run (with IPEX) | Future Runs (cached) |
|--------------|----------------------|---------------------|
| RAM needed | ~117GB | ~117GB runtime |
| IPEX optimization peak | ~250GB total | N/A (already optimized) |
| Cache saving peak | ~300GB total | N/A (already cached) |
| Cache loading peak | N/A | ~160GB (temporary) |
| Swap recommended | 256GB | Small swap helpful |
| Total memory | 128GB RAM + 256GB swap = 384GB | 160GB RAM ideal |

**Confirmed**:
- **First run**: 128GB RAM + 256GB swap handles the ~300GB peak
- **Future runs**: ~160GB peak during cache loading, then drops to 117GB runtime

#### Why This Much Swap?

- The quantized model uses **~117GB RAM** at runtime
- IPEX optimization peaks at **~250GB total** during repacking
- Cache saving peaks at **~300GB total** when writing optimized weights
- 256GB swap ensures the process completes without OOM kills
- This is a **one-time requirement** - after caching, only 117GB RAM is needed

#### Alternative: Skip IPEX If Limited Disk Space

If you can't create 256GB swap, you can skip IPEX optimization:

```bash
# Run without IPEX (no memory spike, but 3-4x slower inference)
python qwen3_80b.py --load-strategy no-gpu --no-ipex --interactive
```

**Warning**: Without IPEX, expect only 0.5-2 tokens/sec instead of 2-8 tokens/sec!

### Can We Repack Directly to Disk?

**Unfortunately NO** - Technical limitations:
- IPEX optimization works on PyTorch tensors in memory
- The repacking involves complex transformations that can't be streamed
- Graph optimizations need the complete model structure
- The process temporarily needs ~2x model memory (80GB peak for our 40GB model)

### Our Solution: IPEX-Optimized Caching

We save the model **AFTER** IPEX optimization, so you only need extra memory once:

**First Run (with caching on 128GB RAM system):**
1. Load model (~117GB RAM) - 30-60 minutes
2. Apply IPEX optimization (~250GB peak, uses swap) - 1-2 minutes
3. Save IPEX-optimized cache (~300GB peak, uses swap) - 2-3 minutes
   - Creates ~80-90GB cache file (2x original 41GB due to FP16 expansion)
4. Total: ~35-65 minutes, needs 128GB RAM + 256GB swap

**All Future Runs:**
1. Load IPEX-optimized cache (<1 minute)
   - ⚠️ **WARNING**: Memory usage during loading can exceed 200GB!
   - The 80-90GB cache file + model creation = massive memory spike
   - Memory-mapped loading helps but still requires substantial RAM
   - After loading completes, drops back to ~117GB runtime memory
2. No repacking needed!
3. **Recommended**: 200GB+ total memory (RAM + swap) for safe loading
   - 128GB RAM + 128GB swap should handle it
   - Monitor with `watch -n1 free -h` during loading

**Why is the cache 2x larger?**
- Original model: 41GB (4-bit quantized, compressed)
- IPEX cache: 80-90GB (FP16 expanded, CPU-optimized layout)
- Trade-off: Larger disk space for 2-4x faster CPU inference

### IPEX Cache Portability: Who Can "Cook" vs "Eat"

The IPEX-optimized cache has interesting portability characteristics:

#### Who Can "Cook" (Create) the IPEX Cache:
- ✅ **Intel CPUs only** (Core i5/i7/i9, Xeon)
- ✅ Requires Intel Extension for PyTorch installed
- ✅ Needs ~230GB total memory for optimization
- ❌ AMD CPUs cannot create IPEX cache (no IPEX support)
- ❌ ARM/Apple Silicon cannot create IPEX cache

#### Who Can "Eat" (Use) the IPEX Cache:
- ✅ **Intel CPUs** - Full performance benefit
- ✅ **AMD CPUs** - Yes! Can load and use pre-optimized cache
- ✅ Any x86-64 CPU with AVX2 support
- ⚠️ Performance varies by CPU but memory layout benefits remain
- ❌ ARM/Apple Silicon incompatible

#### What This Means in Practice:

```
Scenario 1: Intel User (can Cook & Eat)
- Create your own IPEX cache
- Share cache with AMD friends
- Full 2-4x performance boost

Scenario 2: AMD User (can only Eat)
- Cannot create IPEX cache themselves
- CAN use IPEX cache from Intel users
- Still gets memory layout benefits (~1.5-2x boost)

Scenario 3: Cache Sharing
- One Intel user creates cache (needs 256GB swap once)
- Shares 40GB cache file with team
- AMD users benefit without needing IPEX
```

**The Magic**: The IPEX cache contains reorganized PyTorch tensors in CPU-optimized memory layouts, not Intel-specific binary code. This is why AMD CPUs can still benefit from the pre-optimized format even without IPEX installed!

### Memory Requirements Summary

| Configuration | First Run | Subsequent Runs |
|--------------|-----------|-----------------|
| Without IPEX | 117GB RAM | 117GB RAM |
| With IPEX (no cache) | ~300GB peak (needs swap) | ~300GB peak (needs swap) |
| **With IPEX + Cache** | **~300GB peak (one-time)** | **~160GB peak loading, 117GB runtime** |

### IPEX Usage

```bash
# CPU-only with IPEX (default, provides 2-4x speedup)
python qwen3_80b.py --load-strategy no-gpu

# Disable IPEX if you have limited RAM
python qwen3_80b.py --load-strategy no-gpu --no-ipex

# Clear IPEX cache and rebuild
python qwen3_80b.py --clear-cache --rebuild-cache
```

### Alternative: Direct IPEX Loading (No Cache)

If you're running into memory issues during cache creation:

```bash
# IPEX is enabled by default for memory-efficient caching
python qwen3_80b.py --load-strategy no-gpu --interactive

# Skip caching entirely if memory is too limited
python qwen3_80b.py --load-strategy no-gpu --bypass-cache --interactive

# Use custom cache directory on a larger drive
python qwen3_80b.py --cache-dir /mnt/large_drive/cache --interactive

# This approach:
# - IPEX uses torch.save (memory efficient, ~1x model size)
# - Without IPEX, uses pickle (needs ~3x model size in RAM)
# - Cache creation is automatic on first run
```

The good news: **You only need the extra RAM once!** After the first run, the IPEX-optimized model loads directly from cache in <1 minute without any repacking.

## Usage

### Quick Start (v3.0 - Smart Loading by Default)

```bash
# Interactive chat mode (hybrid CPU/GPU for <30GB VRAM)
python qwen3_80b.py

# Show thinking/reasoning process
python qwen3_80b.py --thinking

# Run performance benchmark
python qwen3_80b.py --benchmark

# For multi-GPU systems (enable GPTQModel)
python qwen3_80b.py --use-gptq

# Check if model is cached locally
python qwen3_80b.py --check

# Test dependencies
python qwen3_80b.py --test-deps

# Get help
python qwen3_80b.py --help
```

### New Tools for Consumer Hardware

```bash
# Automatic swap setup for IPEX (64-128GB RAM systems)
./setup_swap.sh         # Calculates and adds needed swap
./remove_swap.sh        # Removes swap after cache creation

# Hardware performance benchmarking
python benchmark_hardware.py  # Detailed system benchmark
                             # Outputs README-ready table row
                             # Saves JSON results

# Low-memory experimental loader
python qwen3_80b_low_mem.py  # Aggressive GC before IPEX
```

### Advanced Usage

```bash
# Custom memory limits (in GiB)
python qwen3_80b.py --gpu-memory 12 --cpu-memory 80

# CPU-only mode (no GPU)
python qwen3_80b.py --cpu

# Custom generation parameters
python qwen3_80b.py --temperature 0.9 --max-tokens 200

# Verbose mode with detailed information
python qwen3_80b.py --verbose

# Quiet mode (minimal output)
python qwen3_80b.py --quiet
```

The model uses special token 151668 (`</think>`) to separate internal reasoning from responses.

## Model Details

- **Model**: Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound
- **Size**: ~41GB on disk (9 shards of ~4.7GB each)
- **Memory Required**: ~58GB when loaded (4-bit quantized with INT4/FP8 mixed precision)
- **Original Size**: ~160GB (BF16 unquantized)
- **Quantization**: Intel AutoRound v0.5 with SmoothQuant optimization
- **Special Features**: Thinking tokens (151668 for `</think>`) for chain-of-thought reasoning
- **Context Length**: Supports full context as per original model
- **Calibration**: 256 samples, 2048 sequence length, 2/256 learning rate

## Performance

**⚠️ IMPORTANT: First load takes 30-60 minutes**
- The model has 9 shards of ~4.5GB each that load sequentially (single-threaded limitation)
- v3.0 prevents double loading - saves 30-60 minutes for <30GB VRAM users!
- Subsequent loads use OS file cache (slightly faster)

Expected performance (once loaded):
- **CPU-only**: 0.5-5 tokens/second (50GB+ RAM required)
- **GPU 30GB+**: 50-100 tokens/second (full GPU loading)
- **GPU <30GB**: CPU-only mode (INT4 doesn't support mixed placement)

### Understanding "Loading checkpoint shards"

The "Loading checkpoint shards" message refers to **loading the model from disk into RAM**, not downloading. Each shard is being:

1. **Read from disk** (your local cache at `~/.cache/huggingface/hub/`)
2. **Loaded into RAM** (system memory)
3. **Quantized/dequantized** on-the-fly (4-bit weights need processing)
4. **Mapped to GPU/CPU** based on the device_map

The slow speed (~2.5 minutes per shard) is due to:
- **Single-threaded library limitation** - The transformers library's safetensors loader only uses 1 CPU thread
- **On-the-fly quantization** - 4-bit AutoRound weights are being processed during loading
- **Memory mapping** - 80B parameters being distributed across GPU and CPU memory
- **Sequential processing** - Each shard must be loaded and processed in order

This is normal for such a large quantized model. The 9 shards totaling 41GB will take 30-60 minutes to fully load. You can monitor:
- **CPU usage** with `htop` - only 1 thread at 100% (this is the limitation)
- **RAM usage** with `htop` - should gradually increase to ~58GB
- **Disk I/O** with `iotop` - intermittent reads as shards load
- **Network** with `nethogs` - should be zero (using local files)

The loading is slow but working correctly - it's a combination of the single-threaded loader and the complexity of loading 80 billion quantized parameters!

## Troubleshooting

### Model Loading Issues
If the model appears stuck during loading:
1. **BE PATIENT** - The 80B parameter model device mapping is extremely slow on first run
2. Check system resources: `htop` (CPU/RAM) or `nvidia-smi` (GPU)
3. Try the simplified loader: `python run_simple.py` (better progress feedback)
4. Verify model is cached: `python check_model.py`
5. For CPU-only mode, ensure you have 64GB+ RAM available
6. Use `export PYTHON_GIL=0` to avoid RuntimeWarnings with free-threaded Python

### "b_q_weight is not on GPU" Error (FIXED in v3.0)
**Previous Issue**: The INT4 quantized model would try GPU loading, fail, then retry with CPU - causing double shard loading (50+ minutes total).

**v3.0 Solution**: The script now detects your VRAM upfront:
- **<30GB VRAM**: Goes straight to CPU-only mode (no failed GPU attempt)
- **≥30GB VRAM**: Uses full GPU loading
- **Result**: Saves 30-60 minutes by avoiding the retry!

If you still see this error with older versions:
- **Cause**: INT4 quantization requires all weights on same device (GPU or CPU, not mixed)
- **Fix**: Update to v3.0 or use `--cpu` flag to force CPU-only mode

### MoE Expert Offloading (Limited Support)

For GPUs with limited VRAM (<30GB), the script automatically uses CPU-only mode:

```bash
# DEFAULT: Auto-detects and uses appropriate mode
./qwen3_80b.py

# Force specific modes
./qwen3_80b.py --cpu         # Force CPU-only
./qwen3_80b.py --use-gptq    # Enable GPTQModel (30GB+ VRAM only)
```

**Note**: The INT4 quantized model doesn't support mixed CPU/GPU placement well. For <30GB VRAM, it uses CPU-only mode to avoid loading failures.

**Default Behavior (v3.0):**
- GPTQModel is **disabled by default** for better compatibility
- Automatically enables hybrid CPU/GPU loading for GPUs with <30GB VRAM
- Standard transformers handles device mapping intelligently
- MoE experts are offloaded to CPU RAM automatically

**When to use --use-gptq:**
- Multi-GPU systems where GPTQModel can distribute across GPUs
- Systems with sufficient VRAM (30GB+) for full GPU loading
- When specifically optimizing for multi-threaded CPU operations

**MoE Architecture Details:**
- Qwen3-Next-80B has 512 experts with 10 experts activated per token
- Non-expert layers (attention, embeddings) can fit in ~8GB VRAM
- Expert layers require ~32GB (offloaded to CPU RAM)
- Trade-off: Slower inference but enables running on consumer GPUs

**Recommended for Limited VRAM:**
```bash
# For 16GB GPUs - disables GPTQModel, enables hybrid loading
./qwen3_80b.py --no-gptq

# With custom memory limits
./qwen3_80b.py --no-gptq --gpu-memory 14 --cpu-memory 80
```

### GPU Memory Warning (Limited VRAM)
If you see: `Current model requires 28781064256 bytes of buffer for offloaded layers`

This warning from the `accelerate` library means your GPU doesn't have enough VRAM (~28GB needed) for all model layers. **This is normal and expected for GPUs with <24GB VRAM.**

**Important:** We're already using `offload_buffers=True` in all scripts! The warning appears because `accelerate` checks memory **before** knowing we have this flag enabled.

#### Understanding `offload_buffers=True`

**What it does:**
- **Enables**: Temporary buffers used during computation to be offloaded to CPU RAM
- **Purpose**: Prevents GPU OOM errors when model layers need more VRAM than available
- **How it works**: Moves intermediate computation buffers between GPU↔CPU as needed

**Trade-offs of using `offload_buffers=True`:**

| Aspect | Impact | Explanation |
|--------|--------|-------------|
| **Inference Speed** | Slower | Extra CPU↔GPU transfers for buffers |
| **CPU RAM Usage** | Higher | Buffers need space in system memory |
| **PCIe Bandwidth** | High usage | Constant data movement between GPU and CPU |
| **Latency** | Increased | Each layer computation may wait for buffer transfers |

**Comparison:**
- **Without it**: You'd get OOM errors immediately when the model tries to allocate 28GB of buffers on a 16GB GPU
- **With it**: The model runs successfully but slower - this is the **only way** to run an 80B model on consumer GPUs with <24GB VRAM

**For 16GB VRAM GPUs:**
```bash
python qwen3_80b.py  # Automatically configures offload_buffers=True when needed
```

This automatically distributes the model:
- **GPU**: ~14GB (performance-critical layers)
- **CPU RAM**: ~44GB (remaining layers + buffers)
- **Total**: ~58GB memory usage

**Bottom line:** The warning is misleading - it appears before the library knows we're using `offload_buffers=True`. The model works correctly despite the warning.

### Intel Extension for PyTorch Warning (CPU Mode)
If you see: `Better backend is found, please install all the following requirements`
- **This is just a suggestion**, not an error. The model works fine without it
- **Cannot install on Python 3.13t**: Intel Extension doesn't support free-threaded Python yet
- **Performance impact**: Minimal for this quantized model
- **You can safely ignore this warning**

## Project Status

### ✅ Completed Features
- Basic inference script with hardware detection
- Chain-of-thought support (thinking token parsing)
- Interactive chat mode
- Official HuggingFace implementation
- Model caching and verification tools
- Multiple loading scripts for different use cases
- Automatic swap management for RAM-limited systems
- Hardware performance benchmarking suite
- Consumer hardware optimization (64-128GB RAM)

### 🚧 In Progress
- Performance optimization for faster loading
- Memory usage optimization

### 📋 Planned Features
- OpenAI-compatible API server (FastAPI)
- Performance benchmarking suite
- Docker containerization
- Comprehensive unit tests
- Batch inference support
- Web UI interface

## Note

This is a bridge solution while waiting for llama.cpp integration. Once Qwen3-Next is supported in llama.cpp, this project will  eventually be deprecated.

## Documentation

Additional documentation is available in the `docs/` directory:
- [LOADING_STRATEGIES.md](docs/LOADING_STRATEGIES.md) - Detailed loading strategy matrix
- [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) - Performance optimization guide
- [PYTHON_SETUP.md](docs/PYTHON_SETUP.md) - Python environment setup instructions
- [FIX_WARNINGS.md](docs/FIX_WARNINGS.md) - Troubleshooting common warnings

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Qwen workgroup for the original model](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)
- [Intel for the AutoRound quantized model](https://github.com/intel/auto-round)
- [Hugging Face for hosting the Intel model](https://huggingface.co/Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound)
- [The LocalLLaMA community](https://www.reddit.com/r/LocalLLaMA/) for inspiration and feedback
