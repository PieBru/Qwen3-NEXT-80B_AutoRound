# Qwen3-NEXT-80B AutoRound

EDUCATIONAL and EXPERIMENTAL while waiting for `llama.cpp` integration: run Qwen3-Next-80B FP8+FP4 model on consumer hardware using Intel's 4-bit AutoRound quantization, CPU, IPEX, GPU, Flash Linear Attention, GPTQ, and more.

## TL;DR Quick Start

```bash
# Clone and setup
git clone https://github.com/PieBru/Qwen3-NEXT-80B_AutoRound.git
cd Qwen3-NEXT-80B_AutoRound

# FIXME Option A: Python 3.11/3.12 (RECOMMENDED - 2-4x faster CPU inference)
./setup_python311_ipex.sh  # Complete setup with IPEX

# FIXME (show more options) Option B: Python 3.13+ (if already installed)
pyenv install 3.13.3t
pyenv local 3.13.3t
uv venv
source .venv/bin/activate

# Install dependencies (auto-detects Python version)
python install_requirements.py

# For Python 3.13 only - disable GIL
export PYTHON_GIL=0

# Authenticate with HuggingFace (required for model download)
hf auth login

# See all available options
python qwen3_80b.py --help

# Verify your environment is properly configured:
python qwen3_80b.py --test-deps

# Run the CLI (v3.0 - automatic hybrid mode for <30GB VRAM)
python qwen3_80b.py

# For multi-GPU systems (enable GPTQModel optimization)
python qwen3_80b.py --use-gptq

# To see the model's thinking process:
python qwen3_80b.py --thinking
```

## Features

- ✅ **Fast Caching System** (v3.4) - **Enabled by default!** Load model in <1 minute after first run (vs 30-60 minutes)
- ✅ **Smart Loading Strategy** (v3.0) - Auto-detects resources and picks optimal loading strategy, no more double loading!
- ✅ **4-bit Quantized Inference** - Run 80B models with <64GB memory footprint
- ✅ **Chain-of-Thought Support** - Parse and optionally display the model's thinking process
- ✅ **Interactive Chat Mode** - Conversational CLI (WebUI coming soon)
- ✅ **Memory Efficient** - Automatic and manual GPU memory management and CPU offloading
- 🚧 **OpenAI-Compatible API** - Coming soon

## Requirements

### Hardware Requirements
- ~64GB free disk space for model (actual size: ~58GB)
- **For CPU inference**: 50GB+ RAM
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

| Python Version | IPEX Support | CPU Performance | Recommendation   |
|----------------|--------------|-----------------|------------------|
| 3.11-3.12      | ✅ Yes       | 2-4x faster     | **Best choice**  |
| 3.13+          | ❌ No        | Baseline        | Works but slower |

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

See [LOADING_STRATEGIES.md](LOADING_STRATEGIES.md) for complete resource matrix.

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

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Qwen workgroup for the original model](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)
- [Intel for the AutoRound quantized model](https://github.com/intel/auto-round)
- [Hugging Face for hosting the Intel model](https://huggingface.co/Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound)
- [The LocalLLaMA community](https://www.reddit.com/r/LocalLLaMA/) for inspiration and feedback
