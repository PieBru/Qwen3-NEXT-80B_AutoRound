# Qwen3-NEXT-80B AutoRound

Local inference solution for running Qwen3-Next-80B with Intel's 4-bit AutoRound quantization on consumer hardware.

## Overview

This project provides a Python implementation for running the Qwen3-Next 80B parameter model on consumer computers using Intel's AutoRound quantization, reducing memory requirements from 160GB to ~58GB, enabling deployment on systems with 64GB RAM (CPU-only) or GPUs with 16GB+ VRAM.

## TL;DR Quick Start

```bash
# Clone and setup (Arch Linux example)
git clone https://github.com/PieBru/Qwen3-NEXT-80B_AutoRound.git
cd Qwen3-NEXT-80B_AutoRound

# Install Python 3.13.3t (free-threaded) with pyenv
pyenv install 3.13.3t
pyenv local 3.13.3t

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
export UV_LINK_MODE=copy
uv pip install -r requirements.txt --no-build-isolation -U

# Disable GIL for free-threaded Python
export PYTHON_GIL=0

# Authenticate with HuggingFace (required for model download)
hf auth login

# Run the unified inference script
python run_qwen3_80b.py

# See all available options
python run_qwen3_80b.py --help

# Run with CPU-only mode
python run_qwen3_80b.py --cpu

# Run with hybrid GPU/CPU mode
python run_qwen3_80b.py --mode hybrid
```

## Features

- ✅ **4-bit Quantized Inference** - Run 80B models with <64GB memory footprint
- ✅ **Chain-of-Thought Support** - Parse and display the model's thinking process
- ✅ **Interactive Chat Mode** - Real-time conversational interface
- ✅ **Memory Efficient** - Automatic GPU memory management and CPU offloading
- 🚧 **OpenAI-Compatible API** - Coming soon

## Requirements

- Python 3.13+ (free-threaded build recommended, use `export PYTHON_GIL=0`)
- ~64GB free disk space for model (actual size: ~58GB)
- OPTIONAL:
  - NVIDIA GPU with 16GB+ VRAM, CUDA 12.1+

## Usage

### Quick Start

```bash
# Interactive chat mode (auto-detect best configuration)
python run_qwen3_80b.py

# Force CPU-only mode
python run_qwen3_80b.py --cpu

# Force hybrid GPU/CPU mode
python run_qwen3_80b.py --mode hybrid

# Quiet mode with custom settings
python run_qwen3_80b.py --quiet --temperature 0.9

# Get help
python run_qwen3_80b.py --help
```

### Advanced Usage

```bash
# Auto-detect best mode (default)
python run_qwen3_80b.py

# Force CPU-only mode (50GB+ RAM recommended)
python run_qwen3_80b.py --cpu

# Force hybrid GPU/CPU mode
python run_qwen3_80b.py --mode hybrid

# Force GPU-only mode (requires 40GB+ VRAM)
python run_qwen3_80b.py --mode gpu

# Custom memory limits (in GB)
python run_qwen3_80b.py --gpu-memory 12 --cpu-memory 80

# Custom generation parameters
python run_qwen3_80b.py --temperature 0.9 --max-tokens 200

# Verbose mode with detailed information
python run_qwen3_80b.py --verbose

# Quiet mode (minimal output)
python run_qwen3_80b.py --quiet

# Skip test generation and go straight to chat
python run_qwen3_80b.py --skip-test

# Run test only without interactive chat
python run_qwen3_80b.py --no-interactive
```

### Memory Mode Selection

The unified `run_qwen3_80b.py` script automatically selects the best mode based on your hardware:

| Mode | Usage | Requirements |
|------|-------|-------------|
| **auto** (default) | Automatically detects best configuration | Any system |
| **cpu** | CPU-only inference | 50GB+ RAM |
| **gpu** | GPU-only inference | 40GB+ VRAM |
| **hybrid** | Mixed GPU/CPU inference | 16GB+ VRAM, 30GB+ RAM |

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

**⚠️ IMPORTANT: First load is VERY slow!**
- The model is ~58GB and device mapping for 80B parameters takes significant time
- Initial load can take 5-15 minutes depending on hardware
- Subsequent loads are faster due to caching and pre-computed device maps

Expected performance (once loaded):
- **CPU-only**: 0.1-0.5 tokens/second (64GB+ RAM required)
- **RTX 4090**: 2-5 tokens/second (uses 14-20GB VRAM + system RAM)

### Understanding "Loading checkpoint shards"

The "Loading checkpoint shards" message refers to **loading the model from disk into RAM**, not downloading. Each shard is being:

1. **Read from disk** (your local cache at `~/.cache/huggingface/hub/`)
2. **Loaded into RAM** (system memory)
3. **Quantized/dequantized** on-the-fly (4-bit weights need processing)
4. **Mapped to GPU/CPU** based on the device_map

The slow speed (~2.5 minutes per shard) is due to:
- **Single-threaded loading** - The transformers library's safetensors loader only uses 1 CPU thread
- **On-the-fly quantization** - 4-bit AutoRound weights are being processed during loading
- **Memory mapping** - 80B parameters being distributed across GPU and CPU memory
- **Sequential processing** - Each shard must be loaded and processed in order

This is normal for such a large quantized model. The 9 shards totaling 41GB will take ~20-25 minutes to fully load. You can monitor:
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

### "b_q_weight is not on GPU" Error
If you encounter `RuntimeError: b_q_weight is not on GPU`:
- **Cause**: The quantization library expects all model weights on GPU, but your GPU doesn't have enough VRAM
- **Solution 1**: Use CPU-only mode: `python run_qwen3_80b.py --cpu`
- **Solution 2**: Use hybrid mode with proper offloading: `python run_qwen3_80b.py --mode hybrid`
- **Explanation**: The 4-bit quantization post-processing requires either all weights on GPU or CPU, not mixed

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
python run_qwen3_80b.py --mode hybrid  # Automatically configures offload_buffers=True
```

This automatically distributes the model:
- **GPU**: ~14GB (performance-critical layers)
- **CPU RAM**: ~44GB (remaining layers + buffers)
- **Total**: ~58GB memory usage

**Bottom line:** The warning is misleading - it appears before the library knows we're using `offload_buffers=True`. The model works correctly despite the warning.

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

- [Intel for the AutoRound quantized model](https://huggingface.co/Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound)
- Hugging Face for hosting the model
- [The LocalLLaMA community](https://www.reddit.com/r/LocalLLaMA/) for inspiration and feedback
