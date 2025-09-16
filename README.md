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

# Authenticate and run
hf auth login
python run_qwen3_80b.py

# To see the model's thinking process:
python qwen3_thinking.py    # Chain-of-Thought Mode

# Verify your environment is properly configured:
python test_deps.py
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

### Available Scripts

1. **`run_official.py`** - Official HuggingFace implementation with thinking tokens
2. **`qwen3_thinking.py`** - Alternative implementation with thinking visualization
3. **`run_qwen3_80b.py`** - Basic inference with hardware detection
4. **`run_optimized.py`** - Optimized for limited VRAM (16GB) + 64GB+ RAM
5. **`run_simple.py`** - Simplified loader with better progress feedback
6. **`check_model.py`** - Verify model cache status
7. **`test_deps.py`** - Check dependencies

### Running the Model

```bash
# Official implementation (recommended)
python run_official.py

# See the model's thinking process
python qwen3_thinking.py

# Basic interactive chat
python run_qwen3_80b.py
```

The model uses special token 151668 (`</think>`) to separate internal reasoning from responses.

## Model Details

- **Model**: Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound
- **Size**: ~58GB (4-bit quantized with INT4/FP8 mixed precision)
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

## Troubleshooting

### Model Loading Issues
If the model appears stuck during loading:
1. **BE PATIENT** - The 80B parameter model device mapping is extremely slow on first run
2. Check system resources: `htop` (CPU/RAM) or `nvidia-smi` (GPU)
3. Try the simplified loader: `python run_simple.py` (better progress feedback)
4. Verify model is cached: `python check_model.py`
5. For CPU-only mode, ensure you have 64GB+ RAM available
6. Use `export PYTHON_GIL=0` to avoid RuntimeWarnings with free-threaded Python

### GPU Memory Warning (Limited VRAM)
If you see: `Current model requires 28781064256 bytes of buffer for offloaded layers`

This means your GPU doesn't have enough VRAM (~28GB needed) for all model layers. **This is normal for GPUs with <24GB VRAM.**

**Solution for RTX 4090 laptop (16GB VRAM) + high RAM:**
```bash
python run_optimized.py  # Optimized for 16GB VRAM + 64GB+ RAM
```

This script uses CPU+GPU offloading to distribute the model:
- **GPU**: ~12GB (fast layers)
- **CPU RAM**: ~46GB (remaining layers)
- **Total**: <64GB memory usage

This prevents OOM errors while maintaining reasonable performance.

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

- Intel for the AutoRound quantization technology
- Hugging Face for hosting the model
- The localllama community for inspiration and feedback
