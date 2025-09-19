# Qwen3-80B AutoRound Optimizations Guide

## Overview

This guide documents the performance optimizations available for the Intel AutoRound quantized Qwen3-80B model.

## Key AutoRound Features We Leverage

### 1. Mixed-Bit Quantization
The model uses sophisticated layer-wise mixed-bit quantization:
- **4-bit**: Most layers (default)
- **8-bit**: Critical attention and projection layers
- **16-bit**: Shared expert gate layers
- **Group size**: 128 (optimal for performance/quality balance)

This intelligent bit allocation maintains model quality while reducing memory usage by ~75%.

### 2. Symmetric Quantization
- Uses symmetric quantization (`sym: True`) for faster inference
- No zero-point calculations needed during inference
- Better hardware acceleration support

### 3. AutoRound Format
- Packing format: `auto_round:auto_gptq`
- Optimized for both CPU and GPU inference
- Automatic backend selection based on hardware

## Available Optimizations

### Environment Optimizations (Already Implemented)

```bash
# Thread optimization
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

# GPU optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tokenizer optimization
export TOKENIZERS_PARALLELISM=true

# GIL-free Python (3.13t)
export PYTHON_GIL=0
```

### Loading Strategies

1. **CPU-only Mode** (`--load-strategy no-gpu`)
   - Completely disables GPU
   - Most reliable, works on any system
   - 25-30 minute load time
   - Best for systems with 50GB+ RAM

2. **Minimal GPU** (`--load-strategy min-gpu`) [DEFAULT]
   - Loads non-expert layers on GPU (~12GB)
   - Expert layers on CPU (~28GB)
   - Best balance of speed and memory usage
   - Recommended for RTX 4090

3. **Maximum GPU** (`--load-strategy max-gpu`)
   - Uses all available VRAM
   - Fastest inference
   - Requires careful memory management

### Scripts Available

1. **Main Script**: `qwen3_80b.py`
   - Full-featured with all loading strategies
   - Interactive and benchmark modes
   - Thinking mode support (`--thinking`)
   - Built-in performance benchmarking (`--perf-test`)

2. **Performance Scripts**: `scripts/`
   - `benchmark_hardware.py` - Comprehensive hardware benchmark
   - `optimize_performance.py` - System optimization recommendations
   - `diagnose_loading_speed.py` - Diagnose slow loading issues

## Performance Tips

### For Fastest Loading
```bash
# Use cached model with minimal GPU
python qwen3_80b.py --load-strategy min-gpu --quiet

# Skip all warnings and checks
python qwen3_80b.py --no-warnings --quiet
```

### For Best Inference Speed
```bash
# Maximum GPU utilization
python qwen3_80b.py --load-strategy max-gpu --gpu-memory 14

# Run benchmarks to test
python qwen3_80b.py --perf-test
```

### For Reliability
```bash
# Pure CPU mode - always works
python qwen3_80b.py --load-strategy no-gpu --interactive
```

## System Requirements

### Minimum Requirements
- **RAM**: 50GB available (CPU-only mode)
- **Storage**: 40GB for model cache
- **CPU**: Any x86_64 processor

### Recommended Setup
- **RAM**: 64GB+ total
- **GPU**: RTX 4090 or better (12GB+ VRAM)
- **Storage**: SSD for model cache and offloading
- **OS**: Linux with recent kernel

## Detected Optimizations on Your System

✅ **Available:**
- Intel MKL (matrix operations acceleration)
- AutoRound 0.7.0 (quantization support)
- PyTorch 2.8.0 with CUDA 12.8
- torch.compile support (JIT compilation)
- GIL-free Python 3.13t (true parallelism)
- 32 CPU cores
- RTX 4090 Laptop GPU (15.6GB VRAM)

❌ **Not Available:**
- Intel Extension for PyTorch (incompatible with Python 3.13t)
- Note: IPEX would provide 2-4x CPU speedup if using Python 3.9-3.12

## Troubleshooting

### Slow Loading
- Normal: Model loading takes 20-30 minutes (single-threaded limitation)
- Monitor with `htop` - only 1 CPU core will be at 100%
- Use `--quiet` to reduce output overhead

### Out of Memory
- Try `--load-strategy no-gpu` for CPU-only mode
- Reduce `--gpu-memory` parameter
- Close other applications
- Enable swap space

### CUDA Errors
- Use `--load-strategy no-gpu` to bypass GPU entirely
- Update NVIDIA drivers
- Check CUDA version compatibility

## Future Optimizations

When Python compatibility improves:
- Intel Extension for PyTorch (2-4x CPU speedup)
- Intel Neural Compressor integration
- OpenVINO backend support
- ONNX Runtime optimization

## Conclusion

The AutoRound quantized model already provides excellent compression (4x smaller) while maintaining quality. The mixed-bit quantization strategy ensures critical layers maintain higher precision. Combined with the available system optimizations, this provides a good balance of performance and quality for the 80B parameter model.