# Qwen3-Next-80B Loading Strategies Matrix

## Model Specifications
- **Model Size**: ~40GB (INT4 quantized)
- **Shards**: 9 files × ~4.5GB each
- **Original Size**: 160GB (FP16)
- **Architecture**: MoE with 512 experts, 10 active per token

## Resource Requirements by Loading Type

### 1. FULL GPU LOADING
**Requirements**: 30GB+ VRAM
**Performance**: ⚡⚡⚡⚡⚡ (Fastest)
**Load Time**: 30-60 minutes (first time)

| VRAM | RAM | Strategy | Notes |
|------|-----|----------|-------|
| 30-32GB | 16GB+ | Full GPU | All shards load to GPU. Minimum viable (tight). |
| 40GB | 16GB+ | Full GPU | Comfortable fit with headroom. |
| 48GB | 16GB+ | Full GPU | Professional GPUs (A6000, RTX 6000). |
| 80GB | 16GB+ | Full GPU | Data center GPUs (A100, H100). |

**Devices**: RTX 4090 Ti (32GB), RTX 6000 Ada (48GB), A100 (40/80GB)

### 2. HYBRID GPU/CPU LOADING
**Requirements**: 14-29GB VRAM + 60GB+ RAM
**Performance**: ⚡⚡⚡ (Moderate)
**Load Time**: 30-60 minutes + overhead

| VRAM | RAM | Strategy | Distribution |
|------|-----|----------|--------------|
| 14-16GB | 60-80GB | Hybrid-Heavy-CPU | 2-3 shards GPU, 6-7 shards CPU |
| 20-24GB | 50-70GB | Hybrid-Balanced | 4-5 shards GPU, 4-5 shards CPU |
| 24-29GB | 40-60GB | Hybrid-Heavy-GPU | 6-7 shards GPU, 2-3 shards CPU |

**MoE Distribution**:
- **GPU**: Embeddings, attention layers, routers (~8-12GB)
- **CPU**: Expert FFN layers (~28-32GB)
- **Shared**: KV cache dynamically allocated

**Devices**: RTX 4070 Ti (16GB), RTX 4080 (16GB), RTX 4090 (24GB)

### 3. CPU-ONLY LOADING
**Requirements**: 50GB+ RAM
**Performance**: ⚡ (Slowest)
**Load Time**: 30-60 minutes

| VRAM | RAM | Strategy | Feasibility |
|------|-----|----------|-------------|
| 0GB | 50-64GB | CPU-Tight | Model loads, very slow inference, high swap risk |
| 0GB | 80-96GB | CPU-Standard | Reasonable performance, no swapping |
| 0GB | 128GB+ | CPU-Comfort | Good CPU performance, space for OS/apps |

**Notes**:
- Requires FP32 for stability (doubles memory vs FP16)
- Intel MKL/OpenBLAS optimization helps
- Consider GGUF format for better CPU performance

### 4. STREAMING/OFFLOADED LOADING
**Requirements**: 8-13GB VRAM + 32GB+ RAM
**Performance**: ⚡⚡ (Slow but functional)
**Load Time**: 30-60 minutes + constant loading

| VRAM | RAM | Strategy | Mechanism |
|------|-----|----------|-----------|
| 8-12GB | 32-48GB | Stream-Aggressive | Load 1-2 shards at a time, constant swapping |
| 8-12GB | 48-64GB | Stream-Cached | Keep recent shards in RAM cache |
| 12-16GB | 32-48GB | Offload-Buffers | Computation buffers to RAM, weights on GPU |

**Techniques**:
- Load shards on-demand
- LRU cache for recently used shards
- Memory-mapped files for fast swapping
- Pinned memory for faster CPU↔GPU transfer

### 5. DISK-BACKED LOADING (Experimental)
**Requirements**: 4GB+ VRAM/RAM + Fast NVMe
**Performance**: ⚡ (Very slow)
**Load Time**: Continuous

| Storage | RAM | Strategy | Use Case |
|---------|-----|----------|----------|
| NVMe | 16-32GB | mmap-based | Research/testing only |
| SSD | 32-48GB | Cached-mmap | Development work |
| RAID 0 NVMe | 16GB | Striped-mmap | Experimental inference |

## Optimization Strategies by Resource Level

### Tier 1: Premium (30GB+ VRAM)
```python
# Full GPU loading
device_map = "auto"
max_memory = {0: "30GiB"}
torch_dtype = torch.float16
```

### Tier 2: High-End Consumer (16-24GB VRAM, 64GB+ RAM)
```python
# Hybrid with CPU offloading
device_map = "auto"
max_memory = {0: "14GiB", "cpu": "50GiB"}
torch_dtype = torch.float16
offload_buffers = True
```

### Tier 3: Mid-Range (8-16GB VRAM, 32-64GB RAM)
```python
# CPU-primary with GPU assist
device_map = "cpu"
torch_dtype = torch.float32
# Consider GGUF format instead
```

### Tier 4: Low-End (No GPU, 64GB+ RAM)
```python
# Pure CPU mode
device_map = "cpu"
torch_dtype = torch.float32
low_cpu_mem_usage = True
```

## Loading Time Optimizations

### First Load (No Cache)
1. **Sequential Loading**: 30-60 minutes
2. **Parallel Shard Loading**: Not supported (model limitation)
3. **Memory-mapped Loading**: 25-45 minutes (experimental)

### Subsequent Loads
1. **OS File Cache**: ~15-20 minutes (if files in RAM cache)
2. **Saved State Dict**: ~5-10 minutes (pickle/torch.save)
3. **Memory-mapped Cache**: ~2-5 minutes (pre-allocated)
4. **Persistent Process**: 0 minutes (keep model in memory)

## Recommended Configurations

### For 16GB VRAM + 64GB RAM (Common Setup)
```bash
# Option 1: CPU-only (stable)
./qwen3_80b.py --cpu

# Option 2: Hybrid attempt (may fail)
./qwen3_80b.py --gpu-memory 14 --cpu-memory 50

# Option 3: Use GGUF format
# Convert to GGUF and use llama.cpp
```

### For 24GB VRAM + 128GB RAM (Enthusiast)
```bash
# Hybrid loading with good performance
./qwen3_80b.py --gpu-memory 20 --cpu-memory 80
```

### For 48GB+ VRAM (Professional)
```bash
# Full GPU loading
./qwen3_80b.py --use-gptq --gpu-memory 40
```

## Key Insights

1. **INT4 Quantization Issues**: The AutoRound INT4 quantization has issues with mixed CPU/GPU placement due to custom CUDA kernels

2. **MoE Advantage**: The 512 experts can theoretically be distributed, but current implementation doesn't support it well

3. **Memory Overhead**: Actual memory usage is ~1.5x model size due to:
   - Attention caches
   - Gradient buffers (even in inference)
   - Framework overhead

4. **Speed Trade-offs**:
   - Full GPU: 50-100 tokens/sec
   - Hybrid: 5-20 tokens/sec
   - CPU: 0.5-5 tokens/sec

5. **Alternative Formats**:
   - **GGUF**: Better CPU performance, supports mixed precision
   - **ONNX**: Good compatibility, lacks MoE optimizations
   - **TensorRT**: Excellent GPU performance, NVIDIA only

## Future Optimizations

1. **Shard-level Caching**: Cache individual shards after loading
2. **Dynamic Loading**: Load only active experts per token
3. **Quantization Flexibility**: Mix INT4/INT8/FP16 per layer
4. **Pipeline Parallelism**: Stream tokens through layers
5. **Compilation**: torch.compile() for specific hardware

## Decision Tree

```
Start → Check VRAM
├── ≥30GB → Full GPU Loading ✅
├── 16-29GB → Check RAM
│   ├── ≥64GB → Try Hybrid (may fail) ⚠️
│   └── <64GB → CPU-only or GGUF 🔄
├── 8-15GB → Check RAM
│   ├── ≥64GB → CPU-only ✅
│   └── <64GB → Consider GGUF or smaller model ❌
└── <8GB → Need GGUF or cloud solution ❌
```

## Troubleshooting

### "b_q_weight is not on GPU"
- **Cause**: Mixed placement not supported
- **Solution**: Force CPU-only mode

### Double Shard Loading
- **Cause**: Failed GPU attempt + CPU retry
- **Solution**: Detect resources upfront, skip GPU attempt

### OOM During Loading
- **Cause**: Insufficient RAM for buffers
- **Solution**: Increase swap or reduce max_memory limits

### Slow Token Generation
- **Cause**: Excessive CPU↔GPU transfers
- **Solution**: Adjust device_map or go full CPU/GPU