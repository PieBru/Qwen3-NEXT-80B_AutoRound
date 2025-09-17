#!/usr/bin/env python3
"""
Hybrid MoE Loader for Qwen3-Next-80B
Loads non-expert layers on GPU (FP8) and expert layers on CPU (INT4)
Designed for GPUs with limited VRAM (16GB)
"""

import os
import sys
import time
import torch
import gc
import json
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Environment setup for optimal performance
os.environ["PYTHON_GIL"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def print_memory_status():
    """Print current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU Memory: {gpu_allocated:.1f}/{gpu_reserved:.1f}/{gpu_memory:.1f} GB (allocated/reserved/total)")

    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / (1024**3)
    total_ram = psutil.virtual_memory().total / (1024**3)
    print(f"   RAM Usage: {ram_usage:.1f}/{total_ram:.1f} GB")

def analyze_model_architecture(model_path: Path) -> Dict[str, Any]:
    """Analyze the model architecture to identify expert and non-expert layers"""
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\n📊 Model Architecture Analysis:")
    print(f"   Model Type: {config.get('model_type', 'unknown')}")
    print(f"   Hidden Size: {config.get('hidden_size', 'unknown')}")
    print(f"   Num Layers: {config.get('num_hidden_layers', 'unknown')}")

    # Check for MoE configuration
    moe_config = {
        'num_experts': config.get('num_experts', 60),  # Qwen3 has 60 experts
        'num_experts_per_tok': config.get('num_experts_per_tok', 4),
        'expert_interval': config.get('expert_interval', 1),
        'num_shared_experts': config.get('num_shared_experts', 8)
    }

    print(f"   Number of Experts: {moe_config['num_experts']}")
    print(f"   Experts per Token: {moe_config['num_experts_per_tok']}")
    print(f"   Shared Experts: {moe_config['num_shared_experts']}")

    return config, moe_config

def create_device_map_for_moe(config: Dict, moe_config: Dict, gpu_memory_gb: float) -> Dict[str, Any]:
    """
    Create a custom device map that puts non-expert layers on GPU and experts on CPU

    For Qwen3-Next-80B MoE:
    - Attention layers, embeddings, layer norms → GPU
    - Expert FFN layers → CPU
    - Shared experts → GPU (if space available)
    """
    device_map = {}

    # Estimate memory requirements (rough estimates)
    embedding_size_gb = 2.0  # Embeddings and output layers
    attention_per_layer_gb = 0.3  # Self-attention layers
    expert_size_gb = 0.5  # Each expert FFN
    shared_expert_size_gb = 0.4  # Shared experts are smaller

    num_layers = config.get('num_hidden_layers', 64)
    num_experts = moe_config['num_experts']
    num_shared = moe_config['num_shared_experts']

    # Calculate what fits on GPU
    gpu_budget = gpu_memory_gb * 0.85  # Leave 15% headroom
    current_gpu_usage = 0.0

    # Priority 1: Embeddings and output layers (always on GPU)
    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = 0
    device_map['lm_head'] = 0
    current_gpu_usage += embedding_size_gb

    print(f"\n🎯 Device Mapping Strategy:")
    print(f"   GPU Budget: {gpu_budget:.1f} GB")
    print(f"   Embeddings on GPU: {embedding_size_gb:.1f} GB")

    # Priority 2: Attention layers (on GPU if possible)
    attention_on_gpu = 0
    for layer_idx in range(num_layers):
        layer_key = f'model.layers.{layer_idx}'

        # Attention components
        if current_gpu_usage + attention_per_layer_gb < gpu_budget:
            device_map[f'{layer_key}.self_attn'] = 0
            device_map[f'{layer_key}.input_layernorm'] = 0
            device_map[f'{layer_key}.post_attention_layernorm'] = 0
            current_gpu_usage += attention_per_layer_gb
            attention_on_gpu += 1
        else:
            device_map[f'{layer_key}.self_attn'] = 'cpu'
            device_map[f'{layer_key}.input_layernorm'] = 'cpu'
            device_map[f'{layer_key}.post_attention_layernorm'] = 'cpu'

    print(f"   Attention layers on GPU: {attention_on_gpu}/{num_layers}")
    print(f"   Current GPU usage: {current_gpu_usage:.1f} GB")

    # Priority 3: Shared experts (on GPU if space available)
    shared_on_gpu = 0
    if current_gpu_usage + (num_shared * shared_expert_size_gb) < gpu_budget:
        for layer_idx in range(num_layers):
            layer_key = f'model.layers.{layer_idx}'
            if hasattr(config, 'shared_expert'):
                device_map[f'{layer_key}.shared_expert'] = 0
                shared_on_gpu += 1
        current_gpu_usage += (num_shared * shared_expert_size_gb)
        print(f"   Shared experts on GPU: {shared_on_gpu}")

    # Priority 4: Regular experts (always on CPU for memory efficiency)
    experts_on_cpu = 0
    for layer_idx in range(num_layers):
        layer_key = f'model.layers.{layer_idx}'
        # Map all expert layers to CPU
        for expert_idx in range(num_experts):
            device_map[f'{layer_key}.mlp.experts.{expert_idx}'] = 'cpu'
            experts_on_cpu += 1
        # Router stays on GPU for efficiency
        device_map[f'{layer_key}.mlp.router'] = 0

    print(f"   Expert FFNs on CPU: {experts_on_cpu}")
    print(f"   Routers on GPU: {num_layers}")
    print(f"   Final GPU usage estimate: {current_gpu_usage:.1f} GB")

    return device_map

def load_model_hybrid(model_name: str, device_map: Optional[Dict] = None):
    """Load model with hybrid GPU/CPU placement"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

    print("\n⏳ Loading model with hybrid GPU/CPU placement...")
    print("   This will take 20-30 minutes on first load")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Memory settings for hybrid loading
    max_memory = {
        0: "14GiB",  # GPU
        "cpu": "90GiB"  # CPU RAM
    }

    try:
        # Load with custom device map if provided
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,  # Use FP16 for GPU parts
            "low_cpu_mem_usage": True,
            "max_memory": max_memory,
        }

        if device_map:
            load_kwargs["device_map"] = device_map
        else:
            # Let accelerate figure it out with constraints
            load_kwargs["device_map"] = "auto"

        # For INT4 quantized model
        load_kwargs["load_in_4bit"] = False  # Don't use bitsandbytes

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        print("\n✅ Model loaded successfully!")

        # Print actual device distribution
        if hasattr(model, 'hf_device_map'):
            print("\n📍 Actual device distribution:")
            devices_count = {"cuda:0": 0, "cpu": 0}
            for name, device in model.hf_device_map.items():
                if "expert" in name.lower():
                    devices_count["cpu"] += 1
                else:
                    dev = str(device)
                    if dev in devices_count:
                        devices_count[dev] += 1

            print(f"   GPU layers: {devices_count.get('cuda:0', 0) + devices_count.get('0', 0)}")
            print(f"   CPU layers: {devices_count.get('cpu', 0)}")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")

        # Fallback: Try with even more aggressive offloading
        print("\n🔄 Retrying with more aggressive CPU offloading...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "10GiB", "cpu": "100GiB"},
            offload_buffers=True
        )

        return model, tokenizer

def main():
    print("=" * 60)
    print("Qwen3-Next-80B Hybrid MoE Loader")
    print("Non-experts on GPU | Experts on CPU")
    print("=" * 60)

    # Check cache
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"
    model_cache_name = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{model_cache_name}"

    if not model_path.exists():
        print("\n❌ Model not cached. Please download first:")
        print("   python qwen3_80b.py --check")
        return 1

    snapshots = model_path / "snapshots"
    snapshot_dirs = list(snapshots.iterdir())
    if not snapshot_dirs:
        print("❌ No snapshot found")
        return 1

    snapshot_path = snapshot_dirs[0]
    print(f"\n📂 Model path: {snapshot_path}")

    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ No GPU detected. Hybrid mode requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n🎮 GPU: {gpu_name}")
    print(f"   VRAM: {gpu_memory:.1f} GB")

    # Analyze model architecture
    try:
        config, moe_config = analyze_model_architecture(snapshot_path)
    except Exception as e:
        print(f"\n❌ Failed to analyze model: {e}")
        return 1

    # Create custom device map
    print("\n🗺️ Creating hybrid device map...")
    device_map = create_device_map_for_moe(config, moe_config, gpu_memory)

    print("\n" + "=" * 60)
    print("Loading Strategy Summary:")
    print("- Non-expert layers (attention, embeddings) → GPU (FP16)")
    print("- Expert FFN layers → CPU (INT4)")
    print("- Shared experts → GPU if space available")
    print("- Routers → GPU for fast routing")
    print("=" * 60)

    print_memory_status()

    # Load model
    try:
        model, tokenizer = load_model_hybrid(str(snapshot_path), device_map)
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\n💡 Try reducing GPU memory limit or using standard loading")
        return 1

    print_memory_status()

    # Test generation
    print("\n🧪 Testing generation...")
    test_prompt = "What is 2+2?"
    print(f"Prompt: {test_prompt}")

    inputs = tokenizer(test_prompt, return_tensors="pt")
    # Move inputs to GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_time = time.time() - start_time

    print(f"Response: {response}")
    print(f"Generation time: {gen_time:.2f}s")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("Note: Inference will be slower due to CPU↔GPU transfers")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(user_input):].strip()

            print(f"\nResponse: {response}")
            print(f"Time: {time.time() - start_time:.2f}s")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())