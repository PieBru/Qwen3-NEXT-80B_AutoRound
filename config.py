#!/usr/bin/env python3
"""
Central configuration for Qwen3-80B AutoRound project
All memory requirements and system parameters in one place
"""

from pathlib import Path
from typing import Dict, Any

# Model Information
MODEL_NAME = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"
MODEL_SHORT_NAME = "Qwen3-80B"

# Memory Requirements (in GB)
MEMORY_REQUIREMENTS = {
    # Base model sizes
    "model_quantized_size": 41,        # 4-bit quantized model on disk
    "model_loaded_ram": 45,            # Model loaded in RAM (with overhead)

    # Cache sizes
    "cache_size_pickle": 80,           # Old pickle cache size (deprecated)
    "cache_size_torch": 40,            # New torch.save cache size
    "cache_raw_checkpoint": 40,        # Raw checkpoint cache before IPEX

    # Loading strategies
    "gpu_minimal": 10,                 # Minimum GPU memory for hybrid mode
    "gpu_optimal": 14,                 # Optimal GPU memory (leaves headroom)
    "gpu_full": 30,                    # Full GPU loading requirement

    # CPU requirements
    "cpu_minimum_ram": 45,             # Minimum RAM to run cached model
    "cpu_recommended_ram": 128,        # Recommended RAM for comfortable operation

    # IPEX optimization requirements
    "ipex_total_required": 160,        # Total memory needed during IPEX optimization
    "ipex_additional": 115,            # Additional memory beyond model size
    "ipex_swap_recommended": 200,      # Recommended total (RAM+swap) for IPEX

    # Running requirements after cache
    "inference_ram_cached": 45,        # RAM needed to run from cache
    "inference_overhead": 5,           # Additional RAM for inference
}

# Cache Configuration
CACHE_CONFIG = {
    "cache_dir": Path.home() / ".cache/qwen3_fast_loader",
    "cache_version": "v4.0",  # v4.0: torch.save with memory mapping
    "cache_timeout": 3600,  # 1 hour cache validity for temporary files
    "max_cache_size_gb": 100,  # Maximum cache size before cleanup
}

# Performance Settings
PERFORMANCE = {
    "default_max_tokens": 512,
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "batch_size": 1,
    "num_threads": None,  # Auto-detect
    "torch_compile": False,  # Experimental
    "loading_thread_ratio": 0.75,  # Use 75% of threads for loading
    "fast_load_threshold_seconds": 120,  # Consider load fast if under 2 minutes
    "slow_load_threshold_seconds": 600,  # Warn if load takes over 10 minutes
    "baseline_load_time_seconds": 2700,  # 45 minutes average load from scratch
}

# System Detection Thresholds
SYSTEM_THRESHOLDS = {
    "high_end_ram": 128,     # GB - System considered high-end
    "mid_range_ram": 64,     # GB - System considered mid-range
    "low_end_ram": 32,       # GB - System considered low-end
    "gpu_available_threshold": 8,  # GB - Minimum GPU memory to consider using
}

# Thread Configuration
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_NUM_THREADS",
    "OPENBLAS_NUM_THREADS"
]

# File Safety Settings
FILE_SAFETY = {
    "use_atomic_writes": True,  # Use atomic file operations
    "temp_suffix": ".tmp",      # Temporary file suffix
    "backup_suffix": ".bak",    # Backup file suffix
    "validate_paths": True,     # Validate all file paths
}

def get_memory_requirement(requirement_type: str) -> float:
    """Get memory requirement by type with fallback"""
    return MEMORY_REQUIREMENTS.get(requirement_type, 40)  # Default 40GB

def get_cache_path(model_name: str = MODEL_NAME) -> Path:
    """Get cache path for a specific model"""
    safe_name = model_name.replace("/", "_")
    return CACHE_CONFIG["cache_dir"] / safe_name

def validate_system_memory(available_ram: float, available_gpu: float = 0) -> Dict[str, Any]:
    """Validate if system has enough memory and suggest strategy"""
    result = {
        "can_run": False,
        "recommended_strategy": None,
        "warnings": [],
        "suggestions": []
    }

    total_available = available_ram + available_gpu

    # Check minimum requirements
    if total_available < MEMORY_REQUIREMENTS["cpu_minimum_ram"]:
        result["warnings"].append(f"Insufficient memory: {total_available:.1f}GB available, need {MEMORY_REQUIREMENTS['cpu_minimum_ram']}GB minimum")
        result["suggestions"].append("Add more RAM or increase swap space")
        return result

    result["can_run"] = True

    # Recommend strategy based on available resources
    if available_gpu >= MEMORY_REQUIREMENTS["gpu_full"]:
        result["recommended_strategy"] = "max-gpu"
        result["suggestions"].append("Full GPU loading available - fastest performance")
    elif available_gpu >= MEMORY_REQUIREMENTS["gpu_minimal"]:
        result["recommended_strategy"] = "min-gpu"
        result["suggestions"].append("Hybrid GPU+CPU loading recommended")
    else:
        result["recommended_strategy"] = "no-gpu"
        if available_ram >= MEMORY_REQUIREMENTS["ipex_total_required"]:
            result["suggestions"].append("Consider using IPEX for CPU optimization")
        else:
            result["warnings"].append("Limited RAM may cause slow performance")
            result["suggestions"].append(f"Increase swap to {MEMORY_REQUIREMENTS['ipex_swap_recommended']}GB for better performance")

    return result

# Performance benchmark configuration
BENCHMARK_CONFIG = {
    # Memory benchmark
    "memory": {
        "test_sizes_mb": [1, 4, 8],  # Small (cache)
        "test_sizes_mb_medium": [32, 64, 128],  # Medium (L3/RAM transition)
        "test_sizes_mb_large": [256, 512, 1024],  # Large (pure RAM)
        "test_sizes_mb_xlarge": [2048, 4096],  # Extra large
        "iterations": 3,
        "min_system_memory_gb": {
            "medium": 8,
            "large": 16,
            "xlarge": 32,
        }
    },
    # CPU benchmark
    "cpu": {
        "matrix_size": 1000,  # Matrix multiplication size
        "matrix_size_multithread": 300,  # Smaller for multithread test
        "matrix_iterations": 3,
        "prime_limit": 100000,  # Prime calculation upper limit
        "prime_range_multithread": (10000, 15000),  # Range for multithread test
        "hash_size_mb": 100,
        "hash_iterations": 3,
        "max_test_threads": 4,  # Limit threads to avoid timeout
        "multithread_timeout": 5,  # Seconds
    },
    # Storage benchmark
    "storage": {
        "sequential_size_mb": 512,  # Sequential read test size
        "sequential_block_kb": 4096,
        "random_file_size_mb": 50,
        "random_read_count": 500,
        "random_block_kb": 4,
        "shard_size_mb": 450,  # Model shard simulation (reduced from 4.5GB)
        "num_shards": 3,
        "min_free_space_gb": 10,
    },
    # Performance scoring thresholds
    "scoring": {
        "ram_bandwidth_excellent": 50,  # GB/s
        "ram_bandwidth_good": 30,
        "ram_bandwidth_moderate": 20,
        "cpu_gflops_excellent": 100,
        "cpu_gflops_good": 50,
        "cpu_gflops_moderate": 20,
        "storage_mbps_excellent": 1000,
        "storage_mbps_good": 500,
        "storage_mbps_moderate": 100,
    }
}

# Export key constants
__all__ = [
    'MODEL_NAME',
    'MEMORY_REQUIREMENTS',
    'CACHE_CONFIG',
    'PERFORMANCE',
    'SYSTEM_THRESHOLDS',
    'FILE_SAFETY',
    'BENCHMARK_CONFIG',
    'get_memory_requirement',
    'get_cache_path',
    'validate_system_memory'
]
