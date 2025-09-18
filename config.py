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
    "model_quantized_size": 29,        # 4-bit quantized model on disk
    "model_loaded_ram": 40,            # Model loaded in RAM (with overhead)

    # Loading strategies
    "gpu_minimal": 10,                 # Minimum GPU memory for hybrid mode
    "gpu_optimal": 14,                 # Optimal GPU memory (leaves headroom)
    "gpu_full": 30,                    # Full GPU loading requirement

    # CPU requirements
    "cpu_minimum_ram": 40,             # Minimum RAM for CPU-only mode
    "cpu_recommended_ram": 50,         # Recommended RAM for comfortable operation

    # IPEX optimization
    "ipex_peak_memory": 230,           # Peak memory during IPEX optimization
    "ipex_additional": 115,            # Additional memory needed during IPEX
    "ipex_final_memory": 40,           # Memory after IPEX optimization

    # System recommendations
    "swap_recommended": 256,           # Recommended swap for IPEX
    "inference_overhead": 5,           # Additional RAM for inference
}

# Cache Configuration
CACHE_CONFIG = {
    "cache_dir": Path.home() / ".cache/qwen3_fast_loader",
    "cache_version": "v3.0",
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
        if available_ram >= MEMORY_REQUIREMENTS["ipex_peak_memory"]:
            result["suggestions"].append("Consider using IPEX for CPU optimization")
        else:
            result["warnings"].append("Limited RAM may cause slow performance")
            result["suggestions"].append(f"Increase swap to {MEMORY_REQUIREMENTS['swap_recommended']}GB for better performance")

    return result

# Export key constants
__all__ = [
    'MODEL_NAME',
    'MEMORY_REQUIREMENTS',
    'CACHE_CONFIG',
    'PERFORMANCE',
    'SYSTEM_THRESHOLDS',
    'FILE_SAFETY',
    'get_memory_requirement',
    'get_cache_path',
    'validate_system_memory'
]