# Legacy Scripts

This folder contains the original individual scripts that have been consolidated into the unified `qwen3_80b.py` CLI tool.

## Archived Scripts

- `run_official.py` - Original official HuggingFace implementation
- `qwen3_thinking.py` - Thinking tokens visualization
- `run_qwen3_80b.py` - Basic inference script
- `run_qwen3_80b_realistic.py` - Realistic expectations version
- `run_qwen3_80b_optimized.py` - Memory optimized version
- `run_qwen3_80b_multicore.py` - Multi-threading attempts
- `run_qwen3_80b_parallel_load.py` - Parallel loading experiments
- `run_qwen3_80b_local.py` - Local cache loading
- `run_optimized.py` - VRAM optimization
- `run_simple.py` - Simplified loader
- `check_model.py` - Cache checker
- `test_deps.py` - Dependency tester
- `diagnose_loading.py` - Loading diagnostics

## Migration Guide

All functionality has been merged into the main `qwen3_80b.py` script:

| Old Script | New Command |
|------------|-------------|
| `python run_official.py` | `python qwen3_80b.py` |
| `python qwen3_thinking.py` | `python qwen3_80b.py --thinking` |
| `python run_optimized.py` | `python qwen3_80b.py` (auto-optimizes) |
| `python check_model.py` | `python qwen3_80b.py --check` |
| `python test_deps.py` | `python qwen3_80b.py --test-deps` |

## Note

These scripts are kept for reference but are no longer maintained. Please use the unified `qwen3_80b.py` script for all functionality.