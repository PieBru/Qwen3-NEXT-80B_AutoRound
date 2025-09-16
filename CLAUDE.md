# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen3-Next-80B model implementation using Intel's AutoRound 4-bit quantization. The project focuses on running large language models efficiently with reduced memory footprint while maintaining the model's "thinking" capabilities.

## Environment Setup

### Python Version
This project requires Python 3.13.3t (free-threaded build) for GIL-free performance. Managed with pyenv.

### Virtual Environment Commands
```bash
# Create and activate environment
uv venv
source .venv/bin/activate

# Set Python version
pyenv local 3.13.3t
```

### Installation
```bash
# Install all dependencies
uv pip install -r requirements.txt --no-build-isolation

# Authenticate with HuggingFace (required for model downloads)
hf auth login
```

### Critical Dependency Note
The model requires the latest development version of Transformers to support the `qwen3_next` architecture:
```bash
uv pip install --upgrade git+https://github.com/huggingface/transformers.git
```

## Key Scripts

### `run_qwen3_80b.py`
Main inference script with interactive mode. Features:
- Auto-detects CUDA availability and manages GPU memory
- Includes both test prompt and interactive chat mode
- Memory-efficient loading with 4-bit quantization
- Configures max GPU memory usage (14GiB default)

### `qwen3_thinking.py`
Specialized script for the "thinking" variant that:
- Separates internal thinking process from final response
- Parses special tokens (151668 for `</think>`)
- Shows both reasoning and answer in interactive mode

### `test_deps.py`
Dependency verification script - run this first to ensure environment is properly configured.

## Common Development Tasks

```bash
# Test dependencies
python test_deps.py

# Run the main model (interactive mode)
python run_qwen3_80b.py

# Run thinking variant (shows reasoning process)
python qwen3_thinking.py

# Quick memory check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else '')"
```

## Model Architecture Details

The model is Intel's AutoRound quantized version of Qwen3-Next-80B:
- **Model ID**: `Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound`
- **Quantization**: 4-bit mixed precision using AutoRound
- **Size**: ~29GB quantized (vs ~160GB unquantized)
- **Special Features**: Thinking tokens for chain-of-thought reasoning

## Memory Management

The project handles GPU memory carefully:
- Default allocation: 14GiB GPU memory (leaves headroom for inference)
- Uses `device_map="auto"` for automatic CPU/GPU distribution
- Float16 precision for activations

## Known Issues & Solutions

1. **GIL Warning**: The free-threaded Python build shows warnings for modules not GIL-free. This is expected and can be suppressed with `PYTHON_GIL=0`.

2. **Model Type Error**: If you see `qwen3_next` not recognized, ensure you're using Transformers from source (not the stable release).

3. **OOM Errors**: Adjust `max_memory` parameter in model loading:
   ```python
   max_memory={0: "12GiB"}  # Reduce if needed
   ```

## Code Patterns

- All scripts use similar model loading patterns with `AutoModelForCausalLM.from_pretrained()`
- Interactive loops follow the pattern: prompt → tokenize → generate → decode → display
- Error handling includes traceback printing for debugging
- Time measurements included for performance monitoring