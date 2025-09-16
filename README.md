# Qwen3-NEXT-80B AutoRound

Local inference solution for running Qwen3-Next-80B with Intel's 4-bit AutoRound quantization on consumer hardware.

## Overview

This project provides a Python implementation for running the massive 80B parameter Qwen3-Next model on consumer GPUs using Intel's AutoRound quantization, reducing memory requirements from 160GB to ~29GB while maintaining model quality.

## Features

- ✅ **4-bit Quantized Inference** - Run 80B models with ~29GB memory footprint
- ✅ **Chain-of-Thought Support** - Parse and display the model's thinking process
- ✅ **Interactive Chat Mode** - Real-time conversational interface
- ✅ **Memory Efficient** - Automatic GPU memory management and CPU offloading
- 🚧 **OpenAI-Compatible API** - Coming soon

## Requirements

- Python 3.13+ (free-threaded build recommended)
- NVIDIA GPU with 16GB+ VRAM (24GB recommended)
- CUDA 12.1+
- ~30GB free disk space for model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PieBru/Qwen3-NEXT-80B_AutoRound.git
cd Qwen3-NEXT-80B_AutoRound
```

2. Create virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Authenticate with Hugging Face (required for model download):
```bash
huggingface-cli login
```

## Usage

### Basic Inference

Run the main inference script with interactive chat:

```bash
python run_qwen3_80b.py
```

### Chain-of-Thought Mode

To see the model's thinking process:

```bash
python qwen3_thinking.py
```

### Test Dependencies

Verify your environment is properly configured:

```bash
python test_deps.py
```

## Model Details

- **Model**: Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound
- **Size**: ~29GB (4-bit quantized)
- **Original Size**: ~160GB (unquantized)
- **Quantization**: Intel AutoRound 4-bit mixed precision
- **Special Features**: Thinking tokens for chain-of-thought reasoning

## Performance

On RTX 4090 (24GB VRAM):
- Loading time: 2-3 minutes
- Token generation: ~2-5 tokens/second
- Memory usage: 14-20GB GPU + system RAM for overflow

## Roadmap

- [x] Basic inference script
- [x] Chain-of-thought support
- [x] Interactive chat mode
- [ ] OpenAI-compatible API server
- [ ] Performance benchmarking suite
- [ ] Docker containerization
- [ ] Unit tests

## Note

This is a bridge solution while waiting for llama.cpp integration. Once Qwen3-Next is supported in llama.cpp, this project will provide migration guides and eventually be deprecated.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Intel for the AutoRound quantization technology
- Hugging Face for hosting the model
- The localllama community for inspiration and feedback