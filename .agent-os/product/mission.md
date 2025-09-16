# Product Mission

> Last Updated: 2025-09-16
> Version: 1.0.0

## Pitch

Qwen3-80B AutoRound is a local inference solution that enables localllama enthusiasts to run the Qwen3-Next-80B model on consumer hardware using Intel's 4-bit AutoRound quantization while waiting for llama.cpp integration.

## Users

### Primary Customers

- **LocalLlama Community**: Developers and AI enthusiasts who run LLMs on their own hardware
- **Early Adopters**: Users wanting to experiment with Qwen3-Next-80B before broader tool support

### User Personas

**LocalLlama Enthusiast** (20-40 years old)
- **Role:** Developer/Researcher/Hobbyist
- **Context:** Runs LLMs locally for experimentation, privacy, or cost reasons
- **Pain Points:** Limited hardware resources, waiting for llama.cpp support, need for efficient quantization
- **Goals:** Run state-of-the-art models locally, explore chain-of-thought capabilities

## The Problem

### Hardware Limitations for Large Models

Running 80B parameter models requires significant GPU memory (160GB+ unquantized). Most consumer GPUs have 16-24GB VRAM, making these models inaccessible.

**Our Solution:** 4-bit AutoRound quantization reduces memory to ~29GB, enabling consumer GPU deployment.

### Gap in Tool Support

Qwen3-Next-80B lacks support in popular tools like llama.cpp, leaving users without inference options.

**Our Solution:** Provide a working Python implementation as a bridge solution until broader tool support arrives.

## Differentiators

### Intel AutoRound Quantization

Unlike standard quantization methods, we leverage Intel's AutoRound for superior 4-bit mixed precision, maintaining model quality while drastically reducing memory requirements. This results in accessible deployment on consumer RTX 4090 or similar GPUs.

### Chain-of-Thought Reasoning

We provide specialized support for the model's thinking tokens, allowing users to see and utilize the model's internal reasoning process, a feature not yet common in local inference tools.

## Key Features

### Core Features

- **4-bit Quantized Inference:** Run 80B models in ~29GB memory footprint
- **Interactive Chat Mode:** Real-time conversational interface with the model
- **Chain-of-Thought Support:** Separate parsing and display of thinking tokens
- **Memory Management:** Automatic GPU memory allocation and CPU offloading

### API Features

- **OpenAI-Compatible Server:** Drop-in replacement for OpenAI API (planned)
- **Batch Processing:** Efficient handling of multiple requests (planned)
- **Performance Benchmarking:** Token generation speed tracking and optimization