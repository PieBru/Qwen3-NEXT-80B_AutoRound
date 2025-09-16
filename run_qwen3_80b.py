#!/usr/bin/env python3
"""
Run Qwen3-Next-80B quantized model with AutoRound
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

    print("=" * 60)
    print(f"Loading model: {model_name}")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Load model with memory constraints
    print("\n2. Loading model (this may take several minutes)...")
    print("   Note: Using 4-bit quantization for memory efficiency")
    print("   Model size: ~58GB - downloading if not cached...")
    print("   Cache location: ~/.cache/huggingface/hub/")
    start_time = time.time()

    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"   GPU detected: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU VRAM: {gpu_memory:.1f} GB")

            # For GPUs with limited VRAM, use conservative settings
            if gpu_memory < 24:
                print("   Limited VRAM detected - will use CPU+GPU offloading")
                device_map = "auto"
                max_memory = {0: "12GiB", "cpu": "60GiB"}  # ~58GB model needs <64GB RAM
            else:
                device_map = "auto"
                max_memory = {0: "20GiB"}  # More generous for larger GPUs
        else:
            print("   No GPU detected - will use CPU (slower but works)")
            device_map = "cpu"
            max_memory = None

        print("   Starting model load - this is SLOW for 80B model!")
        print("   The model has 80B parameters - device mapping takes time...")
        print("   (This step is faster on subsequent runs)")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"   ✓ Model loaded in {time.time() - start_time:.2f}s")

        # Model info
        print("\n3. Model information:")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Quantization: 4-bit mixed precision (AutoRound)")

        # Simple inference test
        print("\n4. Testing inference...")
        prompt = "What is artificial intelligence?"

        print(f"   Prompt: '{prompt}'")
        print("   Generating response...")

        inputs = tokenizer(prompt, return_tensors="pt")
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

        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Tokens/second: {50/gen_time:.2f}")

        print("\n5. Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        print("\n✓ Model successfully loaded and tested!")

        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive mode - Type 'quit' to exit")
        print("=" * 60)

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            inputs = tokenizer(user_input, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("\nGenerating response...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Time: {time.time() - start_time:.2f}s")
            print("\nResponse:")
            print(response)

    except Exception as e:
        print(f"\n✗ Error loading or running model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    exit(main())