#!/usr/bin/env python3
"""
Qwen3-Next-80B CPU-Only Runner
Optimized for systems with limited GPU memory
"""

import torch
import time
import gc
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig
import warnings
import traceback

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

def print_system_info():
    """Print system memory information."""
    print("\n🔧 System Information:")

    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    print(f"   CPU: {cpu_count} logical threads")

    # RAM info
    ram = psutil.virtual_memory()
    print(f"   RAM: {ram.total / (1024**3):.1f}GB total, {ram.available / (1024**3):.1f}GB available")

    # GPU info (if available)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {vram:.1f}GB")
        print("   ⚠️  Note: GPU will NOT be used in CPU-only mode")

    print()

def main():
    try:
        print("="*60)
        print("Qwen3-Next-80B CPU-Only Mode")
        print("="*60)

        print_system_info()

        print("⚠️  Important Notes:")
        print("   - This will use CPU-only mode (no GPU acceleration)")
        print("   - Requires ~40-50GB RAM for model + inference")
        print("   - Inference will be slower than GPU mode")
        print("   - Recommended: 64GB+ RAM for comfortable operation")
        print()

        # Check available RAM
        ram_available = psutil.virtual_memory().available / (1024**3)
        if ram_available < 45:
            print(f"⚠️  Warning: Only {ram_available:.1f}GB RAM available")
            print("   Model may fail to load or run very slowly")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return

        # Load tokenizer
        print("1. Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"   ✓ Tokenizer loaded in {time.time() - start_time:.1f}s\n")

        # Configure for CPU-only
        print("2. Configuring for CPU-only mode...")
        device_map = "cpu"

        # Load model
        print("3. Loading model (CPU-only, this will take time)...")
        print("   Progress: Loading checkpoint shards...")
        print("   💡 CPU inference is slower but more stable for large models")

        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start_time
        print(f"\n   ✓ Model loaded successfully in {load_time/60:.1f} minutes")

        # Memory usage
        ram_used = psutil.Process().memory_info().rss / (1024**3)
        print(f"   Memory used: {ram_used:.1f}GB")

        # Test generation
        print("\n" + "="*60)
        print("Testing Model Generation")
        print("="*60)

        test_prompt = "What is the capital of France?"
        print(f"\nTest prompt: {test_prompt}")

        messages = [{"role": "user", "content": test_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")

        print("Generating response (this may take a minute on CPU)...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        gen_time = time.time() - start_time
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(f"\nResponse: {response}")
        print(f"Generation time: {gen_time:.1f}s")

        # Interactive mode
        print("\n" + "="*60)
        print("Interactive Chat Mode (CPU)")
        print("Type 'quit' or 'exit' to end the conversation")
        print("="*60 + "\n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages = [{"role": "user", "content": user_input}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt")

            print("\nGenerating (CPU mode - please wait)...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            gen_time = time.time() - start_time
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()

            print(f"\nAssistant: {response}")
            print(f"[Generated in {gen_time:.1f}s]")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nDetailed error:")
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Ensure you have 50GB+ free RAM")
        print("   2. Close other memory-intensive applications")
        print("   3. Consider using a system with more RAM")
        print("   4. Try the standard GPU version if you have sufficient VRAM")

if __name__ == "__main__":
    main()