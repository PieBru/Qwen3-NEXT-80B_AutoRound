#!/usr/bin/env python3
"""
Run Qwen3-Next-80B with thinking capabilities
Based on the official HuggingFace example
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

print("=" * 60)
print(f"Loading Qwen3-Next-80B Thinking Model")
print("4-bit quantized version for efficiency")
print("=" * 60)

# Load the tokenizer and the model
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Tokenizer loaded")

print("\nLoading model (this will take several minutes)...")
print("Note: The model is ~58GB in 4-bit quantization")
print("Device mapping is slow on first run, faster on subsequent runs")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
print("✓ Model loaded successfully!")

def generate_with_thinking(prompt, max_tokens=32768):
    """Generate response with thinking content separated"""

    # Prepare the model input
    messages = [
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerating response...")
    start_time = time.time()

    # Conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    gen_time = time.time() - start_time

    return thinking_content, content, gen_time

# Test with example prompt
print("\n" + "=" * 60)
print("Example Generation")
print("=" * 60)

prompt = "Give me a short introduction to large language models."
print(f"\nPrompt: {prompt}")

thinking, response, duration = generate_with_thinking(prompt, max_tokens=512)

print(f"\nGeneration time: {duration:.2f}s")
print(f"\n{'='*40}")
print("THINKING CONTENT:")
print("="*40)
if thinking:
    print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
else:
    print("[No thinking content generated]")

print(f"\n{'='*40}")
print("RESPONSE:")
print("="*40)
print(response)

# Interactive mode
print("\n" + "=" * 60)
print("Interactive Mode - Type 'quit' to exit")
print("The model will show its thinking process separately from the response")
print("=" * 60)

while True:
    user_input = input("\n> ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        break

    thinking, response, duration = generate_with_thinking(user_input, max_tokens=1024)

    print(f"\n⏱ Generation time: {duration:.2f}s")

    if thinking:
        print(f"\n💭 THINKING:")
        print("-" * 40)
        # Show first 500 chars of thinking if it's long
        if len(thinking) > 500:
            print(thinking[:500] + "...[truncated]")
        else:
            print(thinking)

    print(f"\n📝 RESPONSE:")
    print("-" * 40)
    print(response)

print("\nGoodbye!")