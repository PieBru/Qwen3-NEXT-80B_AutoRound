#!/usr/bin/env python3
"""
ABSOLUTE MINIMAL LOADER - Based on Intel's official example
This is the simplest possible way to load the model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3-80B (simplest method)...")
print("This WILL take 30-60 minutes. Be patient!\n")

model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

# Just load it - no fancy options
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("\n✅ Model loaded! Testing...")

# Test
text = "What is 2+2?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nQ: {text}")
print(f"A: {result[len(text):]}")

print("\n🎉 Success! Model is working!")