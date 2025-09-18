from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Intel/Qwen3-Next-80B-A3B-Thinking-int4-mixed-AutoRound"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content) # no opening <think> tag
print("content:", content)
"""
thinking content: Okay, the user asked for a short introduction to large language models. Let me start by recalling what I know. LLMs are AI systems trained on massive text data. They generate human-like text. But I need to keep it concise.

First, define what an LLM is. Mention they're based on deep learning, specifically transformer architectures. Highlight the scale—billions of parameters. Explain their training process: unsupervised learning on huge datasets like books, articles, websites. 

Then, what they can do: answer questions, write stories, code, translate languages. But also note limitations—can make mistakes, lack true understanding. Maybe mention they're probabilistic, so outputs aren't always accurate.

Wait, the user might not know technical terms. Avoid jargon. Use simple terms. Instead of "transformer architecture," maybe say "advanced neural networks." But "transformer" is standard; maybe just say "specialized AI architecture" to simplify.

Key points: trained on vast text, predict next words, generate coherent text, versatile applications. But emphasize they don't understand like humans; they mimic patterns.

Also, common examples: ChatGPT, Gemini, Claude. But maybe not necessary for a short intro. The user might just want the basics.

Check if the response is too technical. Need to balance simplicity and accuracy. Maybe start with "Large language models (LLMs) are AI systems that..." then explain training, capabilities, limitations.

Structure: 
- Definition
- How they work (briefly)
- What they can do
- Limitations
- Common uses

But keep it very short. Maybe 3-4 sentences. User said "short," so avoid going into detail.

Possible structure:
"Large language models (LLMs) are advanced AI systems trained on massive amounts of text data. They learn to predict and generate human-like text by analyzing patterns in the data. LLMs can answer questions, write stories, code, and more, but they don't truly understand meaning and can sometimes produce inaccurate or biased outputs. Examples include models like ChatGPT, Gemini, and Claude."

That's concise. Check for accuracy. Yes, they are trained on text data, predict next tokens, generate text, have limitations. Examples are correct. 

Is "advanced AI systems" clear? Maybe "AI systems designed to understand and generate human language." But "advanced" is okay. 

Should I mention "deep learning" or "neural networks"? Maybe not necessary for a short intro. The term "LLM" itself implies that. 

Also, "probabilistic" might be too technical. Better to say "generate text based on patterns" or "predict likely next words."

Yes, "analyze patterns in the data" is good. 

Check for common misconceptions. Emphasize they don't understand, just mimic. That's important.

So final version: 

"Large language models (LLMs) are AI systems trained on vast text datasets to generate human-like responses. They predict the next word in a sequence based on patterns learned from data, enabling tasks like answering questions, writing, and coding. However, they lack true understanding and can sometimes produce incorrect or biased information. Popular examples include ChatGPT, Gemini, and Claude."

That's about 4 sentences. Good. 

Wait, the user might not know "predict the next word." Maybe simplify to "generate text by predicting likely words." 

Yes, "generate text by predicting the most probable next words" is accurate. 

Also, "vast text datasets" is fine. 

Yes. That should work.
</think>
content: Large language models (LLMs) are AI systems trained on massive amounts of text data to understand and generate human-like language. They predict the next word in a sequence based on patterns learned during training, enabling tasks like answering questions, writing stories, coding, and translating languages. However, they don’t "understand" meaning like humans—they mimic patterns and can sometimes produce inaccurate or biased outputs. Popular examples include ChatGPT, Gemini, and Claude.
"""
