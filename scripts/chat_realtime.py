#!/usr/bin/env python3
"""Real-time interactive chat with SmoothQuant-quantized gpt2 (via HF load)."""
import torch
from src.utils import load_model
from transformers import AutoTokenizer

def chat():
    print("Loading model (gpt2) via HF — this may take a moment...")
    model = load_model('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded. Type your prompt (enter to send, 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        inputs = tokenizer(user_input, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response:", decoded[len(user_input):].strip())

if __name__ == "__main__":
    chat()
