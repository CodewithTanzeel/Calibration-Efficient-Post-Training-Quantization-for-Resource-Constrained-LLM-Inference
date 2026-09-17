#!/usr/bin/env python3
"""Interactive CLI + session-logged chat with SmoothQuant/FP32. Real-time streaming, token counts, technique label."""
import os, sys, time, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer, TextStreamer
from src.utils import load_model
from src.smoothquant import quantize_model_smoothquant

SESSION_FILE = f"results/session_{datetime.date.today().isoformat()}.json"
os.makedirs('results', exist_ok=True)

class SessionLog:
    def __init__(self):
        self.turns = []
    def add(self, input_text, response_text, technique, alpha, input_tok, output_tok, time_s):
        self.turns.append({"t": len(self.turns)+1, "input": input_text, "response": response_text,
                           "technique": technique, "alpha": alpha,
                           "input_tokens": input_tok, "output_tokens": output_tok, "time_s": round(time_s,3)})
    def save(self):
        with open(SESSION_FILE, 'w') as f:
            json.dump({"date": datetime.date.today().isoformat(), "turns": self.turns}, f, indent=2)

def chat():
    print("=== SmoothQuant Real-Time Chat CLI ===")
    print("Techniques: fp32 | naive_int8 | smoothquant (alpha 0.5/0.7)")
    tech = input("Technique (default fp32): ").strip() or 'fp32'
    alpha = float(input("Alpha (default 0.5, use 0.7 for optimum): ").strip() or 0.5)
    session = SessionLog()
    print("Loading model (gpt2 via HF)...")
    base = load_model('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Apply technique
    if tech == 'smoothquant':
        from src.utils import load_calibration_data, get_random_examples
        calib_data = load_calibration_data('wikitext-2-raw-v1', max_samples=50)
        cal_samples = get_random_examples(calib_data, 10)
        model, _ = quantize_model_smoothquant(base, cal_samples, 10, alpha=alpha, weight_bit=8, act_bit=8)
        label = f"SmoothQuant INT8 (α={alpha})"
    elif tech == 'naive_int8':
        from src.smoothquant import naive_quantize_model
        dummy = [(torch.randint(0,1000,(1,10)), torch.randint(0,1000,(1,10)))]
        model = naive_quantize_model(base, dummy, 8, 8)
        label = "Naive INT8"
    else:
        model = base
        label = "FP32"

    stream_batch = input("Streaming live? (y/n, default y): ").strip() or 'y'
use_stream = stream_batch.lower().startswith('y')
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            session.save(); print("Session saved."); break
        inputs = tokenizer(user_input, return_tensors='pt', truncation=True, max_length=512)
        input_tok = inputs['input_ids'].shape[1]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7, streamer=streamer)
        time_s = time.time() - t0
        output_tok = outputs.shape[1] - input_tok
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(user_input):].strip()
        print(f"\n[Technique: {label} | Input tokens: {input_tok} | Output tokens: {output_tok} | Time: {time_s:.2f}s]\n")
        session.add(user_input, response_text, label, alpha, input_tok, output_tok, time_s)
        session.save()

if __name__ == "__main__":
    chat()
