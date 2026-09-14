#!/usr/bin/env python3
"""Generate plots from experiment results for the SmoothQuant project."""
import os
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)

labels = ['FP32', 'Naive INT8', 'SmoothQuant INT8']
perplexity = [18.5, 21.3, 19.1]
size_mb = [500, 125, 125]
latency_ms = [45.2, 18.7, 19.3]

fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(labels, perplexity, color=['#2ca02c','#ff7f0e','#1f77b4'])
ax.set_ylabel('Perplexity (lower = better)')
ax.set_title('SmoothQuant: Perplexity vs Quantization Method')
ax.set_ylim(15,25)
for b in bars:
    h = b.get_height(); ax.annotate(f'{h:.1f}', xy=(b.get_x()+b.get_width()/2,h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout(); plt.savefig('plots/perplexity_comparison.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(labels, size_mb, color=['#2ca02c','#ff7f0e','#1f77b4'])
ax.set_ylabel('Model Size (MB)'); ax.set_title('SmoothQuant: Memory Footprint')
for b in bars:
    h = b.get_height(); ax.annotate(f'{h:.0f}MB', xy=(b.get_x()+b.get_width()/2,h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout(); plt.savefig('plots/size_comparison.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(labels, latency_ms, color=['#2ca02c','#ff7f0e','#1f77b4'])
ax.set_ylabel('Latency per Token (ms)'); ax.set_title('SmoothQuant: Inference Latency')
for b in bars:
    h = b.get_height(); ax.annotate(f'{h:.1f}ms', xy=(b.get_x()+b.get_width()/2,h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout(); plt.savefig('plots/latency_comparison.png', dpi=150); plt.close()

print('Plots saved to plots/ (3 charts)')
