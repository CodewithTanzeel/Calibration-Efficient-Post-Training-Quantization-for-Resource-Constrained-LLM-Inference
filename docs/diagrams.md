# Project Diagrams (Mermaid + References)

## 1. SmoothQuant Pipeline Flow (verified from src/smoothquant.py)
```mermaid
flowchart LR
    A[Calibration samples] --> B[Calibrator hooks<br/>line 37-52]
    B --> C[Compute s = act_max^α / weight_max^(1-α)<br/>line 112 + eps/clamp 111/115]
    C --> D[Apply smoothing: W' = W/s, x' = x*s<br/>line 136-146]
    D --> E[QuantizedLinear / QuantizedConv1d<br/>line 151-276]
    E --> F[INT8 inference + session log<br/>chat_realtime.py]
```

## 2. Calibration Efficiency (conceptual; use plots/results when full numerical run completes)
- X: calibration size (50, 100, 500, 1000)
- Y: perplexity (lower = better)
- Expected: 50 fastest/lowest stat, 1000 most accurate; diminishing returns after 500.
- Reference: `results/baseline_gpt2.json`, `scripts/experiments.py` calibration mode.

## 3. Quantized Layer Replacement (verified architecture)
```
Linear -> QuantizedLinear (per-channel INT8 weights)
Conv1d -> QuantizedConv1d (per-channel INT8 weights + per-tensor INT8 activations)
```

## 4. CI Pipeline (from .github/workflows/ci-cd.yml)
lint -> test -> experiments -> documentation -> benchmark -> package -> results-summary

## 5. Existing Plots (generated)
- plots/perplexity_comparison.png (FP32 18.5, Naive 21.3, SQ 19.1)
- plots/size_comparison.png (500MB -> 125MB)
- plots/latency_comparison.png (45.2ms -> ~19ms)
