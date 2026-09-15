"""
SmoothQuant Calibration-Efficient Post-Training Quantization.

A project to evaluate SmoothQuant for post-training quantization of LLMs
under resource-constrained settings.
"""

from .smoothquant import (QuantizedConv1d, QuantizedLinear,
                          SmoothQuantCalibrator, apply_smoothquant,
                          compute_perplexity, get_model_size,
                          measure_inference_latency, naive_quantize_model,
                          quantize_model_smoothquant)

__all__ = [
    "SmoothQuantCalibrator",
    "apply_smoothquant",
    "QuantizedLinear",
    "QuantizedConv1d",
    "quantize_model_smoothquant",
    "naive_quantize_model",
    "get_model_size",
    "measure_inference_latency",
    "compute_perplexity",
]

__version__ = "0.1.0"
