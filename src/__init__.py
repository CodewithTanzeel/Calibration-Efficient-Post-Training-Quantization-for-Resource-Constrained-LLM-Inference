"""
SmoothQuant Calibration-Efficient Post-Training Quantization.

A project to evaluate SmoothQuant for post-training quantization of LLMs
under resource-constrained settings.
"""

from .smoothquant import (
    SmoothQuantCalibrator,
    apply_smoothquant,
    QuantizedLinear,
    QuantizedConv1d,
    quantize_model_smoothquant,
    naive_quantize_model,
    get_model_size,
    measure_inference_latency,
    compute_perplexity,
)

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
