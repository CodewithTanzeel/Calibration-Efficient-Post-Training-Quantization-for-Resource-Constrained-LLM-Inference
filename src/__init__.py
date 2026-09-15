"""
SmoothQuant Calibration-Efficient Post-Training Quantization.

A project to evaluate SmoothQuant for post-training quantization of LLMs
under resource-constrained settings.
"""

from .smoothquant import QuantizedConv1d
from .smoothquant import QuantizedLinear
from .smoothquant import SmoothQuantCalibrator
from .smoothquant import apply_smoothquant
from .smoothquant import compute_perplexity
from .smoothquant import get_model_size
from .smoothquant import measure_inference_latency
from .smoothquant import naive_quantize_model
from .smoothquant import quantize_model_smoothquant

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
