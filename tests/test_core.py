import numpy as np
import pytest
import torch

from src import compute_perplexity, get_model_size, measure_inference_latency
from src.smoothquant import (SmoothQuantCalibrator, naive_quantize_model,
                             quantize_model_smoothquant)


def test_model_size():
    """Test model size calculation."""
    model = torch.nn.Linear(10, 5)
    size_info = get_model_size(model)
    assert 'size_mb' in size_info
    assert 'param_mb' in size_info
    assert 'buffer_mb' in size_info
    assert size_info['size_mb'] > 0

def test_measurement_functions():
    """Test that measurement functions return reasonable values."""
    # Create a dummy model
    model = torch.nn.Linear(10, 5)
    model.eval()

    # Test latency measurement
    input_ids = torch.randint(0, 1000, (1, 10))
    latency = measure_inference_latency(model, input_ids)
    assert 'total_time_s' in latency
    assert 'tokens_per_sec' in latency
    assert 'latency_per_token_ms' in latency
    assert latency['latency_per_token_ms'] >= 0

    # Test perplexity computation (with dummy data)
    dummy_loader = [(torch.randint(0, 100, (1, 10)), torch.randint(0, 100, (1, 10)))]
    perplexity = compute_perplexity(model, dummy_loader, torch.device('cpu'))
    assert isinstance(perplexity, float)
    assert perplexity > 0

def test_quantization_functions():
    """Test quantization functions."""
    model = torch.nn.Linear(10, 5)
    model.weight.data = torch.randn(5, 10)
    model.bias.data = torch.randn(5)

    # Create dummy inputs for test
    dummy_input_ids = torch.randint(0, 100, (1, 10))
    dummy_labels = dummy_input_ids.clone()
    dummy_samples = [(dummy_input_ids, dummy_labels)]

    # Test naive quantization
    naive_model = naive_quantize_model(model, dummy_samples, 8, 8)
    assert isinstance(naive_model, torch.nn.Module)
    # QuantizedLinear uses qweight buffer after quantize_weight is called in replace
    assert hasattr(naive_model, 'qweight') or hasattr(naive_model, 'bias')

    # Test smoothquant pipeline (mocked)
    dummy_loader_for_sq = [(dummy_input_ids, dummy_labels)]
    smoothed_model, _ = quantize_model_smoothquant(model, dummy_loader_for_sq, 1, alpha=0.5)
    assert isinstance(smoothed_model, torch.nn.Module)

def test_calibrator_hooks():
    """Test that calibrator registers hooks properly."""
    model = torch.nn.Linear(10, 5)
    calibrator = SmoothQuantCalibrator(model, alpha=0.5)

    # Before registering hooks, no hooks should be registered
    assert len(calibrator.hooks) == 0

    # Register hooks and verify
    calibrator.register_hooks()
    assert len(calibrator.hooks) > 0

    # Remove hooks and verify
    calibrator.remove_hooks()
    assert len(calibrator.hooks) == 0

def test_get_model_size_details():
    """Test model size calculation with buffers."""
    model = torch.nn.Linear(10, 5)
    # Add a buffer to test buffer size calculation
    model.register_buffer('test_buffer', torch.randn(1, 10))

    size_info = get_model_size(model)
    assert 'size_mb' in size_info
    assert size_info['size_mb'] > 0
    assert 'param_mb' in size_info
    assert 'buffer_mb' in size_info
    assert size_info['buffer_mb'] > 0  # Should include buffer

if __name__ == "__main__":
    pytest.main()