"""
SmoothQuant Implementation for Post-Training Quantization of LLMs.

Based on the paper: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
by Xiao et al. (ICML 2023)

This implementation focuses on:
1. Migrating the quantization difficulty from weights to activations
2. Using a smoothing parameter α to balance the quantization difficulty
3. Applying per-channel quantization for weights and per-tensor quantization for activations
"""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SmoothQuantCalibrator:
    """
    Calibrator for collecting activation statistics for SmoothQuant.

    SmoothQuant works by:
    1. Collecting max activation values per channel across calibration samples
    2. Computing the smoothing factor: s = (max_activation)^α / (max_weight)^(1-α)
    3. Scaling weights by 1/s and activations by s
    """

    def __init__(self, model: nn.Module, alpha: float = 0.5):
        self.model = model
        self.alpha = alpha
        self.activation_max = defaultdict(list)
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to collect activation statistics."""
        def hook_fn(module, input, output, name):
            if isinstance(output, tuple):
                output = output[0]
            # Get max activation per channel (for the last dimension)
            if output.dim() >= 2:
                max_vals = output.abs().amax(dim=tuple(range(output.dim()-1)))
                self.activation_max[name].append(max_vals.detach().cpu())

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def calibrate(self, dataloader, num_samples: int):
        """Run calibration on a set of samples."""
        self.register_hooks()
        self.model.eval()

        with torch.no_grad():
            count = 0
            for batch in dataloader:
                if count >= num_samples:
                    break
                # Move batch to model device
                if isinstance(batch, dict):
                    batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = tuple(item.to(next(self.model.parameters()).device) for item in batch)
                else:
                    batch = batch.to(next(self.model.parameters()).device)

                # Run model forward (extract input tensor from tuple/dict)
                if isinstance(batch, dict):
                    _ = self.model(**batch)
                    count += batch.get('input_ids', batch).size(0)
                elif isinstance(batch, (list, tuple)):
                    _ = self.model(batch[0])
                    count += batch[0].size(0)
                else:
                    _ = self.model(batch)
                    count += batch.size(0)

        self.remove_hooks()
        return self.compute_smoothing_factors()

    def compute_smoothing_factors(self) -> Dict[str, torch.Tensor]:
        """Compute smoothing factors for each layer."""
        smoothing_factors = {}

        for name, module in self.model.named_modules():
            if name in self.activation_max and len(self.activation_max[name]) > 0:
                # Concatenate all collected max values and take the maximum
                all_max = torch.stack(self.activation_max[name]).amax(dim=0)

                # Get weight max per output channel
                if isinstance(module, nn.Linear):
                    weight_max = module.weight.data.abs().amax(dim=1)  # per output channel
                elif isinstance(module, nn.Conv1d):
                    weight_max = module.weight.data.abs().amax(dim=(1, 2))  # per output channel
                else:
                    continue

                # Compute smoothing factor: s = (act_max)^α / (weight_max)^(1-α)
                # Add small epsilon to avoid division by zero
                eps = 1e-8
                s = (all_max + eps).pow(self.alpha) / (weight_max + eps).pow(1 - self.alpha)

                # Clamp to reasonable range
                s = s.clamp(min=1e-4, max=1e4)
                smoothing_factors[name] = s

        return smoothing_factors


def apply_smoothquant(model: nn.Module, smoothing_factors: Dict[str, torch.Tensor],
                      alpha: float = 0.5) -> nn.Module:
    """
    Apply SmoothQuant smoothing to the model weights.

    For each layer:
    - Weight: W' = W / s (per output channel)
    - Activation: x' = x * s (per channel, applied during inference)
    """
    model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if name in smoothing_factors:
            s = smoothing_factors[name].to(module.weight.device)

            if isinstance(module, nn.Linear):
                # Scale weights by 1/s (per output channel)
                module.weight.data = module.weight.data / s.unsqueeze(1)
                if module.bias is not None:
                    module.bias.data = module.bias.data / s

            elif isinstance(module, nn.Conv1d):
                # Scale weights by 1/s (per output channel)
                module.weight.data = module.weight.data / s.view(-1, 1, 1)
                if module.bias is not None:
                    module.bias.data = module.bias.data / s

    return model


class QuantizedLinear(nn.Module):
    """Quantized Linear layer with INT8 weights and activations."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bit = weight_bit
        self.act_bit = act_bit

        # Quantized weight (INT8)
        self.register_buffer('qweight', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('weight_zero', torch.zeros(out_features))

        # Activation quantization parameters (per-tensor)
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero', torch.tensor(0))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight to INT8 per output channel."""
        # Per-channel quantization for weights
        w_max = weight.abs().amax(dim=1, keepdim=True)
        w_min = -w_max
        scale = (w_max - w_min) / (2**self.weight_bit - 1)
        zero = -w_min / scale

        # Clamp to INT8 range
        qweight = torch.clamp(torch.round(weight / scale + zero), -2**(self.weight_bit-1), 2**(self.weight_bit-1)-1)

        self.qweight.data = qweight.to(torch.int8)
        self.weight_scale.data = scale.squeeze(1)
        self.weight_zero.data = zero.squeeze(1)

    def quantize_activation(self, x: torch.Tensor):
        """Quantize activation to INT8 per tensor."""
        # Per-tensor quantization for activations
        x_max = x.abs().max()
        x_min = -x_max
        scale = (x_max - x_min) / (2**self.act_bit - 1)
        zero = -x_min / scale

        qx = torch.clamp(torch.round(x / scale + zero), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)

        return qx.to(torch.int8), scale, zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights and activations."""
        # Quantize activation
        qx, act_scale, act_zero = self.quantize_activation(x)

        # Dequantize weight for computation
        weight = (self.qweight.float() - self.weight_zero.unsqueeze(1)) * self.weight_scale.unsqueeze(1)

        # Compute output
        output = torch.nn.functional.linear(qx.float(), weight, self.bias)

        return output


class QuantizedConv1d(nn.Module):
    """Quantized Conv1d layer with INT8 weights and activations."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_bit = weight_bit
        self.act_bit = act_bit

        self.register_buffer('qweight', torch.zeros(out_channels, in_channels, kernel_size, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_channels))
        self.register_buffer('weight_zero', torch.zeros(out_channels))
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero', torch.tensor(0))

        if bias:
            self.register_buffer('bias', torch.zeros(out_channels))
        else:
            self.bias = None

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight to INT8 per output channel."""
        w_max = weight.abs().amax(dim=(1, 2), keepdim=True)
        w_min = -w_max
        scale = (w_max - w_min) / (2**self.weight_bit - 1)
        zero = -w_min / scale

        qweight = torch.clamp(torch.round(weight / scale + zero), -2**(self.weight_bit-1), 2**(self.weight_bit-1)-1)

        self.qweight.data = qweight.to(torch.int8)
        self.weight_scale.data = scale.squeeze(-1).squeeze(-1)
        self.weight_zero.data = zero.squeeze(-1).squeeze(-1)

    def quantize_activation(self, x: torch.Tensor):
        """Quantize activation to INT8 per tensor."""
        x_max = x.abs().max()
        x_min = -x_max
        scale = (x_max - x_min) / (2**self.act_bit - 1)
        zero = -x_min / scale

        qx = torch.clamp(torch.round(x / scale + zero), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)

        return qx.to(torch.int8), scale, zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights and activations."""
        qx, _, _ = self.quantize_activation(x)

        weight = (self.qweight.float() - self.weight_zero.view(-1, 1, 1)) * self.weight_scale.view(-1, 1, 1)

        output = torch.nn.functional.conv1d(qx.float(), weight, self.bias,
                                            self.stride, self.padding)

        return output


def replace_linear_with_quantized(model: nn.Module, weight_bit: int = 8, act_bit: int = 8) -> nn.Module:
    """Replace all Linear layers with QuantizedLinear layers."""
    model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model
            if parent_name:
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)

            quantized = QuantizedLinear(
                module.in_features, module.out_features,
                module.bias is not None, weight_bit, act_bit
            )
            quantized.quantize_weight(module.weight.data)
            if module.bias is not None:
                quantized.bias.data = module.bias.data

            setattr(parent, child_name, quantized)

    return model


def quantize_model_smoothquant(model: nn.Module, dataloader, num_samples: int,
                               alpha: float = 0.5, weight_bit: int = 8, act_bit: int = 8) -> nn.Module:
    """
    Full SmoothQuant quantization pipeline:
    1. Calibrate to get smoothing factors
    2. Apply smoothing to weights
    3. Replace layers with quantized versions
    """
    # Step 1: Calibrate
    calibrator = SmoothQuantCalibrator(model, alpha)
    smoothing_factors = calibrator.calibrate(dataloader, num_samples)

    # Step 2: Apply smoothing
    smoothed_model = apply_smoothquant(model, smoothing_factors, alpha)

    # Step 3: Replace with quantized layers
    quantized_model = replace_linear_with_quantized(smoothed_model, weight_bit, act_bit)

    return quantized_model, smoothing_factors


def naive_quantize_model(model: nn.Module, dataloader, num_samples: int,
                         weight_bit: int = 8, act_bit: int = 8) -> nn.Module:
    """
    Naive INT8 quantization without SmoothQuant smoothing.
    Just quantizes weights per-channel and activations per-tensor directly.
    """
    model = copy.deepcopy(model)
    return replace_linear_with_quantized(model, weight_bit, act_bit)


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return {'size_mb': size_mb, 'param_mb': param_size / (1024 ** 2), 'buffer_mb': buffer_size / (1024 ** 2)}


def measure_inference_latency(model: nn.Module, input_ids: torch.Tensor,
                              num_runs: int = 10, warmup: int = 3) -> Dict[str, float]:
    """Measure inference latency and throughput."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    # Measure
    import time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    total_time = end - start
    num_tokens = input_ids.numel()
    tokens_per_sec = (num_tokens * num_runs) / total_time
    latency_per_token = total_time / (num_tokens * num_runs) * 1000  # ms

    return {
        'total_time_s': total_time,
        'tokens_per_sec': tokens_per_sec,
        'latency_per_token_ms': latency_per_token
    }


def compute_perplexity(model: nn.Module, dataloader, device: torch.device) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
            else:
                input_ids = batch.to(device)
                labels = batch.to(device)

            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity