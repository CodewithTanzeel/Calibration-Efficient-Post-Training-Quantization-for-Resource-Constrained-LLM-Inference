# Calibration-Efficient Post-Training Quantization for Resource-Constrained LLM Inference

This project implements and evaluates SmoothQuant for post-training quantization of large language models (LLMs) under resource-constrained settings, particularly focusing on calibration efficiency and CPU inference performance.

## Base Paper
SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
Official code: mit-han-lab/smoothquant

## Main Research Question
Can SmoothQuant preserve language-model and reasoning performance when only a very small calibration set is available and inference is performed on CPU-constrained hardware?

## Proposed Contributions
1. **Reproduce the baseline**: Compare FP32 model, naive INT8 quantization, and SmoothQuant INT8 quantization starting with GPT-2 small or OPT-125M.
2. **Study calibration efficiency**: Test different calibration-set sizes (50, 100, 500, 1000 samples) and investigate whether carefully selected calibration examples perform better than random examples.
3. **Optimize the smoothing parameter**: Test different values of the SmoothQuant parameter (α = 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) and analyze the trade-off between perplexity, memory usage, inference latency, and quantization error.
4. **Evaluate reasoning separately**: Use a small reasoning dataset or a subset of GSM8K to measure whether quantization affects normal language modeling, arithmetic reasoning, exact-match accuracy, output length, and error types.
5. **Benchmark on realistic CPU hardware**: Report peak RAM usage, model size, tokens per second, latency per generated token, perplexity, and reasoning accuracy.

## How to Make It Impressive
- Clean reproducible code
- Proper baseline
- Ablation studies
- Multiple random seeds where possible
- Tables and plots
- Error analysis
- Limitations section
- Short technical report
- GitHub repository with exact commands to reproduce results

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the baseline experiments: `python experiments/run_baseline.py`
4. Follow the instructions in the `experiments` directory for ablation studies.

## License
MIT