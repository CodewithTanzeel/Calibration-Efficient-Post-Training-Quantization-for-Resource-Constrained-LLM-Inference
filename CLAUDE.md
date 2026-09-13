# Calibration-Efficient Post-Training Quantization for Resource-Constrained LLM Inference

## Project Overview
This project implements and evaluates **SmoothQuant** for post-training quantization of large language models (LLMs) under resource-constrained settings, with a focus on calibration efficiency and CPU inference performance.

### Main Research Question
> **Can SmoothQuant preserve language-model and reasoning performance when only a very small calibration set is available and inference is performed on CPU-constrained hardware?**

### Key Contributions

1. **Reproduce the baseline**: Compare FP32, naive INT8, and SmoothQuant INT8 quantization
2. **Study calibration efficiency**: Evaluate different calibration-set sizes (50, 100, 500, 1000 samples)
3. **Optimize smoothing parameter**: Test α ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
4. **Evaluate reasoning separately**: Use GSM8K subset to measure reasoning accuracy degradation
5. **Benchmark on CPU hardware**: Report RAM usage, tokens/sec, latency, perplexity, reasoning accuracy

### Project Structure
- `src/` - Core implementation (SmoothQuant, quantization, utilities)
- `experiments/` - Experiment scripts and ablation studies
- `results/` - Experiment results and visualizations
- `data/` - Datasets and checkpoints
- `notebooks/` - Jupyter notebooks for analysis
- `tests/` - Unit tests and validation scripts
- `docs/` - Documentation and technical reports

## Implementation Details

### Technical Approach

1. **SmoothQuant Implementation**:
   - Collect activation statistics during calibration
   - Compute per-layer smoothing factors: $s = \text{act\_max}^{\alpha} / \text{weight\_max}^{1-\alpha}$
   - Scale weights by $1/s$ and apply smoothing during inference
   - Support for both Linear and Conv1d layers

2. **Quantization Strategy**:
   - Per-channel quantization for weights (INT8)
   - Per-tensor quantization for activations (INT8)
   - Custom quantized layers with efficient forward passes

3. **Experimental Design**:
   - Multiple calibration sizes (50, 100, 500, 1000)
   - Random vs. selected calibration examples
   - Comprehensive benchmarking on CPU
   - Reasoning-specific evaluation with GSM8K

### Libraries and Frameworks
- PyTorch ≥2.0.0
- Transformers ≥4.30.0
- NumPy, SciPy for numerical computations
- Matplotlib, Seaborn for visualization

## Research Questions and Hypotheses

### Primary Research Question
How does SmoothQuant performance scale with decreasing calibration set size on CPU-constrained hardware?

### Specific Questions
1. Does SmoothQuant maintain better accuracy than naive quantization with small calibration sets?
2. What is the optimal α parameter for different calibration sizes?
3. Does carefully selected calibration data outperform random sampling?
4. How does quantization affect reasoning tasks compared to language modeling?
5. What are the practical performance trade-offs for CPU inference?

### Expected Findings
- SmoothQuant can maintain >80% of FP32 accuracy with as few as 50 calibration samples
- Random calibration examples may be sufficient for small models
- Reasoning performance degrades more than language modeling accuracy
- Smooth parameter α ≈ 0.7 may provide optimal trade-offs

## Setup Instructions

### Prerequisites
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Clone SmoothQuant baseline code (if not already included):
   ```bash
   git submodule add https://github.com/mit-han-lab/smoothquant.git
   ```

### Running Experiments
1. Set up your model and data paths in `experiments/config.py`
2. Run baseline experiments:
   ```bash
   python experiments/run_baseline.py
   ```
3. Run ablation studies:
   ```bash
   python experiments/run_ablation.py --calibration-sizes 50 100 500 1000
   ```

### Result Analysis
```bash
# Generate visualizations
python scripts/visualize_results.py

# Create comprehensive report
python scripts/generate_report.py
```

## Project Timeline

### Phase 1: Foundation (Week 1-2)
- [x] Set up project structure and issue templates
- [x] Implement core SmoothQuant quantization
- [x] Add basic model loading and quantization utilities
- [x] Create reproduction script for baseline

### Phase 2: Baseline Experiments (Week 3-4)
- [ ] Run FP32 vs. naive vs. SmoothQuant comparisons
- [ ] Study calibration efficiency across different sizes
- [ ] Collect comprehensive performance metrics
- [ ] Create initial result visualizations

### Phase 3: Optimization Studies (Week 5-6)
- [ ] Optimize SmoothQuant smoothing parameter α
- [ ] Study calibration example selection strategies
- [ ] Evaluate reasoning vs. language modeling performance
- [ ] Analyze trade-offs and create recommendations

### Phase 4: Final Analysis (Week 7-8)
- [ ] Run comprehensive benchmarks on CPU hardware
- [ ] Validate findings with multiple random seeds
- [ ] Write technical report and analysis
- [ ] Prepare documentation and reproducibility package

### Phase 5: Publication Preparation (Week 9-10)
- [ ] Create final GitHub repository
- [ ] Prepare paper and supplementary materials
- [ ] Setup CI/CD pipeline
- [ ] Publish results and code

## Development Workflow

### Code Quality Standards
1. **Documentation**: Every function, class, and method must have docstrings
2. **Testing**: Comprehensive test coverage for all new functionality
3. **Reproducibility**: All experiments must be reproducible with fixed random seeds
4. **Performance**: Profiling and optimization of critical paths
5. **Code Style**: PEP 8 compliant with type hints

### Branch Management
- `main` branch: Contains stable, tested code
- Feature branches: Each issue/task on its own branch
- PRs must pass all tests before merge

### Issue Management
- Use GitHub issue templates for new requests
- Each task should be documented as a GitHub issue
- PR descriptions should link to related issues

## Technical Details

### SmoothQuant Formula
For each layer with weight matrix $W \in \mathbb{R}^{out \times in}$ and activation tensor $x$:
1. Compute per-channel activation max: $\text{act\_max}_c = \max(x_c)$
2. Compute per-channel weight max: $\text{weight\_max}_c = \max(|W_c|)$
3. Calculate smoothing factor: $s_c = (\text{act\_max}_c)^{\alpha} / (\text{weight\_max}_c)^{1-\alpha}$
4. Apply smoothing: $W' = W / s_c$, $x' = x \cdot s_c$

### Quantization Configuration
- Weight bit-width: 8 bits (INT8)
- Activation bit-width: 8 bits (INT8)
- Calibration size: Variable (50-1000 samples)
- Random seed: Fixed for reproducibility

### Evaluation Metrics
1. **Language Modeling**: Perplexity on validation set
2. **Reasoning**: Exact-match accuracy on GSM8K
3. **Performance**: Tokens per second, latency per token
4. **Memory**: Peak RAM usage during inference
5. **Efficiency**: Calibration samples vs. accuracy trade-offs

## Future Work

### Extensions
1. **Dynamic calibration**: Adaptive calibration based on model properties
2. **Mixed precision**: Combine with other quantization techniques
3. **Hardware optimizations**: Integration with CPU-specific optimizations
4. **Real-world datasets**: Evaluation on diverse datasets

### Research Directions
1. **Theoretical analysis**: Formal guarantees for SmoothQuant on language models
2. **Scaling laws**: Understanding scaling behavior with model size
3. **Transfer learning**: Applying quantization knowledge across models
4. **Resource constraints**: Generalizing to different hardware platforms

## Reproduction Instructions

### Complete Setup
1. Clone this repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Run all experiments: `python -m experiments.full_pipeline`
4. Generate final report: `python scripts/final_report.py`

### Verification
Each experiment result should be verifiable by:
1. Running with fixed random seeds
2. Checking output files in `results/` directory
3. Validating plots and metrics against expectations
4. Cross-referencing with analysis notebooks

## Acknowledgments

Special thanks to:
- The SmoothQuant authors for providing the baseline implementation
- The PyTorch and HuggingFace communities for excellent tools
- MIT-Han Lab for the SmoothQuant paper and reference code
- All contributors and reviewers for feedback and suggestions

## Contact

For questions, issues, or collaboration opportunities:
- GitHub Issues: [Open an issue](https://github.com/[your-username]/[repository-name]/issues)
- Email: [your-email]
- Project Repository: [Link to GitHub repo]

---

*Last updated: $(date -I)*
*Version: 0.1.0*