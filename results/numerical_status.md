# Numerical Status — Post-Workflow Verification
Dataset: wikitext-2-raw-v1 parquet (36,718 rows) — real, loaded via src/utils.py
Calibration pipeline: structurally verified (calibrator hook, tuple fix, smoothing factor computation, quantized layers)
Results: baseline_gpt2.json present — values approximated (FP32=18.5, Naive=21, SQ=19); SmoothQuant size=null
Blocker: Python executable unavailable on host — model load / perplexity computation not executed
Next: run `python scripts/experiments.py --mode baseline --calibration-size 50` with full env to replace approximations
Co-Authored-By: Claude Code <noreply@anthropic.com>
