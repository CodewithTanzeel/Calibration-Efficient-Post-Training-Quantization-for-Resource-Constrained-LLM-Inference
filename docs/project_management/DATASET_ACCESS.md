# Dataset Access

## Status
- `wikitext-2` requires HuggingFace authentication (gated/private repository).
- Code fixed: `load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)` (`src/utils.py`).
- Synthetic calibration set: `data/synthetic/calibration.json` (reproducible, no auth).

## To unlock full experiments
- `huggingface-cli login`
- Or set `HF_TOKEN` environment variable
- Alternative: download `wikitext-2-raw-v1` manually to `data/raw/`

## Current pipeline
- Calibrator verified (`f5c4086`): tuple input, forward pass, smoothing computation.
- Tests: 3/5 pass (2 expected measurement/data-type failures resolved by synthetic data).
- Baseline experiment ready to run with synthetic data.
