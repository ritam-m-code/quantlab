# QuantLab

An interactive explorer of weight quantization on GPT-2 (124M). It shows that rounding weights to 4 bits only works if
few enough weights share each grid: one scale per matrix destroys the model (perplexity 26 → ~5,900), while one scale
per 128 weights keeps it nearly intact (26–31).

## What it covers, and what it doesn't

This is **weight-only** quantization. The stored weights are rounded to an int grid and dequantized before use. The
activations flowing between layers stay fp32 throughout. Quantizing activations is a different, harder problem: a few
activation channels run 10–100× larger than the rest, so one shared activation scale crushes everything else into a
couple of levels. That is the problem LLM.int8(), SmoothQuant and AWQ each address, and it is not modelled here.

Quantized: the 48 block matrices (attention QKV + output, MLP up + down, × 12 blocks). Embeddings and the tied `lm_head`
stay fp32. That's a choice, not a requirement.

## Layout

```
core/quant.py      quantize / dequantize: per-tensor, per-channel, group-wise, percentile clipping
core/sweep.py      24 configs x 48 matrices -> per-layer metrics, perplexity, greedy generations
core/evaluate.py   perplexity on 2,048 tokens of Pride and Prejudice; 4 fixed prompts, greedy
core/export.py     -> web/public/data/*.json (static; no backend)
web/               Vite + React + TS
```

## Reproduce

```
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install transformers numpy pytest
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m core.sweep      # ~4 min on CPU
.venv\Scripts\python -m core.export
cd web && npm install && npm run dev
```
