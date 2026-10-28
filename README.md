# QuantLab

QuantLab is an interactive page that shows what happens to GPT-2 when you round its weights down to 8 or 4 bits.

Quantization stores each weight as a small integer on an evenly spaced grid instead of a 32-bit float. The big question
is how many weights should share one grid. With one grid for a whole matrix, 4-bit GPT-2 falls apart (perplexity goes
from 26 to about 5,900). With one grid per 128 weights, it barely changes (26–31). The page lets you flip between 24
settings and see why: the weight distribution before and after rounding, which layers take the most damage, and the
text the model writes.

## How it works

1. `core/quant.py` contains the rounding math: per-matrix, per-channel and per-128-weight grids, symmetric and
   asymmetric, with optional clipping.
2. `core/sweep.py` loads GPT-2 small, rounds its 48 block matrices under each setting, and records per-layer error,
   perplexity, and greedy text generations.
3. `core/export.py` writes the results to `web/public/data/` as static JSON.
4. `web/` is a small Vite + React page that reads that JSON. There's no backend, so it can be hosted anywhere static.

## Setup

You need Python 3.11+ and Node 18+.

**Python side** (only needed if you want to rerun the sweep; the results are already committed):

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy pytest

python -m pytest -q               # 16 tests for the quantization math
python -m core.sweep              # ~4 min on CPU, downloads GPT-2 (~500 MB) the first time
python -m core.export             # writes web/public/data/*.json
```

**Web side:**

```
cd web
npm install
npm run dev                       # local dev server
npm run build                     # static site in web/dist
```

**Deploying:** on Vercel, import the repo and set the Root Directory to `web`. It's detected as a Vite project with no
other settings needed.

## Scope

This is weight-only quantization: the stored weights are rounded, while the activations passing between layers stay
32-bit. Embeddings and the output layer aren't rounded. Rounding activations is a separate, harder problem (see
LLM.int8(), SmoothQuant, AWQ) and isn't covered here.
