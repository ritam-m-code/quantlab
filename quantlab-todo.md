# QuantLab — interactive quantization explorer

Two-day build. Python core computes everything offline and exports JSON; the frontend
is static (no backend, no inference server, free hosting).

**Working mode:** VS Code, Claude as companion. Convention for this build — Claude
writes signatures, docstrings and tests; **I write the function bodies.** The core math
is ~15 lines and it is the entire point of the project. Anything scaffolding-shaped
(JSON plumbing, chart boilerplate, build config) is fair game to generate.

Location: `C:\Users\ritam\OneDrive\Desktop\tasks\QuantLab` (git repo, branch `main`).
Commits carry no Claude attribution. Vercel linking: me.

---

## 3-hour compression (2026-10-28)

| time | me | Claude |
|---|---|---|
| 0:00–0:15 | read quant.py docstrings + tests | venv, deps, gpt2 download; signatures + tests ✅ |
| 0:15–0:50 | **write quant.py bodies → pytest green** (Claude explains on request) | sweep / evaluate / export; Vite+TS+React scaffold on fake JSON |
| 0:50–1:00 | time 1 config + fp32 baseline; write predictions | if >40s/config, cut grid to 8 |
| 1:00–1:30 | full sweep in background | Panel 1 + Panel 3 |
| 1:30–2:20 | real JSON in; Panel 2 + perplexity readout | |
| 2:20–2:45 | push, Vercel link, README | |
| 2:45–3:00 | buffer, Review section | |

Cut: activation-outlier stretch (README still states weight-only vs activation).

Decisions locked:
- `axis` = channel axis (one scale per index). Conv1D per-output-channel → `axis=1`.
- Symmetric uses restricted range `[-(2^(b-1)-1), 2^(b-1)-1]`.
- Asymmetric clipping splits the tail mass: percentiles `(100-p)/2` and `(100+p)/2`.
- Perplexity: 2 non-overlapping 1024-token windows (GPT-2 context limit).
- Panel 1 overlays a 256-bin histogram of `w_hat` per layer x config (~1.5 MB) on the
  original histogram; exact grid lines drawn only for per-tensor.

---

## Repo layout

```
quantlab/
  core/
    quant.py          # the math. I write this.
    sweep.py          # loop over layers x configs, collect metrics
    evaluate.py       # perplexity + fixed-prompt generation
    hooks.py          # day-2 stretch: activation outlier capture
    export.py         # -> web/public/data/*.json
    tests/test_quant.py
  web/                # Vite + TS
  data/               # generated JSON (gitignored if large)
```

Deps: `torch` (CPU), `transformers`, `numpy`, `pytest`. All free.

---

## Day 1 — morning: the math

- [ ] `quantize_tensor(w, bits, symmetric, axis=None, group_size=None, clip_pct=100)`
      returns `(q, scale, zero_point)`
- [ ] `dequantize(q, scale, zero_point)` returns `w_hat`
- [ ] `fake_quantize(w, **cfg)` = quantize then dequantize, same shape as input

The four cases to get right:

| granularity | how |
|---|---|
| per-tensor | one scale for the whole matrix |
| per-channel | one scale per output channel (reduce over the input dim) |
| group-wise | split input dim into blocks of `group_size`, one scale per block |
| clipping | calibrate on the `clip_pct` percentile instead of true min/max |

Asymmetric (uint): `scale = (max-min)/(2**bits-1)`, `zp = round(qmin - min/scale)`.
Symmetric (int): `qmax = 2**(bits-1)-1`, `scale = absmax/qmax`, `zp = 0`.
Both: `q = clamp(round(w/scale) + zp, qmin, qmax)`.

### ⚠ The gotcha that will cost an hour if unflagged

GPT-2 does not use `nn.Linear`. It uses `transformers.pytorch_utils.Conv1D`, whose
weight is shaped **(in_features, out_features)** — transposed relative to Linear.
So "per output channel" means reducing over **dim=0** here, not dim=1. Get this
backwards and you silently compute per-*input*-channel scales, which look plausible
and behave wrong. Assert the shape before trusting any number.

- [ ] Tests (write these before the bodies):
  - [ ] round-trip of a tensor already on the grid is exact
  - [ ] `bits=8, per-tensor, symmetric` → max abs error <= scale/2
  - [ ] per-channel error <= per-tensor error on a matrix with one high-variance row
  - [ ] `group_size=in_features` is identical to per-channel
  - [ ] `clip_pct=100` clips nothing; `clip_pct=99` clips roughly 1%
  - [ ] shape and dtype preserved through `fake_quantize`

## Day 1 — afternoon: sweep + evaluate

- [ ] Load `gpt2` (124M, CPU). Target the 4 matrices per block: `c_attn`, `attn.c_proj`,
      `mlp.c_fc`, `mlp.c_proj` → 48 matrices. Leave embeddings and `lm_head` in fp32
      (note in the UI that this is a choice, not a law).
- [ ] Per layer per config, record: MSE, cosine similarity, max abs error,
      % values clipped, fraction of the 2**bits levels actually used
- [ ] Perplexity on one fixed ~2k-token passage — the single scalar "did it survive"
- [ ] Generation on 4 fixed prompts, **greedy (`do_sample=False`)** so results are
      reproducible and differences are real, not sampling noise

Config grid (24 total, ~15-20 min on CPU — check this estimate early):

```
bits:        [8, 4]
symmetric:   [True, False]
granularity: [per-tensor, per-channel, group-128]
clip_pct:    [100, 99.9]
```

768 and 3072 are both divisible by 128, so group-128 is clean.

- [ ] Prediction to write down *before* running, then check: which config is the first
      to break, and does per-layer cosine similarity rank the layers in the order that
      perplexity says it should?

## Day 1 — JSON schema (the contract)

Keep it a few MB, not a few hundred. Original weight histograms are
**config-independent** — store them once.

```jsonc
// layers.json
{ "gpt2": { "layers": [ { "id": "h.0.attn.c_attn", "shape": [768, 2304],
    "hist": { "bins": [...257 edges...], "counts": [...256...] } } ] } }

// configs.json
{ "configs": [ { "id": "b8-sym-perchan-c100", "bits": 8, "symmetric": true,
    "granularity": "per-channel", "clip_pct": 100,
    "perplexity": 29.4, "generations": { "prompt_0": "..." },
    "layers": { "h.0.attn.c_attn": { "mse": 1.2e-5, "cos": 0.9998,
      "max_abs_err": 0.004, "pct_clipped": 0.0, "levels_used": 0.87,
      "scale_quantiles": [...5 values...] } } } ] }
```

- [ ] Do **not** dump full per-channel scale arrays (48 layers x 24 configs x 2304
      floats). Store 5 quantiles per layer; the frontend only needs the spread.
- [ ] Baseline fp32 perplexity + generations stored as config id `"fp32"`.

## Day 2 — morning: frontend scaffold

- [ ] Vite + TS + React, static build, deploy target GitHub Pages or Vercel
- [ ] **Panel 1 — weight histogram with the quantization grid overlaid.**
      The money shot. Pick a layer, pick a config, see the grid lines land on the
      distribution. At `bits=4, per-tensor` the grid is visibly too coarse; flip to
      group-128 and it tightens. Reconstruct grid positions in the browser from
      `scale`/`zero_point` rather than shipping level arrays.
- [ ] Layer selector + config toggles wired to one piece of shared state

## Day 2 — afternoon: the rest, then ship

- [ ] **Panel 2 — per-layer sensitivity heatmap.** 48 layers x metric, so you can see
      whether damage concentrates in early blocks, late blocks, or the MLPs.
- [ ] **Panel 3 — generated text side by side.** fp32 vs selected config, same prompt.
      The visceral one: the text degrades as you turn the knobs down.
- [ ] Perplexity readout next to the toggles
- [ ] Deploy. Get the URL.
- [ ] README that states what the page teaches, in two sentences.

### Stretch (only if the above is done): activation outliers

Be precise about this, because it is where most write-ups get sloppy. Everything above
is **weight-only** quantization. The famous outlier problem is about **activations** —
a few channels run 10-100x larger than the rest, so one per-tensor activation scale
gets stretched to cover them and crushes everything else into two or three levels.
That is what LLM.int8(), SmoothQuant and AWQ are each attacking.

- [ ] Forward hooks capturing per-channel abs-max of each target layer's *input* over
      a few batches
- [ ] Panel 4 — per-channel activation magnitude, log scale, outlier channels labelled
- [ ] One sentence on the page connecting the picture to the three paper names

---

## Done means

- [ ] A live URL, not a localhost screenshot
- [ ] `pytest` green
- [ ] I can explain, without notes, why group size is the knob that matters
- [ ] The repo README distinguishes weight-only from activation quantization correctly

## Honest scoping notes

- This is a **teaching tool**, not infrastructure. That is a real product category and
  the only one where a two-day solo build can be best-in-class. Do not oversell it.
- The precompute-to-JSON decision is what makes the 2-day timeline safe and hosting
  free. Resist adding a live inference backend.
- If Day 1 afternoon slips, cut the grid to 8 configs (drop `clip_pct`, drop
  asymmetric). The histogram panel is the product; breadth is garnish.

## Review

_(fill in after build)_
