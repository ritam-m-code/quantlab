"""Quantize GPT-2's 48 block matrices under each config; record per-layer and model-level metrics.

    python -m core.sweep                    # full 24-config grid (resumes: skips configs already on disk)
    python -m core.sweep --grid small       # 8-config fallback: no clipping, symmetric only
    python -m core.sweep --only fp32 b4-sym-pertensor-c100

Writes data/sweep/layers.json (config-independent) and data/sweep/configs/<id>.json.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from core.evaluate import generate, load_model, perplexity
from core.quant import dequantize, qrange, quantize_tensor

OUT = Path(__file__).resolve().parent.parent / "data" / "sweep"
N_BLOCKS = 12
TARGETS = [f"h.{i}.{m}" for i in range(N_BLOCKS) for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")]
GRANULARITIES = {"per-tensor": "pertensor", "per-channel": "perchan", "group-128": "g128"}
N_BINS = 256
HIST_TAIL = 0.0005  # histogram range covers the central 99.9%; the rest is counted in `outside`


def make_grid(size: str = "full") -> list[dict]:
    bits, sym, clip = [8, 4], [True, False], [100, 99.9]
    if size == "small":
        sym, clip = [True], [100]
    grid = []
    for b, s, g, c in itertools.product(bits, sym, GRANULARITIES, clip):
        cid = f"b{b}-{'sym' if s else 'asym'}-{GRANULARITIES[g]}-c{c:g}"
        grid.append(dict(id=cid, bits=b, symmetric=s, granularity=g, clip_pct=c))
    return grid


def quant_kwargs(cfg: dict) -> dict:
    """Map a config to quantize_tensor args. GPT-2 Conv1D is (in, out): output channels are axis 1."""
    g = cfg["granularity"]
    return dict(
        bits=cfg["bits"],
        symmetric=cfg["symmetric"],
        axis=None if g == "per-tensor" else 1,
        group_size=128 if g == "group-128" else None,
        clip_pct=cfg["clip_pct"],
    )


def scale_groups(t: torch.Tensor, granularity: str) -> torch.Tensor:
    """View t as (n_groups, group_len, n_channels): the elements sharing one scale lie along dim 1."""
    n_in, n_out = t.shape
    g = {"per-tensor": None, "per-channel": n_in, "group-128": 128}[granularity]
    return t.reshape(1, -1, 1) if g is None else t.reshape(n_in // g, g, n_out)


def levels_used(q: torch.Tensor, cfg: dict) -> float:
    """Mean over scale-groups of the fraction of available integer levels that occur in the group."""
    qmin, qmax = qrange(cfg["bits"], cfg["symmetric"])
    n_levels = qmax - qmin + 1
    idx = scale_groups(q - qmin, cfg["granularity"]).long()
    n_groups, _, n_chan = idx.shape
    group_id = torch.arange(n_groups).view(-1, 1, 1) * n_chan + torch.arange(n_chan).view(1, 1, -1)
    key = (group_id * n_levels + idx).flatten()
    occupied = torch.bincount(key, minlength=n_groups * n_chan * n_levels) > 0
    return occupied.view(-1, n_levels).float().mean().item()


def layer_metrics(w: torch.Tensor, cfg: dict, edges: torch.Tensor) -> tuple[dict, torch.Tensor, list[int]]:
    q, scale, zp = quantize_tensor(w, **quant_kwargs(cfg))
    w_hat = dequantize(q, scale, zp)
    qmin, qmax = qrange(cfg["bits"], cfg["symmetric"])
    raw = torch.round(w / scale) + zp
    m = {
        "mse": F.mse_loss(w_hat, w).item(),
        "cos": F.cosine_similarity(w.flatten(), w_hat.flatten(), dim=0).item(),
        "max_abs_err": (w_hat - w).abs().max().item(),
        "pct_clipped": 100 * ((raw < qmin) | (raw > qmax)).float().mean().item(),
        "levels_used": levels_used(q, cfg),
        "scale_quantiles": torch.quantile(scale.flatten(), torch.tensor([0, 0.25, 0.5, 0.75, 1.0])).tolist(),
    }
    if cfg["granularity"] == "per-tensor":  # one grid for the whole matrix: the frontend can draw it exactly
        m["scale"], m["zero_point"] = scale.item(), zp.item()
    whist = torch.histc(w_hat, bins=N_BINS, min=edges[0].item(), max=edges[-1].item())
    return m, w_hat, whist.long().tolist()


def target_modules(model) -> dict[str, Conv1D]:
    mods = {}
    for name in TARGETS:
        mod = model.transformer.get_submodule(name)
        # The gotcha: GPT-2 uses Conv1D with weight (in, out), not nn.Linear's (out, in).
        assert isinstance(mod, Conv1D) and mod.weight.shape[1] == mod.nf, name
        mods[name] = mod
    return mods


def layer_info(name: str, w: torch.Tensor) -> dict:
    lo, hi = torch.quantile(w.flatten(), torch.tensor([HIST_TAIL, 1 - HIST_TAIL])).tolist()
    r = max(abs(lo), abs(hi))  # symmetric range so 0 sits in the middle of the plot
    edges = torch.linspace(-r, r, N_BINS + 1)
    return {
        "id": name,
        "shape": list(w.shape),
        "min": w.min().item(),
        "max": w.max().item(),
        "hist": {
            "bins": edges.tolist(),
            "counts": torch.histc(w, bins=N_BINS, min=-r, max=r).long().tolist(),
            "outside": [(w < -r).sum().item(), (w > r).sum().item()],
        },
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["full", "small"], default="full")
    ap.add_argument("--only", nargs="*", help="config ids to run (fp32 for the baseline)")
    ap.add_argument("--force", action="store_true", help="rerun configs already on disk")
    args = ap.parse_args()

    model, tok = load_model()
    mods = target_modules(model)
    originals = {n: m.weight.detach().clone() for n, m in mods.items()}

    layers = [layer_info(n, w) for n, w in originals.items()]
    edges = {l["id"]: torch.tensor(l["hist"]["bins"]) for l in layers}
    write_json(OUT / "layers.json", {"gpt2": {"layers": layers}})

    configs = [dict(id="fp32")] + make_grid(args.grid)
    if args.only:
        known = {c["id"]: c for c in configs + make_grid("full")}
        configs = [known[i] for i in args.only]

    for cfg in configs:
        path = OUT / "configs" / f"{cfg['id']}.json"
        if path.exists() and not args.force:
            print(f"skip {cfg['id']} (on disk)")
            continue
        t0 = time.perf_counter()
        result, whists = dict(cfg), {}
        if cfg["id"] != "fp32":
            result["layers"] = {}
            for name, mod in mods.items():
                m, w_hat, whists[name] = layer_metrics(originals[name], cfg, edges[name])
                result["layers"][name] = m
                mod.weight.copy_(w_hat)
        t_quant = time.perf_counter() - t0
        result["perplexity"] = perplexity(model, tok)
        result["generations"] = generate(model, tok)
        result["whist"] = whists
        for name, mod in mods.items():
            mod.weight.copy_(originals[name])
        write_json(path, result)
        dt = time.perf_counter() - t0
        print(f"{cfg['id']:<24} ppl {result['perplexity']:9.2f}   {dt:5.1f}s (quant+metrics {t_quant:4.1f}s)", flush=True)


if __name__ == "__main__":
    main()
